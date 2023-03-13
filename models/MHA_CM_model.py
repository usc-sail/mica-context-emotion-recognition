#### contains the layer definition consisting of cross-modal operation 
#define the cross modal layer consisting of multi head attention operation and then composition of individual operations 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import copy
from transformers import BertModel
from position_encoding import PositionalEncoding
from torchvision import models

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
class CrossModalLayer(nn.Module):

    def __init__(self, dim_model=512, 
                dim_feedforward=2048, 
                num_heads=8, drop_prob=0.2, 
                add_bias=True, 
                activation="relu", 
                batch_first=True,
                add_pos=True):

        super(CrossModalLayer, self).__init__()
        
        #basic declarations
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation

        #linear layers and dropout values 
        self.linear1 = nn.Linear(self.dim_model, self.dim_feedforward)
        self.dropout = nn.Dropout(self.drop_prob)
        self.linear2 = nn.Linear(self.dim_feedforward, self.dim_model)

        #layer norm and dropout layer declarations
        self.norm1 = nn.LayerNorm(self.dim_model)
        self.norm2 = nn.LayerNorm(self.dim_model)
        self.dropout1 = nn.Dropout(self.drop_prob)
        self.dropout2 = nn.Dropout(self.drop_prob)

        self.MHA_unit=nn.MultiheadAttention(  #declaration of MHA unit 
                        embed_dim=self.dim_model, 
                        num_heads=self.num_heads, 
                        dropout=self.drop_prob, 
                        bias=self.add_bias, 
                        batch_first=self.batch_first)

        self.activation = _get_activation_fn(activation) #activation function

    # def with_pos_embed(self, tensor, pos):
    #     if(pos is None):

    #     return tensor if pos is None else tensor + pos

    def forward (self, query, key, value, src_key_mask):

        # #add positional embedding to query and key
        # if(self.add_pos):
        #     query=self.with_pos_embed(query, pos_query)
        #     key=self.with_pos_embed(key, pos_key)
        #     value=self.with_pos_embed(value, pos_key)

        #MHA operation
        attn_output, attn_output_weights = self.MHA_unit(query, key, value, key_padding_mask=src_key_mask)
        #attn_output gives attention output and attn_output_weights gives attention weights

        #add and norm
        src = query + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_output_weights

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#create an encoder operation 
class CrossModalEncoder(nn.Module):

    def __init__(self, dim_model=512, 
                dim_feedforward=2048, 
                num_heads=8, 
                drop_prob=0.2, 
                add_bias=True, 
                activation="relu", 
                batch_first=True,
                num_layers=4):


        super(CrossModalEncoder, self).__init__()
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.num_layers=num_layers
        print(self.batch_first)

        self.layer = CrossModalLayer(dim_model=self.dim_model, 
                                    dim_feedforward=self.dim_feedforward, 
                                    num_heads=self.num_heads, 
                                    drop_prob=self.drop_prob, 
                                    add_bias=self.add_bias, 
                                    activation=self.activation, 
                                    batch_first=self.batch_first)

        self.layers = _get_clones(self.layer, self.num_layers)

    def forward(self, query, key, value, src_key_mask):

        output = query
        attn_weights = []

        for layer in self.layers:
            output, attn_weight = layer(output, key, value, src_key_mask)
            attn_weights.append(attn_weight)

        return output, attn_weights

#need to decode the format for src_key_padding_mask

        
#define the cross modal fusion model here with different small models 
class caption_person_MHA_model_scene_late_fusion_AVD_Discrete(nn.Module):

    def __init__(self, dim_model,
                dim_feedforward,
                num_heads,
                drop_prob,
                add_bias,
                activation,
                batch_first, 
                num_layers,
                scene_feat_dim, 
                person_model_option,
                text_model,
                person_max_len,
                text_max_len,
                num_discrete_classes,
                num_cont_classes,
                add_pos
                ):

        super(caption_person_MHA_model_scene_late_fusion_AVD_Discrete, self).__init__()


        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.scene_feat_dim=scene_feat_dim
        self.text_model=text_model
        self.person_model_option=person_model_option
        self.person_max_len=person_max_len
        self.text_max_len=text_max_len
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.num_layers=num_layers
        self.add_pos=add_pos
        self.cls_feat_dim=self.dim_model+self.scene_feat_dim


        #text feature map layer 
        self.text_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person text model 
        self.person_text_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        #initialize position embeddings for query, key and value
        self.pos_query=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.person_max_len)
        self.pos_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.text_max_len)

    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0]
        text_feat=self.text_feature_map_layer(text_feat) #(converting the dimensionality to 512)
        text_mask=~text_mask #inverting the mask for nn.MultiheadAttention

        #text feature map layer
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)
        
        #person text fusion model   
        if(self.add_pos):
            inp_person_feat =   inp_person_feat.permute(1,0,2)
            inp_person_feat = self.pos_query(inp_person_feat) # (49,B,512)
            inp_person_feat = inp_person_feat.permute(1,0,2) # (B,49,512)

            text_feat =   text_feat.permute(1,0,2)
            text_feat = self.pos_key(text_feat) # (49,B,512)
            text_feat = text_feat.permute(1,0,2) # (B,49,512)
        
            # reshape to (max_len , batch size, dim_model)
        #text guided person fusion
        text_guided_person_feat,attn_weights=self.person_text_model(inp_person_feat,text_feat,text_feat,text_mask)

        text_guided_feat_avg=text_guided_person_feat.mean(dim=1) #average pooling over the time dimension
        scene_feat_cls=inp_scene_feat[:,0,:] #cls token feature for scene

        ov_feat=torch.cat([text_guided_feat_avg,scene_feat_cls],dim=1)

        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation
 
        return(logits_cont,logits_discrete)


class caption_person_dual_MHA_model_scene_late_fusion_AVD_Discrete(nn.Module):

    def __init__(self, dim_model,
                dim_feedforward,
                num_heads,
                drop_prob,
                add_bias,
                activation,
                batch_first, 
                num_layers,
                scene_feat_dim, 
                person_model_option,
                text_model,
                person_max_len,
                scene_max_len,
                text_max_len,
                num_discrete_classes,
                num_cont_classes,
                add_pos
                ):

        super(caption_person_dual_MHA_model_scene_late_fusion_AVD_Discrete, self).__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.scene_feat_dim=scene_feat_dim
        self.text_model=text_model
        self.person_model_option=person_model_option
        self.person_max_len=person_max_len
        self.text_max_len=text_max_len
        self.scene_max_len=scene_max_len
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.num_layers=num_layers
        self.add_pos=add_pos
        self.cls_feat_dim=2*self.dim_model

        #text feature map layer
        self.text_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)
        self.scene_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person text model 
        self.person_text_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        self.person_scene_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        #initialize position embeddings for query, key and value
        self.pos_person_query=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.person_max_len)
        self.pos_text_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.text_max_len)
        self.pos_scene_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.scene_max_len)

    def forward(self, image_data, inp_scene_feat, input_ids, text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0]
        text_feat=self.text_feature_map_layer(text_feat) #(converting the dimensionality to 512)
        text_mask=~text_mask #inverting the mask for nn.MultiheadAttention

        #text feature map layer
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)

        #scene feature map layer
        inp_scene_feat=self.scene_feature_map_layer(inp_scene_feat) #(B,49,512)

        #person text fusion model   
        if(self.add_pos):

            inp_person_feat =   inp_person_feat.permute(1,0,2)
            inp_person_feat = self.pos_person_query(inp_person_feat) # (49,B,512)
            inp_person_feat = inp_person_feat.permute(1,0,2) # (B,49,512)

            text_feat =   text_feat.permute(1,0,2)
            text_feat = self.pos_text_key(text_feat) # (49,B,512)
            text_feat = text_feat.permute(1,0,2) # (B,49,512)

            inp_scene_feat =   inp_scene_feat.permute(1,0,2)
            inp_scene_feat = self.pos_scene_key(inp_scene_feat) # (49,B,512)
            inp_scene_feat = inp_scene_feat.permute(1,0,2) # (B,49,512)

        person_text_feat,_=self.person_text_model(inp_person_feat,text_feat,text_feat,text_mask) #(B,49,512)

        #person scene fusion model
        person_scene_feat,_=self.person_scene_model(inp_person_feat,inp_scene_feat,inp_scene_feat,None) #(B,49,512)

        #avg both and concatenate 
        person_text_feat_avg=person_text_feat.mean(dim=1) #(B,512)
        person_scene_feat_avg=person_scene_feat.mean(dim=1) #(B,512)

        ov_feat=torch.cat((person_text_feat_avg,person_scene_feat_avg),dim=1) #(B,1024)

        #classification layer
        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation

        #print(logits_cont.shape,logits_discrete.shape)
        return logits_cont,logits_discrete
        

class person_MHA_model_scene_AVD_Discrete(nn.Module):

    def __init__(self, dim_model,
                dim_feedforward,
                num_heads,
                drop_prob,
                add_bias,
                activation,
                batch_first,
                num_layers,
                scene_feat_dim, 
                person_model_option,
                person_max_len,
                scene_max_len,
                num_discrete_classes,
                num_cont_classes,
                add_pos
                ):

        super(person_MHA_model_scene_AVD_Discrete, self).__init__() 

        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.scene_feat_dim=scene_feat_dim
        self.person_model_option=person_model_option
        self.person_max_len=person_max_len
        self.scene_max_len=scene_max_len
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.add_pos=add_pos
        self.num_layers=num_layers
        self.cls_feat_dim=self.dim_model
        
        self.scene_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)

        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        self.person_scene_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        
        self.pos_person_query=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.person_max_len)
        self.pos_scene_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.scene_max_len)

    def forward(self, image_data, inp_scene_feat):
        
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)

        #scene feature map layer
        inp_scene_feat=self.scene_feature_map_layer(inp_scene_feat) #(B,49,512)

        #person text fusion model   
        if(self.add_pos):
            inp_person_feat =   inp_person_feat.permute(1,0,2)
            inp_person_feat = self.pos_person_query(inp_person_feat) # (49,B,512)
            inp_person_feat = inp_person_feat.permute(1,0,2) # (B,49,512)

            inp_scene_feat =   inp_scene_feat.permute(1,0,2)
            inp_scene_feat = self.pos_scene_key(inp_scene_feat) # (49,B,512)
            inp_scene_feat = inp_scene_feat.permute(1,0,2) # (B,49,512)

        #person scene fusion model
        person_scene_feat,_=self.person_scene_model(inp_person_feat,inp_scene_feat,inp_scene_feat,None) #(B,49,512)

        person_scene_feat_avg=person_scene_feat.mean(dim=1) #(B,512)

        #classification layer
        logits_cont=self.classifier_continuous(person_scene_feat_avg) #continuous logits
        logits_discrete=self.classifier_discrete(person_scene_feat_avg) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation

        return logits_cont,logits_discrete

class caption_person_MHA_model_text_AVD_Discrete(nn.Module):

    def __init__(self, dim_model,
                dim_feedforward,
                num_heads,
                drop_prob,
                add_bias,
                activation,
                batch_first, 
                num_layers,
                text_feat_dim, 
                person_model_option,
                text_model,
                person_max_len,
                text_max_len,
                num_discrete_classes,
                num_cont_classes,
                add_pos
                ):

        super(caption_person_MHA_model_text_AVD_Discrete, self).__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.text_feat_dim=text_feat_dim
        self.text_model=text_model
        self.person_model_option=person_model_option
        self.person_max_len=person_max_len
        self.text_max_len=text_max_len
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.num_layers=num_layers
        self.add_pos=add_pos
        self.cls_feat_dim=self.dim_model

        #text feature map layer
        self.text_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person text model 
        self.person_text_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        #initialize position embeddings for query, key and value
        self.pos_person_query=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.person_max_len)
        self.pos_text_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.text_max_len)

    def forward(self, image_data, input_ids, text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0]
        text_feat=self.text_feature_map_layer(text_feat) #(converting the dimensionality to 512)
        text_mask=~text_mask #inverting the mask for nn.MultiheadAttention

        #text feature map layer
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)

        #person text fusion model   
        if(self.add_pos):

            inp_person_feat =   inp_person_feat.permute(1,0,2)
            inp_person_feat = self.pos_person_query(inp_person_feat) # (49,B,512)
            inp_person_feat = inp_person_feat.permute(1,0,2) # (B,49,512)

            text_feat =   text_feat.permute(1,0,2)
            text_feat = self.pos_text_key(text_feat) # (49,B,512)
            text_feat = text_feat.permute(1,0,2) # (B,49,512)

        person_text_feat,_=self.person_text_model(inp_person_feat,text_feat,text_feat,text_mask) #(B,49,512)
        #avg both and concatenate 
        person_text_feat_avg=person_text_feat.mean(dim=1) #(B,512)

        #classification layer
        logits_cont=self.classifier_continuous(person_text_feat_avg) #continuous logits
        logits_discrete=self.classifier_discrete(person_text_feat_avg) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation

        #print(logits_cont.shape,logits_discrete.shape)
        return logits_cont,logits_discrete



class caption_face_dual_MHA_model_scene_late_fusion_AVD_Discrete(nn.Module):

    def __init__(self, dim_model,
                dim_feedforward,
                num_heads,
                drop_prob,
                add_bias,
                activation,
                batch_first, 
                num_layers,
                scene_feat_dim, 
                face_model_option,
                text_model,
                face_max_len,
                scene_max_len,
                text_max_len,
                num_discrete_classes,
                num_cont_classes,
                add_pos
                ):

        super(caption_face_dual_MHA_model_scene_late_fusion_AVD_Discrete, self).__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.add_bias = add_bias
        self.batch_first = batch_first
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.scene_feat_dim=scene_feat_dim
        self.text_model=text_model
        self.face_model_option=face_model_option
        self.face_max_len=face_max_len
        self.text_max_len=text_max_len
        self.scene_max_len=scene_max_len
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.num_layers=num_layers
        self.add_pos=add_pos
        self.cls_feat_dim=2*self.dim_model

        #text feature map layer
        self.text_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)
        self.scene_feature_map_layer=nn.Linear(self.scene_feat_dim, self.dim_model)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)

        #person text model 
        self.face_text_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        self.face_scene_model=CrossModalEncoder(dim_model=self.dim_model,
                                                dim_feedforward=self.dim_feedforward,
                                                num_heads=self.num_heads,
                                                drop_prob=self.drop_prob,
                                                add_bias=self.add_bias,
                                                activation=self.activation,
                                                batch_first=self.batch_first,
                                                num_layers=self.num_layers)

        #initialize position embeddings for query, key and value
        self.pos_face_query=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.face_max_len)
        self.pos_text_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.text_max_len)
        self.pos_scene_key=PositionalEncoding(self.dim_model, self.drop_prob, max_len=self.scene_max_len)

    def forward(self, image_data, inp_scene_feat, input_ids, text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0]
        text_feat=self.text_feature_map_layer(text_feat) #(converting the dimensionality to 512)
        text_mask=~text_mask #inverting the mask for nn.MultiheadAttention

        #text feature map layer
        inp_face_feat=self.face_model_option(image_data) # (B,512,7,7) 
        inp_face_feat=inp_face_feat.view(inp_face_feat.size(0),inp_face_feat.size(1),-1) #(B,512,49)
        inp_face_feat=inp_face_feat.permute(0,2,1) #(B,49,512)

        #scene feature map layer
        inp_scene_feat=self.scene_feature_map_layer(inp_scene_feat) #(B,49,512)

        #person text fusion model   
        if(self.add_pos):

            inp_face_feat =  inp_face_feat.permute(1,0,2)
            inp_face_feat = self.pos_face_query(inp_face_feat) # (49,B,512)
            inp_face_feat = inp_face_feat.permute(1,0,2) # (B,49,512)

            text_feat =   text_feat.permute(1,0,2)
            text_feat = self.pos_text_key(text_feat) # (49,B,512)
            text_feat = text_feat.permute(1,0,2) # (B,49,512)

            inp_scene_feat =   inp_scene_feat.permute(1,0,2)
            inp_scene_feat = self.pos_scene_key(inp_scene_feat) # (49,B,512)
            inp_scene_feat = inp_scene_feat.permute(1,0,2) # (B,49,512)

        face_text_feat,_=self.face_text_model(inp_face_feat,text_feat,text_feat,text_mask) #(B,49,512)

        #person scene fusion model
        face_scene_feat,_=self.face_scene_model(inp_face_feat,inp_scene_feat,inp_scene_feat,None) #(B,49,512)

        #avg both and concatenate 
        face_text_feat_avg=face_text_feat.mean(dim=1) #(B,512)
        face_scene_feat_avg=face_scene_feat.mean(dim=1) #(B,512)

        ov_feat=torch.cat((face_text_feat_avg,face_scene_feat_avg),dim=1) #(B,1024)

        #classification layer
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        
        return logits_discrete
        
        














# pretrained_model=models.resnet34(pretrained=True)
# person_model=torch.nn.Sequential(*list(pretrained_model.children())[:-1])

# model_dict={'dim_model':512,
#             'dim_feedforward':2048,
#             'num_heads':8,
#             'drop_prob':0.1,
#             'add_bias':True,
#             'activation':'relu',
#             'batch_first':True,
#             'num_layers':2,
#             'scene_feat_dim':768,
#             'person_model_option':person_model,
#             'person_max_len':49,
#             'text_max_len':512,
#             'text_model':'bert-base-uncased',
#             'num_discrete_classes':26,
#             'num_cont_classes':3,
#             'add_pos':True
#             }

# model=caption_person_MHA_model_scene_late_fusion_AVD_Discrete(**model_dict)
# print(model)







        

        


