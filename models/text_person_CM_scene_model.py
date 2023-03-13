#text+scene information must be combined through MCAN 
#person information must be processed by adapter layers
import torch
import torch.nn as nn 
import torchvision
from MCAN_model import *
from mcan_config import *
from transformers import BertModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel

#simple cross modal model with person and text attention followed by scene 
#person backbone frozen + MCAN fusion for text and scene 
class text_person_MCAN_scene_model_AVD_Discrete(nn.Module):
    def __init__(self,mcan_config,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    text_model,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(text_person_MCAN_scene_model_AVD_Discrete, self).__init__()

        self.mcan_config=mcan_config
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.person_model_option=person_model_option 
        self.fusion_option=fusion_option
        self.text_model=text_model 
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.cls_feat_dim=2*self.scene_feat_dim 

        #feature mapping fc layer
        self.feature_map_layer=nn.Linear(self.feat_dim,self.scene_feat_dim)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        #freezing the person model parameters
        # for params in self.person_model_option.parameters():
        #     params.requires_grad=False
        #discrete FC and continuous FC layer 
        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person and scene information fusion model
        self.person_scene_model=MCA(self.mcan_config)

    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2) #needed for MCAN models
        
        #scene forward pass
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)
        inp_person_feat=self.feature_map_layer(inp_person_feat) #(B,49,768)
        #=> (B,512,49) => (B,49,512) (reshape from (512,49) to (49,512))

        text_feat,mca_feat=self.person_scene_model(text_feat,inp_person_feat,text_mask,None) #use mca feat later
        mca_feat_avg=mca_feat.mean(dim=1) #average pooling over the time dimension
        scene_feat_cls=inp_scene_feat[:,0,:] #cls token feature for scene

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([mca_feat_avg,scene_feat_cls],dim=1) #conctenation over the feature dimension space

        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation
 
        return(logits_cont,logits_discrete)


class text_person_MCAN_scene_model_masked_caption_AVD_Discrete(nn.Module):
    def __init__(self,mcan_config,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    text_model,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(text_person_MCAN_scene_model_masked_caption_AVD_Discrete, self).__init__()

        self.mcan_config=mcan_config
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.person_model_option=person_model_option 
        self.fusion_option=fusion_option
        self.text_model=text_model 
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.cls_feat_dim=2*self.scene_feat_dim+self.feat_dim

        #feature mapping fc layer
        self.feature_map_layer=nn.Linear(self.feat_dim,self.scene_feat_dim)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        #freezing the person model parameters
        # for params in self.person_model_option.parameters():
        #     params.requires_grad=False
        #discrete FC and continuous FC layer 
        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person and scene information fusion model
        self.person_scene_model=MCA(self.mcan_config)

    def forward(self,image_data,inp_scene_feat,input_ids,masked_text_feat,text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2) #needed for MCAN models

        #masked text features from CLIP 
        masked_text_feat=masked_text_feat.squeeze(1) #(B,512,512)
        
        #scene forward pass
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)
        inp_person_feat=self.feature_map_layer(inp_person_feat) #(B,49,768)
        #=> (B,512,49) => (B,49,512) (reshape from (512,49) to (49,512))

        text_feat,mca_feat=self.person_scene_model(text_feat,inp_person_feat,text_mask,None) #use mca feat later
        mca_feat_avg=mca_feat.mean(dim=1) #average pooling over the time dimension
        scene_feat_cls=inp_scene_feat[:,0,:] #cls token feature for scene

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([mca_feat_avg,masked_text_feat,scene_feat_cls],dim=1) #conctenation over the feature dimension space

        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation
 
        return(logits_cont,logits_discrete)

class text_person_MCAN_scene_model_masked_caption_AVD_Discrete(nn.Module):
    def __init__(self,mcan_config,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    text_model,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(text_person_MCAN_scene_model_masked_caption_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.person_model_option=person_model_option 
        self.fusion_option=fusion_option
        self.text_model=text_model 
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.cls_feat_dim=2*self.scene_feat_dim+self.feat_dim

        #feature mapping fc layer
        self.feature_map_layer=nn.Linear(self.feat_dim,self.scene_feat_dim)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        #freezing the person model parameters
        # for params in self.person_model_option.parameters():
        #     params.requires_grad=False
        #discrete FC and continuous FC layer 
        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person and scene information fusion model
        self.person_scene_model=MCA(self.mcan_config)

    def forward(self,image_data,inp_scene_feat,input_ids,masked_text_feat,text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2) #needed for MCAN models

        #masked text features from CLIP 
        masked_text_feat=masked_text_feat.squeeze(1) #(B,512,512)
        
        #scene forward pass
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)
        inp_person_feat=self.feature_map_layer(inp_person_feat) #(B,49,768)
        #=> (B,512,49) => (B,49,512) (reshape from (512,49) to (49,512))

        text_feat,mca_feat=self.person_scene_model(text_feat,inp_person_feat,text_mask,None) #use mca feat later
        mca_feat_avg=mca_feat.mean(dim=1) #average pooling over the time dimension
        scene_feat_cls=inp_scene_feat[:,0,:] #cls token feature for scene

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([mca_feat_avg,masked_text_feat,scene_feat_cls],dim=1) #conctenation over the feature dimension space

        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation
 
        return(logits_cont,logits_discrete)


class language_guided_person_encoder_scene_late_fusion_AVD_Discrete(nn.Module):

    def __init__(self,mcan_config,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    text_model,
                    num_discrete_classes,
                    num_cont_classes):

            super(language_guided_person_encoder_scene_late_fusion_AVD_Discrete, self).__init__()
            
            self.mcan_config=mcan_config
            self.feat_dim=feat_dim
            self.scene_feat_dim=scene_feat_dim
            self.person_model_option=person_model_option
            self.fusion_option=fusion_option
            self.text_model=text_model
            self.num_discrete_classes=num_discrete_classes
            self.num_cont_classes=num_cont_classes
            self.cls_feat_dim=2*self.scene_feat_dim #just scene + person for now 

            #text model (pretrained bert )
            self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

            #feature mapping fc layer
            self.feature_map_layer=nn.Linear(self.feat_dim,self.scene_feat_dim)

            #freezing bert model parameters
            for params in self.bert_model.parameters():
                params.requires_grad=False

            #freeze the person model as well
            # for params in self.person_model_option.parameters():
            #     params.requires_grad=False #later unfreeze this 
            
            # for params in self.person_model_option.parameters():
            #     print(params.requires_grad)
            
            self.img_txt_attention=MCA(self.mcan_config)
            self.img_2_img_attention=CMA(self.mcan_config)

            #discrete FC and continuous FC layer
            self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
            self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)

            self.sigmoid_layer=nn.Sigmoid()

    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):


        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2) #needed for img_txt_attention model 

        #person model 
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)
        inp_person_feat=self.feature_map_layer(inp_person_feat) #(B,49,768)

        #img_txt_attention model
        text_feat,mca_feat=self.img_txt_attention(text_feat,inp_person_feat,text_mask,None)

        #add mca feat to the inp person feat
        text_guided_feat=inp_person_feat+mca_feat #(B,49,768) #use this as query and key 

        #img_2_img_attention model
        fin_person_feat=self.img_2_img_attention(text_guided_feat,inp_person_feat,None,None) #(B,49,768)

        #average pooling over the time dimension  
        fin_person_feat_avg=fin_person_feat.mean(dim=1) #average pooling over the time dimension

        #cls token feature for scene
        scene_feat_cls=inp_scene_feat[:,0,:] #cls token feature for scene

        #concatenation over the feature dimension space
        ov_feat=torch.cat([fin_person_feat_avg,scene_feat_cls],dim=1)

        #discrete logits
        logits_discrete=self.classifier_discrete(ov_feat)
        logits_cont=self.classifier_continuous(ov_feat) 

        #sigmoid activation
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)

#model for cm attention between (text and person feature) and (scene and person feature) + avg pool over both the features followed by concat /addition/ point wise multiplication
#no language guided person encoder - just simple cross modal attention 
#keep resnet frozen 

class dual_CM_text_person_scene_late_fusion_model(nn.Module):

    def __init__(self,mcan_config,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    text_model,
                    num_discrete_classes,
                    num_cont_classes):

            super(dual_CM_text_person_scene_late_fusion_model, self).__init__()
            
            self.mcan_config=mcan_config
            self.feat_dim=feat_dim
            self.scene_feat_dim=scene_feat_dim
            self.person_model_option=person_model_option
            self.fusion_option=fusion_option
            self.text_model=text_model
            self.num_discrete_classes=num_discrete_classes
            self.num_cont_classes=num_cont_classes
            self.cls_feat_dim=2*self.scene_feat_dim

            #text model (pretrained bert )
            self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

            #feature mapping fc layer
            self.feature_map_layer=nn.Linear(self.feat_dim,self.scene_feat_dim)

            #freezing bert model parameters
            for params in self.bert_model.parameters():
                params.requires_grad=False

            #text guided person attention model 
            self.text_guided_person_attention=MCA(self.mcan_config)

            #scene guided person attention model
            self.scene_guided_person_attention=MCA(self.mcan_config)

            #discrete FC and continuous FC layer
            self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
            self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
            self.sigmoid_layer=nn.Sigmoid()


    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):


        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2) #needed for img_txt_attention model 

        #person model 
        inp_person_feat=self.person_model_option(image_data) # (B,512,7,7) 
        inp_person_feat=inp_person_feat.view(inp_person_feat.size(0),inp_person_feat.size(1),-1) #(B,512,49)
        inp_person_feat=inp_person_feat.permute(0,2,1) #(B,49,512)
        inp_person_feat=self.feature_map_layer(inp_person_feat) #(B,49,768)

        #text guided person attention model
        _,text_guided_person_feat=self.text_guided_person_attention(text_feat,inp_person_feat,text_mask,None) #(B,49,768)

        #scene guided person attention model
        _,scene_guided_person_feat=self.scene_guided_person_attention(inp_scene_feat,inp_person_feat,None,None) #(B,49,768)

        #average pooling over the time dimension
        text_guided_person_feat_avg=text_guided_person_feat.mean(dim=1) #average pooling over the time dimension
        scene_guided_person_feat_avg=scene_guided_person_feat.mean(dim=1) #average pooling over the time dimension

        #concatenate 
        if self.fusion_option=='concat':
            ov_feat=torch.cat([text_guided_person_feat_avg,scene_guided_person_feat_avg],dim=1)

        #fc layer 
        logits_discrete=self.classifier_discrete(ov_feat)
        logits_cont=self.classifier_continuous(ov_feat)

        #sigmoid activation
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)



        
            


















            

            
