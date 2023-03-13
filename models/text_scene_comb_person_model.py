#text+scene information must be combined through MCAN 
#person information must be processed by adapter layers
import torch
import torch.nn as nn 
import torchvision
from MCAN_model import *
from mcan_config import *
from transformers import BertModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel

class Adapter_model(nn.Module):
    def __init__(self,in_dim,mid_dim):
        super(Adapter_model, self).__init__()
        self.in_dim=in_dim 
        self.mid_dim=mid_dim 
        #upsampled and downsampled operations
        self.fc_downsample=nn.Linear(self.in_dim,self.mid_dim) #downsamples the input data to mid_dim dimension
        self.fc_upsample=nn.Linear(self.mid_dim,self.in_dim) #upsamples the downsampled data to in_dim dimension
        self.GELU_op=nn.GELU()

    def forward(self,x):
        downsample_feat=self.fc_downsample(x)
        downsample_feat=self.GELU_op(downsample_feat)
        upsample_feat=self.fc_upsample(downsample_feat)
        x=x+upsample_feat
        return(x)


class text_scene_MCAN_person_adapter_model(nn.Module):

    def __init__(self,mcan_config,
                    in_feat_dim=2048,
                    mid_feat_dim=512,
                    scene_feat_dim=768,
                    num_adapter_layers=2,
                    fusion_option='concat',
                    text_model='bert-base-uncased',
                    num_classes=26
                    ):

        super(text_scene_MCAN_person_adapter_model, self).__init__()

        self.in_feat_dim=in_feat_dim
        self.mid_feat_dim=mid_feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.num_adapter_layers=num_adapter_layers
        self.mcan_config=mcan_config
        self.fusion_option=fusion_option
        self.num_classes=num_classes 
        self.text_model=text_model
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        for params in self.bert_model.parameters(): #freeze the bert model
            params.requires_grad=False

        #mcan_config is a class with items LAYER, HIDDEN_SIZE, MULTI_HEAD, DROPOUT_R

        self.person_adapter_module=nn.ModuleList([Adapter_model(self.in_feat_dim,self.mid_feat_dim) for _ in range(self.num_adapter_layers)])

        #person and scene information fusion model
        self.person_scene_model=MCA_ED(self.mcan_config)

        #classifier layer
        self.inp_cls_feat_dim=(self.scene_feat_dim+in_feat_dim)
        self.classifier_fc=nn.Linear(self.inp_cls_feat_dim,self.num_classes)

    def forward(self,person_feat,scene_feat,input_ids,text_mask):
        
        #print(person_feat.size(),scene_feat.size(),input_ids.size(),text_mask.size())
        for person_adapter_layer in self.person_adapter_module:
            person_feat=person_adapter_layer(person_feat)
        #person_feat=self.person_adapter_module(person_feat)

        #extract the output features from bert model 
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,T,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2)
        #coattention modeling by MCAN architecture
        text_feat,mca_feat=self.person_scene_model(text_feat,scene_feat,text_mask,None)
        #mac_feat dimensions: (B,197,768) #scene feature modified by text information
        #print(mca_feat.size())
        mca_feat_avg=mca_feat.mean(dim=1) #average pooling over the time dimension
        #print(mca_feat_avg.size())

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([person_feat,mca_feat_avg],dim=1)

        #print(ov_feat.size())
        logits=self.classifier_fc(ov_feat)

        return(logits)


#combine scene, text and person information through late fusion
class text_scene_SA_person_fc_model_resnet_finetuned_AVD_Discrete(nn.Module):

    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_discrete_classes,
                    num_cont_classes,
                    text_model='bert-base-uncased',
                    ):

        super(text_scene_SA_person_fc_model_resnet_finetuned_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        self.text_model=text_model
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        for params in self.bert_model.parameters(): #freeze the bert model
            params.requires_grad=False
        
        #classifier layer
        self.inp_cls_feat_dim=(self.feat_dim+2*self.scene_feat_dim) #self.text_feat_dim=self.scene_feat_dim
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.inp_cls_feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()

    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):

        #BERT finetuned model + late fusion of (scene + person ) information
        #extract the output features from bert model 

        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,T,768)
        text_feat_cls=text_feat[:,0,:] #CLS feature
       
        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]
    
        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls,text_feat_cls],dim=1)

        logits_cont=self.classifier_continuous(ov_feat)
        logits_discrete=self.classifier_discrete(ov_feat)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)




#Gated Multimodal fusion implementation between visual and text stream
class GatedMultimodal_Fusion(nn.Module):
    def __init__(self, inp_scene_dim, inp_text_dim, comb_dim):
        super(GatedMultimodal_Fusion, self).__init__()

        self.inp_scene_dim=inp_scene_dim
        self.inp_text_dim=inp_text_dim
        self.comb_dim=comb_dim

        #weight visual and text fully connected layer
        self.weight_visual=nn.Linear(self.inp_scene_dim,self.comb_dim)
        self.weight_text=nn.Linear(self.inp_text_dim,self.comb_dim)
        self.modality_weighting=nn.Linear(self.inp_scene_dim+self.inp_text_dim,1)

        #gate visual and text fully connected layer
        self.tanh_visual=nn.Tanh()
        self.tanh_text=nn.Tanh()
        self.sigmoid_gate=nn.Sigmoid()

    def forward(self,scene_feat,text_feat):

        #weighting
        mapped_feat_visual=self.weight_visual(scene_feat)
        mapped_feat_text=self.weight_text(text_feat)

        #gate
        tanh_visual=self.tanh_visual(mapped_feat_visual)
        tanh_text=self.tanh_text(mapped_feat_text)
        gate=self.sigmoid_gate(self.modality_weighting(torch.cat([scene_feat,text_feat],dim=1)))

        #weighted sum
        weighted_sum=(gate*tanh_visual)+(1-gate)*tanh_text

        return(weighted_sum)


#text+scene+person AVD+Discrete classification
class text_scene_person_fc_model_resnet_finetuned_AVD_Discrete(nn.Module):

    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_discrete_classes,
                    num_cont_classes,
                    text_model='bert-base-uncased',
                    ):

        super(text_scene_person_fc_model_resnet_finetuned_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        self.text_model=text_model
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        for params in self.bert_model.parameters(): #freeze the bert model
            params.requires_grad=False
        
        #classifier layer
        self.inp_cls_feat_dim=(self.feat_dim+2*self.scene_feat_dim) #self.text_feat_dim=self.scene_feat_dim
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.inp_cls_feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()

    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):
        
        #BERT finetuned model + late fusion of (scene + person ) information
        #extract the output features from bert model 

        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,T,768)
        text_feat_cls=text_feat[:,0,:] #CLS feature
       
        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]
    
        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls,text_feat_cls],dim=1)

        logits_cont=self.classifier_continuous(ov_feat)
        logits_discrete=self.classifier_discrete(ov_feat)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)


#text_scene_person_fc_model_resnet_finetuned_AVD_Discrete with CLIP text features 
class CLIP_text_scene_person_model_resnet_finetuned_AVD_Discrete(nn.Module):
    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_discrete_classes,
                    num_cont_classes,
                    text_model='openai/clip-vit-base-patch32'):

        super(CLIP_text_scene_person_model_resnet_finetuned_AVD_Discrete, self).__init__()

        #feat dim, scene feat dim, person model 
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        self.text_model=text_model

        #classifier layer
        self.inp_cls_feat_dim=(2*self.feat_dim+self.scene_feat_dim) #self.text_feat_dim=self.scene_feat_dim
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.inp_cls_feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()

    def forward(self,image_data,inp_scene_feat,clip_features):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls,clip_features],dim=1)

        logits_cont=self.classifier_continuous(ov_feat)
        logits_discrete=self.classifier_discrete(ov_feat)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)


#model for just discrete class classification (26 classes - multi label)
class CLIP_text_scene_person_model_resnet_finetuned_Discrete(nn.Module):
    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_discrete_classes,
                    text_model='openai/clip-vit-base-patch32'):

        super(CLIP_text_scene_person_model_resnet_finetuned_Discrete, self).__init__()

        #feat dim, scene feat dim, person model 
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.text_model=text_model

        #classifier layer
        self.inp_cls_feat_dim=(2*self.feat_dim+self.scene_feat_dim) #self.text_feat_dim=self.scene_feat_dim
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)

    def forward(self,image_data,inp_scene_feat,clip_features):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls,clip_features],dim=1)

        logits_discrete=self.classifier_discrete(ov_feat)
        
        return(logits_discrete)


#model for just discrete class classification (26 classes - multi label)
class CLIP_multi_text_scene_person_model_resnet_finetuned_Discrete(nn.Module):
    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_discrete_classes,
                    text_model='openai/clip-vit-base-patch32'):

        super(CLIP_multi_text_scene_person_model_resnet_finetuned_Discrete, self).__init__()

        #feat dim, scene feat dim, person model 
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.text_model=text_model

        #classifier layer
        self.inp_cls_feat_dim=(3*self.feat_dim+self.scene_feat_dim) #self.text_feat_dim=self.scene_feat_dim
        #2*(text_feat_dim)+feat_dim
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)

    def forward(self,image_data,inp_scene_feat,ofa_clip_features,lavis_clip_features):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]

        text_fusion_feat=torch.cat([ofa_clip_features,lavis_clip_features],dim=1)
        ov_feat=torch.cat([inp_person_feat,scene_feat_cls,text_fusion_feat],dim=1)

        logits_discrete=self.classifier_discrete(ov_feat)
        
        return(logits_discrete)

#text+scene+person AVD+Discrete GMU fusion classification
class text_scene_SA_person_fc_model_resnet_finetuned_AVD_Discrete_GMU_Fusion(nn.Module):

    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    num_discrete_classes,
                    num_cont_classes,
                    text_model='bert-base-uncased',
                    ):

        super(text_scene_SA_person_fc_model_resnet_finetuned_AVD_Discrete_GMU_Fusion, self).__init__()
    
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        self.text_model=text_model
        self.cls_feat_dim=self.scene_feat_dim
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)
    
        for params in self.bert_model.parameters():
            params.requires_grad=False

        self.gmu_unit=GatedMultimodal_Fusion(self.feat_dim+self.scene_feat_dim,self.scene_feat_dim,self.scene_feat_dim) 
        #self.scene_feat_dim=self.text_feat_dim (both ViT and BERT have 768 dim)

        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()
        
    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):
        
        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0]
        text_feat_cls=text_feat[:,0,:]

        #scene forward pass
        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]

        #concatenate the scene and text features
        visual_concat_feature=torch.cat([inp_person_feat,scene_feat_cls],dim=1)
        gmu_fused_feature=self.gmu_unit(visual_concat_feature,text_feat_cls)

        logits_cont=self.classifier_continuous(gmu_fused_feature)
        logits_discrete=self.classifier_discrete(gmu_fused_feature)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)



#person backbone frozen + MCAN fusion for text and scene 
class text_scene_MCAN_person_model_AVD_Discrete(nn.Module):
    def __init__(self,mcan_config,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    text_model,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(text_scene_MCAN_person_model_AVD_Discrete, self).__init__()

        self.mcan_config=mcan_config
        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.person_model_option=person_model_option 
        self.fusion_option=fusion_option
        self.text_model=text_model 
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.cls_feat_dim=self.feat_dim+self.scene_feat_dim 

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
        inp_person_feat=self.person_model_option(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat) #(B,512)

        text_feat,mca_feat=self.person_scene_model(text_feat,inp_scene_feat,text_mask,None) #use mca feat later
        mca_feat_avg=mca_feat.mean(dim=1) #average pooling over the time dimension

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,mca_feat_avg],dim=1) #conctenation over the feature dimension space

        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation
 
        return(logits_cont,logits_discrete)


class text_scene_FiLM_person_model_AVD_Discrete(nn.Module):
    #here we don't use self attention for scene features  
    def __init__(self,
                feat_dim,
                scene_feat_dim,
                person_model_option,
                text_model,
                film_dimension,
                fusion_option,
                num_discrete_classes,
                num_cont_classes
                ):

        super(text_scene_FiLM_person_model_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.person_model_option=person_model_option
        self.text_model=text_model
        self.film_dimension=film_dimension
        self.fusion_option=fusion_option
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.cls_feat_dim=self.feat_dim+self.scene_feat_dim

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        #gamma,beta layer for FiLM
        self.film_gamma_layer=nn.Linear(self.scene_feat_dim,self.film_dimension)
        self.film_beta_layer=nn.Linear(self.scene_feat_dim,self.film_dimension)

        #discrete FC and continuous FC layer 
        self.classifier_discrete=nn.Linear(self.cls_feat_dim,self.num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.cls_feat_dim,self.num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()

        #person and scene information fusion model
        #self.person_scene_model=FiLMedNet(self.feat_dim,self.scene_feat_dim)

    def forward(self,image_data,inp_scene_feat,input_ids,text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_feat_cls=text_feat[:,0,:]

        #scene forward pass
        inp_person_feat=self.person_model_option(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat) #(B,512)

        #use film layer to get gamma and beta
        film_gamma=self.film_gamma_layer(text_feat_cls)
        film_beta=self.film_beta_layer(text_feat_cls)

        #film layer
        film_person_text_embedding=torch.mul(film_gamma,inp_person_feat)+film_beta

        #concatenate the film person and scene embedding
    
        #print(film_person_text_embedding.shape)
        scene_feat_cls=inp_scene_feat[:,0,:]
        ov_feat=torch.cat([scene_feat_cls,film_person_text_embedding],dim=1) #conctenation over the feature dimension space

        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation

        return(logits_cont,logits_discrete)


#late fusion of masked caption CLIP features with caption features and scene features 
class CLIP_text_scene_person_model_resnet_finetuned_masked_LAVIS(nn.Module):

    def __init__(self,feat_dim,
                person_model_option,
                scene_feat_dim,
                num_discrete_classes,
                num_cont_classes):

        super(CLIP_text_scene_person_model_resnet_finetuned_masked_LAVIS, self).__init__()

        self.feat_dim=feat_dim
        self.person_model_option=person_model_option 
        self.scene_feat_dim=scene_feat_dim 
        self.num_discrete_classes=num_discrete_classes
        self.num_cont_classes=num_cont_classes
        self.fusion_dim=3*feat_dim+scene_feat_dim 

        #fc layers for discrete and continuous classification 
        self.classifier_discrete=nn.Linear(self.fusion_dim,num_discrete_classes)
        self.classifier_continuous=nn.Linear(self.fusion_dim,num_cont_classes)
        self.sigmoid_layer=nn.Sigmoid()
    
    def forward(self,image_data,inp_scene_feat,unmasked_caption_features,masked_caption_features):
        """
            unmasked_caption_features: CLIP text features for entire captions
            masked_caption_features: CLIP text features for masked captions 
        """
        #scene forward pass
        inp_person_feat=self.person_model_option(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat) #(B,512)

        unmasked_caption_features=torch.squeeze(unmasked_caption_features) #(B,512)
        masked_caption_features=torch.squeeze(masked_caption_features) #(B,512)

        #input scene features
        scene_feat_cls=inp_scene_feat[:,0,:]

        
        ov_feat=torch.cat([inp_person_feat,scene_feat_cls,unmasked_caption_features,masked_caption_features],dim=1)
        
        logits_cont=self.classifier_continuous(ov_feat) #continuous logits
        logits_discrete=self.classifier_discrete(ov_feat) #discrete logits
        logits_cont=self.sigmoid_layer(logits_cont) #sigmoid activation

        return(logits_cont,logits_discrete)




    

        
# mcan_config_c=Cfgs()
# model_dict={
#         'mcan_config': mcan_config_c,
#         'in_feat_dim':2048,
#         'mid_feat_dim':512,
#         'scene_feat_dim':768,
#         'num_adapter_layers':2,
#         'fusion_option':'concat',
#         'num_classes':26
# }
# text_scene_combined_person_model=text_scene_comb_person_model(**model_dict)
# print(text_scene_combined_person_model)













