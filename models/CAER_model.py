import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from MCAN_model import *
from transformers import BertModel

class CAER_face_only_model(nn.Module):
    def __init__(self, face_model,feat_dim=512,num_classes=7):
        super(CAER_face_only_model, self).__init__()

        self.num_classes=num_classes
        self.feat_dim=feat_dim 

        self.face_model=face_model

        #fully connected layer
        self.fc=nn.Linear(self.feat_dim,self.num_classes)

    def forward(self,img):

        #face model 
        face_feat=self.face_model(img)
        face_feat=torch.squeeze(face_feat)

        #fully connected layer
        logits_discrete=self.fc(face_feat)

        return(logits_discrete)

class CAER_face_caption_lf_model(nn.Module):
    def __init__(self, face_model,feat_dim=512,num_classes=7):
        super(CAER_face_caption_lf_model, self).__init__()

        self.num_classes=num_classes
        self.feat_dim=feat_dim 

        self.face_model=face_model

        #fully connected layer
        self.fc=nn.Linear(2*self.feat_dim,self.num_classes)

    def forward(self,img,clip_features):

        #face model 
        face_feat=self.face_model(img)
        face_feat=torch.squeeze(face_feat)
        clip_features=torch.squeeze(clip_features)

        if(len(clip_features.shape)==1):
            clip_features=torch.unsqueeze(clip_features,0)
        #concatenate the features
        concat_feat=torch.cat((face_feat,clip_features),dim=1)
        #fully connected layer
        logits_discrete=self.fc(concat_feat)

        return(logits_discrete)


#CAER model with cross modal attention between face outputs from resnet and bert encoders from text 
#average the features and run the classifier on the average features (dont concatenate the face features)

class text_face_MCAN_AVD_Discrete(nn.Module):
    def __init__(self,mcan_config,
                    feat_dim,
                    text_feat_dim,
                    face_model,
                    text_model,
                    num_classes=7
                    ):

        super(text_face_MCAN_AVD_Discrete, self).__init__()


        self.feat_dim=feat_dim
        self.text_feat_dim=text_feat_dim
        self.num_classes=num_classes
        self.face_model=face_model
        self.text_model=text_model
        self.num_classes=num_classes

        #fully connected layer
        self.linear_map_layer=nn.Linear(self.feat_dim,self.text_feat_dim)

        #text model (pretrained bert )
        self.bert_model = BertModel.from_pretrained(self.text_model,output_hidden_states=True)

        #freezing bert model parameters
        for params in self.bert_model.parameters():
            params.requires_grad=False

        self.classifier_discrete=nn.Linear(self.text_feat_dim,self.num_classes)

        self.text_guided_face_attention=MCA(mcan_config)

    def forward(self,image_data,input_ids,text_mask):

        #text forward pass
        input_ids=input_ids.squeeze(1)
        text_mask=text_mask.squeeze(1)
        bert_output=self.bert_model(input_ids,attention_mask=text_mask)
        text_feat=bert_output[0] #(B,512,768)
        text_mask=text_mask.unsqueeze(1).unsqueeze(2) #needed for MCAN models

        #person face forward pass 
        inp_face_feat=self.face_model(image_data) # (B,512,7,7) 
        inp_face_feat=inp_face_feat.view(inp_face_feat.size(0),inp_face_feat.size(1),-1) #(B,512,49)
        inp_face_feat=inp_face_feat.permute(0,2,1) #(B,49,512)
        inp_face_feat=self.linear_map_layer(inp_face_feat) #(B,49,768)

        #text guided face attention
        text_feat,mca_feat=self.text_guided_face_attention(text_feat,inp_face_feat,text_mask,None) #use mca feat later
        mca_feat_avg=mca_feat.mean(dim=1) #average pooling over the time dimension

        #concatenate the features
        logits_discrete=self.classifier_discrete(mca_feat_avg)
        return(logits_discrete)

#CAER face model with text features later here 




         

