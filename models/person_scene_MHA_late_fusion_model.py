#bottom up features for person and MHA for scene features 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from MCAN_model import *
from mcan_config import *
from torchvision import models 

class Adapter_model(nn.Module):
    def __init__(self,in_dim,mid_dim):
        super(Adapter_model, self).__init__()
        self.in_dim=in_dim 
        self.mid_dim=mid_dim 
        #upsampled and downsampled operations
        self.fc_downsample=nn.Linear(self.in_dim,self.mid_dim) #downsamples the input data to mid_dim dimension
        self.fc_upsample=nn.Linear(self.mid_dim,self.in_dim) #upsamples the downsampled data to in_dim dimension
        self.RELU_op=nn.ReLU()

    def forward(self,x):
        downsample_feat=self.fc_downsample(x)
        downsample_feat=self.RELU_op(downsample_feat)
        upsample_feat=self.fc_upsample(downsample_feat)
        x=x+upsample_feat
        return(x)


class scene_SA_person_adapter_model(nn.Module):

    def __init__(self,mcan_config,
                    in_feat_dim=2048,
                    mid_feat_dim=512,
                    scene_feat_dim=768,
                    num_adapter_layers=2,
                    fusion_option='concat',
                    num_classes=26
                    ):

        super(scene_SA_person_adapter_model, self).__init__()

        self.in_feat_dim=in_feat_dim
        self.mid_feat_dim=mid_feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.num_adapter_layers=num_adapter_layers
        self.mcan_config=mcan_config
        self.fusion_option=fusion_option
        self.num_classes=num_classes 

        #person adapter module 
        self.person_adapter_module=nn.ModuleList([Adapter_model(self.in_feat_dim,self.mid_feat_dim) for _ in range(self.num_adapter_layers)])

        #scene adapter module
        #self.scene_model=SA(self.mcan_config)

        #classifier layer
        self.inp_cls_feat_dim=(self.scene_feat_dim+in_feat_dim)
        self.classifier_fc=nn.Linear(self.inp_cls_feat_dim,self.num_classes)


    def forward(self,inp_person_feat,inp_scene_feat):

        for person_adapter_layer in self.person_adapter_module:
            inp_person_feat=person_adapter_layer(inp_person_feat)

        #scene_feat=self.scene_model(inp_scene_feat,None)

        scene_feat_cls=inp_scene_feat[:,0,:]
    
        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls],dim=1)

        logits=self.classifier_fc(ov_feat)

        return(logits)

class scene_SA_person_fc_model(nn.Module):

    def __init__(self,mcan_config,
                    in_feat_dim=2048,
                    mid_feat_dim=512,
                    scene_feat_dim=768,
                    num_adapter_layers=2,
                    fusion_option='concat',
                    num_classes=26
                    ):

        super(scene_SA_person_fc_model, self).__init__()

        self.in_feat_dim=in_feat_dim
        self.mid_feat_dim=mid_feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.num_adapter_layers=num_adapter_layers
        self.mcan_config=mcan_config
        self.fusion_option=fusion_option
        self.num_classes=num_classes 

        #person adapter module 
        self.person_fc_model=nn.Sequential(
                            nn.Linear(self.in_feat_dim,self.mid_feat_dim),
                            nn.ReLU(),
                            nn.Linear(self.mid_feat_dim,self.scene_feat_dim),
                            nn.ReLU()
        )
        #nn.ModuleList([Adapter_model(self.in_feat_dim,self.mid_feat_dim) for _ in range(self.num_adapter_layers)])

        #scene adapter module
        self.scene_model=SA(self.mcan_config)

        #classifier layer
        self.inp_cls_feat_dim=(self.scene_feat_dim)
        self.classifier_fc=nn.Linear(self.inp_cls_feat_dim,self.num_classes)


    def forward(self,inp_person_feat,inp_scene_feat):

        #for person_adapter_layer in self.person_adapter_module:
        inp_person_feat=self.person_fc_model(inp_person_feat)

        # scene_feat=self.scene_model(inp_scene_feat,None)

        # scene_feat_cls=inp_scene_feat[:,0,:]
    
        # if(self.fusion_option=='concat'):
        #     ov_feat=torch.cat([inp_person_feat,scene_feat_cls],dim=1)

        logits=self.classifier_fc(inp_person_feat)

        return(logits)


class scene_SA_person_fc_model_resnet_finetuned(nn.Module):

    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option='concat',
                    num_classes=26
                    ):

        super(scene_SA_person_fc_model_resnet_finetuned, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.num_classes=num_classes 

        # if(self.person_model_option=='resnet34'):
        #     self.person_model=torch.nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])

        # elif(self.person_model_option=='resnet50'):
        #     self.person_model=torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

        # elif(self.person_model_option=='resnet101'):\
        #     self.person_model=torch.nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

        # for param in self.person_model.parameters():
        #     print(param.requires_grad)
            
        #classifier layer
        self.inp_cls_feat_dim=(self.feat_dim+self.scene_feat_dim)
        self.classifier_fc=nn.Linear(self.inp_cls_feat_dim,self.num_classes)


    def forward(self,image_data,inp_scene_feat):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        #print(inp_person_feat.size())
        scene_feat_cls=inp_scene_feat[:,0,:]
    
        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls],dim=1)

        logits=self.classifier_fc(ov_feat)

        return(logits)

class scene_SA_person_fc_model_resnet_finetuned_AVD(nn.Module):

    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_classes
                    ):

        super(scene_SA_person_fc_model_resnet_finetuned_AVD, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.num_classes=num_classes 

        #if(self.person_model_option=='resnet34'):
        #self.person_model=
        #torch.nn.Sequential(*list(models.resnet34(pretrained=False).children())[:-1])

        # elif(self.person_model_option=='resnet50'):
        #     self.person_model=torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])

        # elif(self.person_model_option=='resnet101'):
            #self.person_model=torch.nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

        # for param in self.person_model.parameters():
        #     print(param.requires_grad)
        #classifier layer
        self.inp_cls_feat_dim=(self.feat_dim+self.scene_feat_dim)
        self.classifier_fc=nn.Linear(self.inp_cls_feat_dim,self.num_classes)
        self.sigmoid_layer=nn.Sigmoid()


    def forward(self,image_data,inp_scene_feat):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        scene_feat_cls=inp_scene_feat[:,0,:]
        #torch.mean(inp_scene_feat[:,1:-1,:],dim=1)
    
        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls],dim=1)

        logits=self.classifier_fc(ov_feat)
        logits=self.sigmoid_layer(logits)

        return(logits)

class scene_person_LF_resnet_finetuned_AVD_Discrete(nn.Module):

    def __init__(self,
                    feat_dim,
                    scene_feat_dim,
                    person_model_option,
                    fusion_option,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(scene_person_LF_resnet_finetuned_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.fusion_option=fusion_option
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        
        #classifier layer
        self.inp_cls_feat_dim=(self.feat_dim+self.scene_feat_dim)
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.inp_cls_feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()


    def forward(self,image_data,inp_scene_feat):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        
        scene_feat_cls=inp_scene_feat[:,0,:]
        #print(inp_person_feat.size(),scene_feat_cls.size())

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([inp_person_feat,scene_feat_cls],dim=1)

        logits_cont=self.classifier_continuous(ov_feat)
        logits_discrete=self.classifier_discrete(ov_feat)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)

class person_fc_model_resnet_finetuned_AVD_Discrete(nn.Module):

    def __init__(self,
                    feat_dim,
                    person_model_option,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(person_fc_model_resnet_finetuned_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        
        #classifier layer
        self.inp_cls_feat_dim=(self.feat_dim)
        self.classifier_discrete=nn.Linear(self.inp_cls_feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.inp_cls_feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()

    def forward(self,image_data):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        
        logits_cont=self.classifier_continuous(inp_person_feat)
        logits_discrete=self.classifier_discrete(inp_person_feat)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)



# in_feat_dim=2048
# mid_feat_dim=512
# scene_feat_dim=768
# num_adapter_layers=2
# fusion_option='concat'
# num_classes=26
# mcan_config_c=Cfgs()
# model=scene_SA_person_adapter_model(mcan_config_c,in_feat_dim,mid_feat_dim,scene_feat_dim,num_adapter_layers,fusion_option,num_classes)
# #print(model)