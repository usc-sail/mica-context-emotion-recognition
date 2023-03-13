import torch.nn as nn
import torch.nn.functional as F
import torch
import math

#person model finetuned with two branches for discrete class and arosual, valence, dominance classification
class person_model_resnet_finetuned_AVD_Discrete(nn.Module):
    def __init__(self,
                    feat_dim,
                    person_model_option,
                    num_discrete_classes,
                    num_cont_classes
                    ):

        super(person_model_resnet_finetuned_AVD_Discrete, self).__init__()

        self.feat_dim=feat_dim
        self.person_model=person_model_option
        self.discrete_num_classes=num_discrete_classes 
        self.cont_num_classes=num_cont_classes
        
        #classifier layer
        self.classifier_discrete=nn.Linear(self.feat_dim,self.discrete_num_classes)
        self.classifier_continuous=nn.Linear(self.feat_dim,self.cont_num_classes)
        self.sigmoid_layer=nn.Sigmoid()

    def forward(self,image_data):

        inp_person_feat=self.person_model(image_data)
        inp_person_feat=torch.squeeze(inp_person_feat)
        
        logits_cont=self.classifier_continuous(inp_person_feat)
        logits_discrete=self.classifier_discrete(inp_person_feat)
        logits_cont=self.sigmoid_layer(logits_cont)

        return(logits_cont,logits_discrete)


