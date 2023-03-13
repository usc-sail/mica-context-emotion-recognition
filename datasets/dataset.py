#dataset should have three streams of data:
#text caption with respective tokenized version 
#image pixels with size as (B,3,224) : will go as input to the scene model to extract the scene features
#person specific feature based on GT data 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torch.nn.functional as F
import pickle 
import pandas as pd 
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
import os 
from PIL import Image 
from torchvision import transforms
from tqdm import tqdm
#(scene feature, text caption, person feature)
#class Person
class Person_Scene_Text_Feature_dataset(Dataset):

    def __init__(self, scene_pkl_file, tokenizer, split_csv_file, split_pkl_file, label_map_file, max_length=512, num_classes=26):

        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.split_pkl_file = split_pkl_file
        self.num_classes=num_classes
        self.max_length=max_length
        self.tokenizer=tokenizer

        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        caption=self.split_csv_data['caption'].iloc[idx]
        gender=self.split_csv_data['gender'].iloc[idx]
        age=self.split_csv_data['age'].iloc[idx]

        person_dict=self.split_pkl_data[filename][person_id]
        person_feature=person_dict['feat']

        #encode the caption
        encoded = self.tokenizer.encode_plus(
            text=caption,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.max_length,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        # Get the input IDs and attention mask in tensor format
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']

    
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            #print(lab)
            discrete_emotion_label[0,self.label_map[lab]]=1

        #continuous emotion will have the key as 'cont_emotion' and three sub categories as arousal, valence, dominance
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={'input_ids':input_ids,
            'attn_mask':attn_mask,
            'token_type_ids':token_type_ids,
            'scene_feat':scene_data,
            'person_feat':person_feature,
            'emotion':discrete_emotion_label}

        #need to tokenize the caption using tokenizerex
        return(return_dict)


class Person_Scene_dataset(Dataset):

    def __init__(self, scene_pkl_file, split_csv_file, split_pkl_file, label_map_file, num_classes=26):

        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.split_pkl_file = split_pkl_file
        self.num_classes=num_classes
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]
        person_feature=person_dict['feat']

        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[0,self.label_map[lab]]=1

        #continuous emotion will have the key as 'cont_emotion' and three sub categories as arousal, valence, dominance
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={'scene_feat':scene_data,
            'person_feat':person_feature,
            'emotion':discrete_emotion_label}

        #need to tokenize the caption using tokenizerex
        return(return_dict)

class Person_Scene_dataset_resnet34(Dataset):

    def __init__(self, scene_pkl_file, split_csv_file, total_split_pkl_file, feature_split_pkl_file, label_map_file, num_classes=26):

        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.total_split_pkl_file = total_split_pkl_file
        self.feature_split_pkl_file = feature_split_pkl_file
        self.num_classes=num_classes

        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.feature_data = pickle.load(open(self.feature_split_pkl_file,'rb'))
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]
        person_feature=self.feature_data[filename][person_id]

        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[self.label_map[lab]]=1 

        scene_feat=self.scene_data[filename.split(".")[0]]

        return_dict={'scene_feat':scene_feat,
            'person_feat':person_feature,
            'emotion':discrete_emotion_label}
        #print(discrete_emotion_label)
        return(return_dict)

class Person_Scene_dataset_finetuned_network(Dataset):

    def __init__(self,scene_pkl_file,split_csv_file,total_split_pkl_file,base_folder,label_map_file,transforms,num_classes=26):

        #pickle and csv files 
        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.total_split_pkl_file = total_split_pkl_file
        self.base_folder=base_folder
        self.num_classes=num_classes
        self.transforms=transforms

        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):
        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]

        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']

        #base_folder,image_folder,image_filename
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)

        #print(image_path,image_array.size)
        image_array=self.transforms(image_array)
        

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[self.label_map[lab]]=1 
        
        #print(discrete_emotion_label,discrete_emotion_list)

        scene_feat=self.scene_data[filename.split(".")[0]]

        return_dict={'scene_feat':scene_feat,
            'image_array':image_array,
            'emotion':discrete_emotion_label}

        return(return_dict)

class Person_Scene_dataset_finetuned_network_AVD(Dataset):

    def __init__(self,scene_pkl_file,split_csv_file,total_split_pkl_file,base_folder,label_map_file,transforms,num_classes=26):

        #pickle and csv files 
        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.total_split_pkl_file = total_split_pkl_file
        self.base_folder=base_folder
        self.num_classes=num_classes
        self.transforms=transforms

        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):
        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]

        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']

        #base_folder,image_folder,image_filename
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)

        #print(image_path,image_array.size)
        image_array=self.transforms(image_array)
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])
        scene_feat=self.scene_data[filename.split(".")[0]]

        return_dict={'scene_feat':scene_feat,
            'image_array':image_array,
            'emotion':cont_vector}

        return(return_dict)

class Person_Scene_dataset_finetuned_network_AVD_discrete(Dataset):

    def __init__(self,scene_pkl_data,split_csv_file,total_split_pkl_file,base_folder,label_map_file,transforms,num_classes=26):

        #pickle and csv files 
        self.scene_data = scene_pkl_data
        self.split_csv_file = split_csv_file
        self.total_split_pkl_file = total_split_pkl_file
        self.base_folder=base_folder
        self.num_classes=num_classes
        self.transforms=transforms

        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        #self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):
        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]

        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']

        #base_folder,image_folder,image_filename
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)

        #print(image_path,image_array.size)
        image_array=self.transforms(image_array)

        #continuous arousal, valence and dominance ratings 
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #scene features for the dataset
        scene_feat=self.scene_data[filename.split(".")[0]]

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        #discrete emotion label in one-hot setting 
        discrete_emotion_label=np.zeros((self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[self.label_map[lab]]=1 

        #return dict setting 
        return_dict={'scene_feat':scene_feat,
            'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)

class Person_dataset_finetuned_network_AVD_discrete(Dataset):

    def __init__(self,split_csv_file,total_split_pkl_file,base_folder,label_map_file,transforms,num_classes=26):

        #pickle and csv files 
        self.split_csv_file = split_csv_file
        self.total_split_pkl_file = total_split_pkl_file
        self.base_folder=base_folder
        self.num_classes=num_classes
        self.transforms=transforms

        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):
        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]

        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']

        #base_folder,image_folder,image_filename
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)

        #print(image_path,image_array.size)
        image_array=self.transforms(image_array)

        #continuous arousal, valence and dominance ratings 
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        #discrete emotion label in one-hot setting 
        discrete_emotion_label=np.zeros((self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[self.label_map[lab]]=1 

        #return dict setting 
        return_dict={'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)


class Person_Scene_Text_Feature_AVD_Discrete_dataset(Dataset):

    def __init__(self, scene_data, tokenizer, split_csv_file, base_folder, total_split_pkl_file, label_map_file, transforms, max_length=512, num_classes=26):

        #self.scene_pkl_file = scene_pkl_file
        self.scene_data=scene_data
        self.split_csv_file = split_csv_file
        self.total_split_pkl_file = total_split_pkl_file
        self.label_map_file = label_map_file
        self.num_classes=num_classes
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.transforms=transforms

        #self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))
        self.base_folder=base_folder

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        caption=self.split_csv_data['caption'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]
        
        #image reading and preprocessing
        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)
        image_array=self.transforms(image_array)

        #encode the caption
        encoded = self.tokenizer.encode_plus(
            text=caption,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.max_length,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        # Get the input IDs and attention mask in tensor format
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']

        #continuous arousal, valence and dominance ratings 
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[0,self.label_map[lab]]=1

        #read the scene data 
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={'input_ids':input_ids,
            'attn_mask':attn_mask,
            'token_type_ids':token_type_ids,
            'scene_feat':scene_data,
            'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)

class Person_Scene_Text_CLIP_Feature_AVD_Discrete_dataset(Dataset):

    def __init__(self, scene_pkl_file, clip_pkl_file, tokenizer, split_csv_file, base_folder, total_split_pkl_file, label_map_file, transforms, max_length=512, num_classes=26):
        #fix the issue with different data sizes
        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.clip_pkl_file=clip_pkl_file
        self.total_split_pkl_file = total_split_pkl_file
        self.label_map_file = label_map_file
        self.num_classes=num_classes
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.transforms=transforms

        self.clip_data=pickle.load(open(self.clip_pkl_file,'rb'))
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))
        self.base_folder=base_folder

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        caption=self.split_csv_data['caption'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]
        
        #image reading and preprocessing
        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)
        image_array=self.transforms(image_array)

        #tokenizer here for CLIP
        #CLIP tokenized output here
        clip_feature_dictionary=self.clip_data[filename]
        text_features=clip_feature_dictionary['text_features']
        
        #continuous arousal, valence and dominance ratings 
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[0,self.label_map[lab]]=1

        #read the scene data 
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={'text_features':text_features,
            'scene_feat':scene_data,
            'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)

class Person_Scene_multi_Text_CLIP_Feature_AVD_Discrete_dataset(Dataset):

    def __init__(self, scene_pkl_file, lavis_clip_pkl_file, ofa_clip_pkl_file, tokenizer, split_csv_file, base_folder, total_split_pkl_file, label_map_file, transforms, max_length=512, num_classes=26):
        #fix the issue with different data sizes
        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.lavis_clip_pkl_file=lavis_clip_pkl_file
        self.ofa_clip_pkl_file=ofa_clip_pkl_file
        self.total_split_pkl_file = total_split_pkl_file
        self.label_map_file = label_map_file
        self.num_classes=num_classes
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.transforms=transforms

        self.lavis_clip_data=pickle.load(open(self.lavis_clip_pkl_file,'rb')) #lavis clip data
        self.ofa_clip_data=pickle.load(open(self.ofa_clip_pkl_file,'rb')) #ofa clip data
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))
        self.base_folder=base_folder

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['person_id'].iloc[idx]
        caption=self.split_csv_data['caption'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]
        
        #image reading and preprocessing
        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)
        image_array=self.transforms(image_array)

        #tokenizer here for CLIP
        #CLIP tokenized output here
        clip_feature_dictionary=self.ofa_clip_data[filename]
        ofa_text_features=clip_feature_dictionary['text_features']
        lavis_text_features=self.lavis_clip_data[filename]
        
        #continuous arousal, valence and dominance ratings 
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[0,self.label_map[lab]]=1

        #read the scene data 
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={
            'ofa_text_features':ofa_text_features,
            'lavis_text_features':lavis_text_features,
            'scene_feat':scene_data,
            'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)


#Person_Scene_multi_Text_CLIP_Feature_AVD_Discrete_dataset with LAVIS masked and non masked features
class Person_Scene_multi_Text_CLIP_Feature_AVD_Discrete_dataset_LAVIS(Dataset):

    def __init__(self, scene_pkl_file, lavis_mask_clip_pkl_file, tokenizer, split_csv_file, base_folder, total_split_pkl_file, label_map_file, transforms, max_length=512, num_classes=26):

        self.scene_pkl_file = scene_pkl_file
        self.split_csv_file = split_csv_file
        self.lavis_mask_clip_pkl_file=lavis_mask_clip_pkl_file
        self.total_split_pkl_file = total_split_pkl_file
        self.label_map_file = label_map_file
        self.num_classes=num_classes
        self.max_length=max_length
        self.tokenizer=tokenizer
        self.transforms=transforms
        self.base_folder=base_folder

        self.lavis_mask_clip_masked_data=pickle.load(open(self.lavis_mask_clip_pkl_file,'rb')) #lavis clip data with non masked captions
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['Person_id'].iloc[idx]
        lavis_caption=self.split_csv_data['LAVIS_caption'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]

        #image reading and preprocessing
        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)
        image_array=self.transforms(image_array)

        #encode the unmasked lavis caption 
        #encode the caption
        encoded = self.tokenizer.encode_plus(
            text=lavis_caption,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.max_length,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
            )

        # Get the input IDs and attention mask in tensor format
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']

        #feature associated with masked token
        lavis_mask_text_features=self.lavis_mask_clip_masked_data[filename][person_id]

        #continuous arousal, valence and dominance ratings 
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        #discrete emotion vector
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[0,self.label_map[lab]]=1

        #read the scene data 
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={'input_ids':input_ids,
            'attn_mask':attn_mask,
            'token_type_ids':token_type_ids,
            'mask_text_features':lavis_mask_text_features,
            'scene_feat':scene_data,
            'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)


class Person_Scene_masked_unmasked_caption_AVD_Discrete_dataset_LAVIS(nn.Module):

    def __init__(self, scene_pkl_file, lavis_mask_clip_pkl_file, lavis_unmask_clip_pkl_file, split_csv_file, base_folder, total_split_pkl_file, label_map_file, transforms, num_classes=26):

        self.scene_pkl_file=scene_pkl_file
        self.lavis_mask_clip_pkl_file=lavis_mask_clip_pkl_file
        self.lavis_unmask_clip_pkl_file=lavis_unmask_clip_pkl_file
        self.split_csv_file=split_csv_file
        self.total_split_pkl_file=total_split_pkl_file
        self.label_map_file=label_map_file
        self.num_classes=num_classes
        self.transforms=transforms
        self.base_folder=base_folder

        self.lavis_mask_clip_masked_data=pickle.load(open(self.lavis_mask_clip_pkl_file,'rb')) #lavis clip data with non masked captions
        self.lavis_mask_clip_unmasked_data=pickle.load(open(self.lavis_unmask_clip_pkl_file,'rb')) #lavis clip data with non masked captions
        self.scene_data=pickle.load(open(self.scene_pkl_file,'rb'))
        self.split_csv_data=pd.read_csv(self.split_csv_file)
        self.split_pkl_data=pickle.load(open(self.total_split_pkl_file,'rb'))
        self.label_map=pickle.load(open(label_map_file,'rb'))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):
        
        #person dict 
        filename=self.split_csv_data['id'].iloc[idx]
        person_id=self.split_csv_data['Person_id'].iloc[idx]
        person_dict=self.split_pkl_data[filename][person_id]

        #image reading and preprocessing
        image_filename=self.split_pkl_data[filename]['filename']
        image_folder=self.split_pkl_data[filename]['folder']
        image_path=os.path.join(self.base_folder,image_folder,image_filename)
        image_array=Image.open(image_path).convert('RGB')
        bbox_data=(person_dict['bbox'])
        bbox_data=[int(x) for x in bbox_data]
        bbox_data=[max(0,x) for x in bbox_data]
        image_array=image_array.crop(bbox_data)
        image_array=self.transforms(image_array)

        #feature associated with masked token
        lavis_mask_text_features=self.lavis_mask_clip_masked_data[filename][person_id]

        #feature associated with unmasked caption
        lavis_unmask_text_features=self.lavis_mask_clip_unmasked_data[filename]

        #continuous arousal, valence and dominance ratings
        continuous_emotion=(person_dict['cont_emotion'])
        cont_vector=np.array([continuous_emotion['valence']/10.0,continuous_emotion['arousal']/10.0,continuous_emotion['dominance']/10.0])

        #discrete emotion
        discrete_emotion=(person_dict['disc_emotion'])
        if(isinstance(discrete_emotion,str)):
            discrete_emotion_list=[discrete_emotion]
        else:
            discrete_emotion_list=list(discrete_emotion)
        
        #discrete emotion vector
        discrete_emotion_label=np.zeros((1,self.num_classes))
        for lab in discrete_emotion_list:
            discrete_emotion_label[0,self.label_map[lab]]=1

        #read the scene data 
        scene_data=self.scene_data[filename.split(".")[0]]

        return_dict={'mask_text_features':lavis_mask_text_features,
            'unmask_text_features':lavis_unmask_text_features,
            'scene_feat':scene_data,
            'image_array':image_array,
            'cont_emotion':cont_vector,
            'disc_emotion':discrete_emotion_label}

        return(return_dict)



        


# scene_pkl_file="/data/Emotic/vit_scene_features/vit_scene_feature_layer_norm.pkl"
# split_csv_file="/home/dbose_usc_edu/codes/context-emotion-recognition/data/train_aligned_bbox_caption_no_missing_data.csv"
# total_split_pkl_file="/data/Emotic/pkl_files/train_aligned_bbox_caption_data.pkl"
# base_folder="/data/Emotic"
# label_map_file="/data/Emotic/pkl_files/discrete_emotion_mapping_Emotic.pkl"
# transforms= transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ]) 

# person_scene_finetuned_dataset=Person_Scene_dataset_finetuned_network_AVD(scene_pkl_file=scene_pkl_file,
#             split_csv_file=split_csv_file,
#             total_split_pkl_file=total_split_pkl_file,
#             base_folder=base_folder,
#             label_map_file=label_map_file,
#             transforms=transforms,
#             num_classes=3)
# person_scene_finetuned_dataloader=DataLoader(person_scene_finetuned_dataset,batch_size=32,shuffle=True)
# for data in tqdm(person_scene_finetuned_dataloader):
#     print(data['image_array'])
    #break
# return_dict=next(iter(person_scene_finetuned_dataloader))
# print(return_dict['image_array'].shape)
# print(return_dict['scene_feat'].shape)
# print(return_dict['emotion'].shape)
# scene_pkl_file="/bigdata/digbose92/Emotic/vit_scene_features/vit_scene_feature_layer_norm.pkl"
# split_csv_file="/data/digbose92/codes/Emotic_experiments/context-emotion-recognition/data/train_aligned_bbox_caption_no_missing_data.csv"
# total_split_pkl_file="/bigdata/digbose92/Emotic/pkl_files/train_aligned_bbox_caption_data.pkl"
# feature_split_pkl_file="/bigdata/digbose92/Emotic/pkl_files/train_data_resnet34_features.pkl"
# label_map_file="/bigdata/digbose92/Emotic/pkl_files/discrete_emotion_mapping_Emotic.pkl"
# num_classes=26     
# person_scene_dataset=Person_Scene_dataset_resnet34(scene_pkl_file, split_csv_file, total_split_pkl_file, feature_split_pkl_file, label_map_file, num_classes)
# person_scene_dl=DataLoader(person_scene_dataset, batch_size=2, shuffle=True, num_workers=0)
# return_dict=next(iter(person_scene_dl))

# print(return_dict['scene_feat'].shape)
# print(return_dict['person_feat'].shape)
# print(return_dict['emotion'].shape)




        




        











# split_pkl_file="/proj/digbose92/emotic_experiments/pkl_files/train_aligned_bbox_caption_data.pkl"
# split_csv_file="/proj/digbose92/emotic_experiments/codes/context-emotion-recognition/data/train_aligned_bbox_caption_data.csv"
# scene_pkl_file="/proj/digbose92/emotic_experiments/vit_scene_features/vit_scene_feature_layer_norm.pkl"
# label_map_file="/proj/digbose92/emotic_experiments/pkl_files/discrete_emotion_mapping_Emotic.pkl"
# tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
# pds=Person_Scene_Text_Feature_dataset(scene_pkl_file,
#                 tokenizer,
#                 split_csv_file,
#                 split_pkl_file,label_map_file)
# pdl=DataLoader(pds,batch_size=2,shuffle=True)
# return_dict=next(iter(pdl))

# input_ids=return_dict['input_ids']
# attn_mask=return_dict['attn_mask']
# token_type_ids=return_dict['token_type_ids']
# scene_feat=return_dict['scene_feat']
# person_feat=return_dict['person_feat']
# emotion=return_dict['emotion']


# print(input_ids.size())
# print(attn_mask.size())

# model_type='bert-base-uncased'
# #config = BertConfig.from_pretrained(model_type, output_hidden_states=True)
# bert_model=BertModel.from_pretrained(model_type,output_hidden_states=True)

# with torch.no_grad():
#     input_ids=input_ids.squeeze(1)
#     attn_mask=attn_mask.squeeze(1)
#     outputs = bert_model(input_ids, attention_mask=attn_mask)

# print(outputs[0].size())





