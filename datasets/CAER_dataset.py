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

class CAER_Face_only_dataset(Dataset):

    def __init__(self,csv_file,transforms,bbox_file,label_map_file):


        self.csv_data=pd.read_csv(csv_file)
        self.transforms=transforms
        self.bbox_file=bbox_file
        self.label_map_file=label_map_file

        self.label_map=pickle.load(open(self.label_map_file,"rb"))
        self.bbox_dict=pickle.load(open(self.bbox_file,"rb"))

    def __getitem__(self,idx):

        file_path=self.csv_data['file_path'][idx]
        label=self.csv_data['label'][idx]
        img_label=self.label_map[label]
        image=Image.open(file_path)

        #read the bbox data 
        box_key=label+"_"+file_path.split("/")[-1]
        #print(box_key)
        bbox=self.bbox_dict[box_key]['boxes']
        probs=self.bbox_dict[box_key]['probs']

        #sort by probability values 
        sorted_idx=np.argsort(probs)[::-1]
        bbox=bbox[sorted_idx[0]]
        bbox=bbox.astype(int)
        image=image.crop(bbox)
        image.save("face_sample.jpg")

        #transform the image
        image=self.transforms(image)

        #mapped label
        mapped_label=self.label_map[label]

        return_dict={'image':image,'label':mapped_label}
        return(return_dict)

    def __len__(self):
        return len(self.csv_data)


class CAER_Face_Caption_Dataset(Dataset):
    def __init__(self,csv_file,transforms,bbox_file,label_map_file,clip_feature_file):

        self.csv_data=pd.read_csv(csv_file)
        self.transforms=transforms
        self.bbox_file=bbox_file
        self.label_map_file=label_map_file
        self.clip_feature_file=clip_feature_file

        self.label_map=pickle.load(open(self.label_map_file,"rb"))
        self.bbox_dict=pickle.load(open(self.bbox_file,"rb"))
        self.clip_feature_dict=pickle.load(open(self.clip_feature_file,"rb"))

    def __getitem__(self,idx):

        file_path=self.csv_data['file_path'][idx]
        label=self.csv_data['label'][idx]
        img_label=self.label_map[label]
        image=Image.open(file_path)

        #read the bbox data 
        box_key=label+"_"+file_path.split("/")[-1]
        #print(box_key)
        bbox=self.bbox_dict[box_key]['boxes']
        probs=self.bbox_dict[box_key]['probs']

        #sort by probability values 
        sorted_idx=np.argsort(probs)[::-1]
        bbox=bbox[sorted_idx[0]]
        bbox=bbox.astype(int)
        image=image.crop(bbox)
        image.save("face_sample.jpg")

        #transform the image
        image=self.transforms(image)

        #mapped label
        mapped_label=self.label_map[label]

        #read the CLIP features 
        clip_feature_key=box_key
        clip_feature=self.clip_feature_dict[clip_feature_key]

        return_dict={'image':image,'label':mapped_label,'clip_feature':clip_feature}
        return(return_dict)

    def __len__(self):
        return len(self.csv_data)

class CAER_Face_Caption_Feature_AVD_Discrete_dataset(Dataset):

    def __init__(self, tokenizer, csv_file, transforms, bbox_file, label_map_file , caption_file, max_length=512, num_classes=7):

        self.csv_file = csv_file        
        self.label_map_file = label_map_file
        self.num_classes=num_classes
        self.caption_file=caption_file
        self.max_length=max_length
        self.bbox_file=bbox_file
        self.tokenizer=tokenizer
        self.transforms=transforms

        self.split_csv_data=pd.read_csv(self.csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))
        self.bbox_dict=pickle.load(open(self.bbox_file,"rb"))
        self.caption_dict=pickle.load(open(self.caption_file,"rb"))

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        file_path=self.split_csv_data['file_path'][idx]
        label=self.split_csv_data['label'][idx]
        img_label=self.label_map[label]
        image=Image.open(file_path)

        #read the bbox data 
        box_key=label+"_"+file_path.split("/")[-1]
        bbox=self.bbox_dict[box_key]['boxes']
        probs=self.bbox_dict[box_key]['probs']

        #sort by probability values 
        sorted_idx=np.argsort(probs)[::-1]
        bbox=bbox[sorted_idx[0]]
        bbox=bbox.astype(int)
        image=image.crop(bbox)

        #transform the image
        image=self.transforms(image)
        
        caption=self.caption_dict[box_key]
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

        #mapped label
        mapped_label=self.label_map[label]

        return_dict={'input_ids':input_ids,
            'attn_mask':attn_mask,
            'token_type_ids':token_type_ids,
            'image_array':image,
            'label':mapped_label}

        return(return_dict)

class CAER_Face_Caption_Scene_Feature_AVD_Discrete_dataset(Dataset):

    def __init__(self, tokenizer, csv_file, transforms, bbox_file, label_map_file , scene_folder, caption_file, max_length=512, num_classes=7):

        self.csv_file = csv_file        
        self.label_map_file = label_map_file
        self.num_classes=num_classes
        self.caption_file=caption_file
        self.max_length=max_length
        self.bbox_file=bbox_file
        self.tokenizer=tokenizer
        self.transforms=transforms

        self.split_csv_data=pd.read_csv(self.csv_file)
        self.label_map=pickle.load(open(label_map_file,'rb'))
        self.bbox_dict=pickle.load(open(self.bbox_file,"rb"))
        self.caption_dict=pickle.load(open(self.caption_file,"rb"))
        self.scene_folder=scene_folder

    def __len__(self):
        return(len(self.split_csv_data))

    def __getitem__(self,idx):

        file_path=self.split_csv_data['file_path'][idx]
        label=self.split_csv_data['label'][idx]
        img_label=self.label_map[label]
        image=Image.open(file_path)

        #read the bbox data 
        box_key=label+"_"+file_path.split("/")[-1]
        bbox=self.bbox_dict[box_key]['boxes']
        probs=self.bbox_dict[box_key]['probs']

        #sort by probability values 
        sorted_idx=np.argsort(probs)[::-1]
        bbox=bbox[sorted_idx[0]]
        bbox=bbox.astype(int)
        image=image.crop(bbox)

        #transform the image
        image=self.transforms(image)
        
        caption=self.caption_dict[box_key]

        scene_file=self.scene_folder+"/"+box_key.split(".")[0]+".npy"
        scene_feat=np.load(scene_file)

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

        #mapped label
        mapped_label=self.label_map[label]

        return_dict={'input_ids':input_ids,
            'attn_mask':attn_mask,
            'token_type_ids':token_type_ids,
            'image_array':image,
            'scene_feat': scene_feat,
            'label':mapped_label}

        return(return_dict)



