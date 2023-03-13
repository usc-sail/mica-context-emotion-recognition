import pandas as pd 
import numpy as np 
import os 
import sys 
import time 

sys.path.append(os.path.join('../..', 'datasets'))
sys.path.append(os.path.join('../..', 'models'))
sys.path.append(os.path.join('../..', 'configs'))
sys.path.append(os.path.join('../..', 'losses'))
sys.path.append(os.path.join('../..', 'optimizers'))
sys.path.append(os.path.join('../..', 'utils'))

#import libraries
from ast import literal_eval
import torch
import yaml
import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from optimizer import *
from loss_functions import *
from log_file_generate import *
from CAER_dataset import *
from CAER_model import *
from tqdm import tqdm 
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats.stats import pearsonr
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import CLIPTokenizer, CLIPModel
from statistics import mean
from scipy.stats.stats import pearsonr
from metrics import *
from eval_CAER_scripts import *
from torchvision import transforms, models
import pickle 

#torch.autograd.set_detect_anomaly(True)
seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_file):
    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

#device and model parameters
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_file="/bigdata/digbose92/CAER/model_dir/face_resnet_fully_finetuned_caption_MCAN_cross_entropy_loss/20221006-015519_face_resnet_fully_finetuned_caption_MCAN_cross_entropy_loss/20221006-015519_face_resnet_fully_finetuned_caption_MCAN_cross_entropy_loss_best_model.pt"
config_file="/bigdata/digbose92/CAER/log_dir/face_resnet_fully_finetuned_caption_MCAN_cross_entropy_loss/20221006-015519_face_resnet_fully_finetuned_caption_MCAN_cross_entropy_loss/20221006-015519_face_resnet_fully_finetuned_caption_MCAN_cross_entropy_loss.yaml"

model=torch.load(model_file)
model.eval()
model=model.to(device)
config_data=load_config(config_file)

label_map_file="/bigdata/digbose92/CAER/pkl_files/emotion_map_dict.pkl"
with open(label_map_file,'rb') as f:
    label_map=pickle.load(f)

label_map={v:k for k,v in label_map.items()}
print(label_map)


size=config_data['transforms']['size']
mean_img=config_data['transforms']['mean']
std_img=config_data['transforms']['std']
transforms_img= transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_img,
            std=std_img
        )
    ]) 

test_csv_file="/bigdata/digbose92/CAER/csv_files/CAER_test_file_list_no_missing_bbox.csv"
label_map_file="/bigdata/digbose92/CAER/pkl_files/emotion_map_dict.pkl"
bbox_file="/bigdata/digbose92/CAER/pkl_files/test_mtcnn_boxes.pkl"
caption_file="/bigdata/digbose92/CAER/pkl_files/test_LAVIS_caption_renamed.pkl"
tokenizer=BertTokenizer.from_pretrained(config_data['model']['text_model'])

#dataset and dataloader declaration #
batch_size=config_data['parameters']['batch_size']

    
test_pds=CAER_Face_Caption_Feature_AVD_Discrete_dataset(
            tokenizer=tokenizer,
            csv_file=test_csv_file,
            label_map_file=label_map_file,
            transforms=transforms_img,
            bbox_file=bbox_file,
            caption_file=caption_file,
            max_length=config_data['parameters']['max_len'],
            num_classes=config_data['model']['n_discrete_classes'])

test_dl=DataLoader(test_pds,batch_size=batch_size,shuffle=False)
discrete_criterion=cross_entropy_loss(device=device)

test_accuracy,f1_score_test,precision_score_test,recall_score_test,test_loss=gen_validate_score_face_only_model_with_text_MCAN(test_dl,model,device,discrete_criterion)
print('Test accuracy:{:.3f},Test f1:{:.3f},Test precision: {:.3f}, Test recall: {:.3f}'.format(test_accuracy,f1_score_test,precision_score_test,recall_score_test))



