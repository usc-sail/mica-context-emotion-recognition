from tqdm import tqdm 
import numpy as np 
import torch 
from statistics import mean 
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats.stats import pearsonr
import sys
import os 
import pandas as pd

def gen_validate_score_face_only_model(valid_dl,model,device,discrete_criterion):

    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    log_softmax=nn.LogSoftmax(dim=-1)
    step=0
    loss_list=[]

    with torch.no_grad():
        for i, (return_dict) in tqdm(enumerate(valid_dl)):

            image_array=return_dict['image'].to(device)
            labels=return_dict['label'].to(device)

            logits_discrete=model(image_array)
            loss=discrete_criterion(logits_discrete,labels)
            loss_list.append(loss.item())
            pred=log_softmax(logits_discrete)

            y_pred = torch.max(pred, 1)[1]
            true=true+labels.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()

    valid_accuracy=accuracy_score(true,pred_list)
    f1_score_val=f1_score(true, pred_list, average='macro')  
    precision_score_val=precision_score(true,pred_list,average='macro')
    recall_score_val=recall_score(true,pred_list,average='macro')
    mean_val_loss=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,mean_val_loss)

def gen_validate_score_face_only_model_with_text(valid_dl,model,device,discrete_criterion):

    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    log_softmax=nn.LogSoftmax(dim=-1)
    step=0
    loss_list=[]

    with torch.no_grad():
        for i, (return_dict) in tqdm(enumerate(valid_dl)):

            image_array=return_dict['image'].to(device)
            text_array=return_dict['clip_feature'].to(device)
            labels=return_dict['label'].to(device)

            logits_discrete=model(image_array,text_array)
            loss=discrete_criterion(logits_discrete,labels)
            loss_list.append(loss.item())
            pred=log_softmax(logits_discrete)

            y_pred = torch.max(pred, 1)[1]
            true=true+labels.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()

    valid_accuracy=accuracy_score(true,pred_list)
    f1_score_val=f1_score(true, pred_list, average='macro')  
    precision_score_val=precision_score(true,pred_list,average='macro')
    recall_score_val=recall_score(true,pred_list,average='macro')
    mean_val_loss=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,mean_val_loss)


def gen_validate_score_face_only_model_with_text_MCAN(valid_dl,model,device,discrete_criterion):

    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    log_softmax=nn.LogSoftmax(dim=-1)
    step=0
    loss_list=[]

    with torch.no_grad():
        for i, (return_dict) in tqdm(enumerate(valid_dl)):

            image_array=return_dict['image_array'].to(device)
            input_ids=return_dict['input_ids'].to(device)
            attn_mask=return_dict['attn_mask'].to(device)
            labels=return_dict['label'].to(device)

            logits_discrete=model(image_array,input_ids,attn_mask)
            loss=discrete_criterion(logits_discrete,labels)
            loss_list.append(loss.item())
            pred=log_softmax(logits_discrete)

            y_pred = torch.max(pred, 1)[1]
            true=true+labels.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()
            # if(step==2):
            #     break

            step=step+1

    valid_accuracy=accuracy_score(true,pred_list)
    f1_score_val=f1_score(true, pred_list, average='macro')  
    precision_score_val=precision_score(true,pred_list,average='macro')
    recall_score_val=recall_score(true,pred_list,average='macro')
    mean_val_loss=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,mean_val_loss)


def gen_validate_score_face_model_with_text_scene_MCAN(valid_dl,model,device,discrete_criterion):

    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    log_softmax=nn.LogSoftmax(dim=-1)
    step=0
    loss_list=[]

    with torch.no_grad():
        for i, (return_dict) in tqdm(enumerate(valid_dl)):

            image_array=return_dict['image_array'].to(device)
            input_ids=return_dict['input_ids'].to(device)
            attn_mask=return_dict['attn_mask'].to(device)
            scene_feat=return_dict['scene_feat'].to(device)
            labels=return_dict['label'].to(device)

            logits_discrete=model(image_array,scene_feat,input_ids,attn_mask)
            loss=discrete_criterion(logits_discrete,labels)
            loss_list.append(loss.item())
            pred=log_softmax(logits_discrete)

            y_pred = torch.max(pred, 1)[1]
            true=true+labels.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()
            step=step+1
            # if(step==2):
            #     break

    valid_accuracy=accuracy_score(true,pred_list)
    f1_score_val=f1_score(true, pred_list, average='macro')  
    precision_score_val=precision_score(true,pred_list,average='macro')
    recall_score_val=recall_score(true,pred_list,average='macro')
    mean_val_loss=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,mean_val_loss)


