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

def main(config_data):

    #csv and pickle files for train, val and test

    ##### initialize the dataset functions ###########
    train_csv_file=config_data['data']['train_csv_file']
    val_csv_file=config_data['data']['val_csv_file']
    label_map_file=config_data['data']['label_map_file']
    bbox_file=config_data['data']['bbox_file']
    train_clip_file=config_data['data']['train_clip_file']
    val_clip_file=config_data['data']['val_clip_file']
    
    ######## transforms ######
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

    #dataset and dataloader declaration #
    batch_size=config_data['parameters']['batch_size']
    shuffle_train=config_data['parameters']['train_shuffle']
    shuffle_val=config_data['parameters']['val_shuffle']
    
    train_pds=CAER_Face_Caption_Dataset(
            csv_file=train_csv_file,
            label_map_file=label_map_file,
            transforms=transforms_img,
            bbox_file=bbox_file,
            clip_feature_file=train_clip_file)

    val_pds=CAER_Face_Caption_Dataset(
            csv_file=val_csv_file,
            label_map_file=label_map_file,
            transforms=transforms_img,
            bbox_file=bbox_file,
            clip_feature_file=val_clip_file)
    
    
    train_dl=DataLoader(train_pds,batch_size=batch_size,shuffle=shuffle_train)
    val_dl=DataLoader(val_pds,batch_size=batch_size,shuffle=shuffle_val)

    
    if(config_data['model']['person_model_option']=='resnet34'):
        pretrained_model=models.resnet34(pretrained=True)
        feature_model=torch.nn.Sequential(*list(pretrained_model.children())[:-1])

    model_dict={
            'feat_dim':config_data['model']['feat_dim'],
            'face_model':feature_model,
            'num_classes':config_data['model']['n_discrete_classes']
    }
    
    model=CAER_face_caption_lf_model(**model_dict)

    if(config_data['device']['is_cuda']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus=torch.cuda.device_count()
    if(n_gpus>1):
        model=nn.DataParallel(model)
    
    model=model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: %d' %(params))

    ########################################### OPTIMIZER + LOSS FUNCTION + SCHEDULER #####################################
    # #number of epochs
    max_epochs=config_data['parameters']['epochs']
    if(config_data['optimizer']['choice']=='AdamW'):
        optim_example=optimizer_adamW(model,float(config_data['optimizer']['lr']),float(config_data['optimizer']['weight_decay']))

    elif(config_data['optimizer']['choice']=='Adam'): 
        optim_example=optimizer_adam(model,float(config_data['optimizer']['lr']),float(config_data['optimizer']['weight_decay']))

    elif(config_data['optimizer']['choice']=='SGD'): 
        optim_example=optimizer_SGD(model,lr=float(config_data['optimizer']['lr']),momentum=float(config_data['optimizer']['momentum']))


    if(config_data['optimizer']['scheduler']=='cosine_annealing'):
        T_max=config_data['optimizer']['T_max']
        eta_min=config_data['optimizer']['eta_min']
        last_epoch=config_data['parameters']['epochs']
        scheduler=cosine_annealing_scheduler(optim_example, T_max, eta_min=eta_min,last_epoch=-1)
        print(scheduler)

    elif(config_data['optimizer']['scheduler']=='exponential_scheduling'):
        gamma=config_data['optimizer']['gamma']
        scheduler=exponential_scheduler(optim_example,gamma=gamma)
        print(scheduler)

    if(config_data['loss']['discrete_loss_option']=='binary_cross_entropy_loss'):
        discrete_criterion=binary_cross_entropy_loss(device=device,pos_weights=pos_weights)
        print(discrete_criterion)

    elif(config_data['loss']['discrete_loss_option']=='cross_entropy_loss'):
        discrete_criterion=cross_entropy_loss(device=device)
        print(discrete_criterion)

    elif(config_data['loss']['discrete_loss_option']=='multilabel_softmargin_loss'):
        discrete_criterion=multilabel_softmargin_loss(device=device)
        print(discrete_criterion)


    #################### LOGGER + BEST MODEL SAVING INFO HERE ###########################
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename=timestr+'_'+config_data['model']['model_type']+'_log.logs'
    yaml_filename=timestr+'_'+config_data['model']['model_type']+'.yaml'
    log_model_subfolder=os.path.join(config_data['output']['log_dir'],config_data['model']['model_type'])
    if(os.path.exists(log_model_subfolder) is False):
        os.mkdir(log_model_subfolder)
    # #create log folder associated with current model
    sub_folder_log=os.path.join(log_model_subfolder,timestr+'_'+config_data['model']['model_type'])
    if(os.path.exists(sub_folder_log) is False):
        os.mkdir(sub_folder_log)

    # #create model folder associated with current model
    model_loc_subfolder=os.path.join(config_data['output']['model_dir'],config_data['model']['model_type'])
    if(os.path.exists(model_loc_subfolder) is False):
        os.mkdir(model_loc_subfolder)

    sub_folder_model=os.path.join(model_loc_subfolder,timestr+'_'+config_data['model']['model_type'])
    if(os.path.exists(sub_folder_model) is False):
        os.mkdir(sub_folder_model)

    # #dump the current config into a yaml file 
    with open (os.path.join(sub_folder_log,yaml_filename),'w') as f:
        yaml.dump(config_data,f)

    # #logger=Logger(os.path.join(config_data['output']['log_dir'],config_data['model']['option']+'_log.txt'))
    logger = log(path=sub_folder_log, file=filename)
    logger.info('Starting training')

    early_stop_counter=config_data['parameters']['early_stop']
    print('Early stop criteria:%d' %(early_stop_counter))
    early_stop_cnt=0

    log_softmax=nn.LogSoftmax(dim=-1)
    val_f1_best=0
    print('Number of epochs:%d' %(max_epochs))
    
    #return_dict=next(iter(train_dl))

    for epoch in range(1, max_epochs+1): #main outer loop
            train_loss_list=[]
            train_logits=[]
            step=0
            t = time.time()
            target_labels=[]
            pred_labels=[]
            val_loss_list=[]

            for return_dict in tqdm(train_dl):

                image_array=return_dict['image'].to(device)
                clip_feature=return_dict['clip_feature'].to(device)
                labels=return_dict['label'].to(device)
                #token_type_ids=return_dict['token_type_ids'].to(device)
                
                optim_example.zero_grad()
                
                logits_discrete=model(image_array,clip_feature)
                loss=discrete_criterion(logits_discrete,labels)
                
                #backprop
                loss.backward()
                optim_example.step()


                train_loss_list.append(loss.item())
                target_labels.append(labels.cpu())
                train_logits_temp=log_softmax(logits_discrete).to('cpu')
                y_pred=torch.max(train_logits_temp, 1)[1]
                #pred_labels.append(logits_softmax_discrete.cpu())

                train_logits.append(y_pred)
                step=step+1
                # if(step==2):
                #     break
            
                if(step%150==0):
                    logger_step_dict={'Running_Train_loss':mean(train_loss_list)}
                    logger.info("Training loss:{:.3f}".format(loss.item()))
            

            target_label_np=torch.cat(target_labels).detach().numpy()
            train_predictions = torch.cat(train_logits).detach().numpy()

            #scheduler step 
            scheduler.step()

            #train stats

            #compute accuracy and f1 score
            train_accuracy=accuracy_score(target_label_np,train_predictions)
            f1_score_train=f1_score(target_label_np, train_predictions, average='macro')  
            precision_score_train=precision_score(target_label_np,train_predictions,average='macro')
            recall_score_train=recall_score(target_label_np,train_predictions,average='macro')

            logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
            logger.info('\ttrain_loss:{:.3f}, train accuracy:{:.3f}, train f1:{:.3f}'.format(mean(train_loss_list),train_accuracy,f1_score_train))
        

            #evaluate here 
            logger.info('Evaluating the dataset')
            valid_accuracy,f1_score_val,precision_score_val,recall_score_val,val_loss=gen_validate_score_face_only_model_with_text(val_dl,model,device,discrete_criterion)
            logger.info('Validation accuracy:{:.3f},Validation f1:{:.3f}'.format(valid_accuracy,f1_score_val))

            model.train(True)

            if(f1_score_val>val_f1_best):
                val_f1_best=f1_score_val
                logger.info('Saving the best model')
                torch.save(model, os.path.join(sub_folder_model,timestr+'_'+config_data['model']['model_type']+'_best_model.pt'))
                early_stop_cnt=0
            else:
                early_stop_cnt=early_stop_cnt+1
                
            if(early_stop_cnt==early_stop_counter):
                print('Validation performance does not improve for %d iterations' %(early_stop_counter))
                break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Location of configuration data', type=str, required=True)
    args = vars(parser.parse_args())
    config_data=load_config(args['config_file'])
    main(config_data)

