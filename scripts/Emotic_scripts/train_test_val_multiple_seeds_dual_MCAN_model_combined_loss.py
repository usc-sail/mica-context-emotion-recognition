import pandas as pd 
import numpy as np 
import os 
import sys 
import time 

sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))

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
from mcan_config_slim import *
from dataset import *
from text_person_CM_scene_model import *
from tqdm import tqdm 
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats.stats import pearsonr
from transformers import BertTokenizer, BertModel, BertConfig
from statistics import mean
from scipy.stats.stats import pearsonr
from metrics import *
from eval_scripts import *
from torchvision import transforms, models
import pickle 

#### global config file declaration ######
train_test_val_config_file="/home/dbose_usc_edu/codes/context-emotion-recognition/configs/config_caption_person_MCAN_model_scene_late_fusion_AVD_discrete_train_test_val_multiple_seeds.yaml"
#"/data/digbose92/emotic_experiments/context-emotion-recognition/configs/config_caption_person_MCAN_model_scene_late_fusion_AVD_discrete_train_test_val_multiple_seeds.yaml"
with open(train_test_val_config_file,'r') as f:
    config_data=yaml.safe_load(f)

##### initialize the filenames ###########
train_split_pkl_file=config_data['data']['train_split_total_pkl_file']
val_split_pkl_file=config_data['data']['val_split_total_pkl_file']
test_split_pkl_file=config_data['data']['test_split_pkl_file']
base_folder=config_data['data']['base_folder']
train_csv_file=config_data['data']['train_csv_file']
val_csv_file=config_data['data']['val_csv_file']
test_csv_file=config_data['data']['test_csv_file']
scene_feature_file=config_data['data']['scene_feature_file']
scene_data=pickle.load(open(scene_feature_file,'rb'))
label_map_file=config_data['data']['label_map_file']
pos_weights_file=config_data['data']['pos_weights_file']
tokenizer=BertTokenizer.from_pretrained(config_data['model']['text_model'])
pos_weights=pickle.load(open(pos_weights_file,'rb'))

seed_value_list=[123457,42,84,75,62]
test_loss_multiple_seeds_list=[]
test_map_multiple_seeds_list=[]
test_corr_multiple_seeds_list=[]
val_map_multiple_seeds_list=[]

for seed_value in seed_value_list:
    # global fixing here 
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Run with random seed: %d' %(seed_value))

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
    shuffle_test=config_data['parameters']['test_shuffle']

    train_pds=Person_Scene_Text_Feature_AVD_Discrete_dataset(
            scene_data=scene_data,
            tokenizer=tokenizer,
            split_csv_file=train_csv_file,
            base_folder=base_folder,
            total_split_pkl_file=train_split_pkl_file,
            label_map_file=label_map_file,
            transforms=transforms_img,
            max_length=config_data['parameters']['max_len'],
            num_classes=config_data['model']['n_discrete_classes'])

    val_pds=Person_Scene_Text_Feature_AVD_Discrete_dataset(
            scene_data=scene_data,
            tokenizer=tokenizer,
            split_csv_file=val_csv_file,
            base_folder=base_folder,
            total_split_pkl_file=val_split_pkl_file,
            label_map_file=label_map_file,
            transforms=transforms_img,
            max_length=config_data['parameters']['max_len'],
            num_classes=config_data['model']['n_discrete_classes'])

    test_pds=Person_Scene_Text_Feature_AVD_Discrete_dataset(
            scene_data=scene_data,
            tokenizer=tokenizer,
            split_csv_file=test_csv_file,
            base_folder=base_folder,
            total_split_pkl_file=test_split_pkl_file,
            label_map_file=label_map_file,
            transforms=transforms_img,
            max_length=config_data['parameters']['max_len'],
            num_classes=config_data['model']['n_discrete_classes'])

    train_dl=DataLoader(train_pds,batch_size=batch_size,shuffle=shuffle_train,num_workers=config_data['parameters']['num_workers'])
    val_dl=DataLoader(val_pds,batch_size=batch_size,shuffle=shuffle_val,num_workers=config_data['parameters']['num_workers'])
    test_dl=DataLoader(test_pds,batch_size=batch_size,shuffle=shuffle_test,num_workers=config_data['parameters']['num_workers'])

    if(config_data['model']['person_model_option']=='resnet34'):
        pretrained_model=models.resnet34(pretrained=True)
        feature_model=torch.nn.Sequential(*list(pretrained_model.children())[:-2])

    for param in feature_model.parameters():
        param.requires_grad=False
    
    mcan_config_c=Cfgs()
    model_dict={
            'mcan_config':mcan_config_c,
            'feat_dim':config_data['model']['feat_dim'],
            'scene_feat_dim':config_data['model']['scene_feat_dim'],
            'person_model_option':feature_model,
            'fusion_option':config_data['model']['fusion_option'],
            'num_discrete_classes':config_data['model']['n_discrete_classes'],
            'num_cont_classes':config_data['model']['n_continuous_classes'],
            'text_model': config_data['model']['text_model'] 
    }
    #text_scene_SA_person_fc_model_resnet_finetuned_AVD_Discrete
    model=dual_CM_text_person_scene_late_fusion_model(**model_dict)

    if(config_data['device']['is_cuda']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weights=torch.tensor(pos_weights).to(device)

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
    
    if(config_data['loss']['discrete_loss_option']=='binary_cross_entropy_loss'):
        discrete_criterion=binary_cross_entropy_loss(device=device,pos_weights=pos_weights)
        print(discrete_criterion)

    elif(config_data['loss']['discrete_loss_option']=='multilabel_softmargin_loss'):
        discrete_criterion=multilabel_softmargin_loss(device=device)
        print(discrete_criterion)

    if(config_data['loss']['continuous_loss_option']=='mean_square_loss'):
        continuous_criterion=mean_square_error_loss(device=device)
        print(continuous_criterion)

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

    weight_discrete_loss=config_data['loss']['weight_discrete_loss']
    weight_continuous_loss=config_data['loss']['weight_continuous_loss']

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

    train_loss_stats=[]
    val_loss_stats=[]

    Sig = nn.Sigmoid()
    best_map_score=0
    print('Number of epochs:%d' %(max_epochs))
    
    #ensuring model works here (trains)
    model.train(True)

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

                scene_feat=return_dict['scene_feat'].to(device).float()
                image_array=return_dict['image_array'].to(device)
                cont_emotion=return_dict['cont_emotion'].to(device).float()
                disc_emotion=return_dict['disc_emotion'].to(device).float().squeeze(1)
                input_ids=return_dict['input_ids'].to(device)
                attn_mask=return_dict['attn_mask'].to(device)
                
                optim_example.zero_grad()
                
                logits_cont,logits_discrete=model(image_array,scene_feat,input_ids,attn_mask)
                discrete_loss=discrete_criterion(logits_discrete,disc_emotion)
                cont_loss=continuous_criterion(logits_cont,cont_emotion)
                loss=(weight_discrete_loss*discrete_loss)+(weight_continuous_loss*cont_loss)
                logits_Sig_discrete=Sig(logits_discrete)

                #backprop
                loss.backward()
                optim_example.step()
                train_loss_list.append(loss.item())
                target_labels.append(disc_emotion.cpu())
                pred_labels.append(logits_Sig_discrete.cpu())

                step=step+1
                # if(step==2):
                #     break
                if(step%150==0):
                    logger_step_dict={'Running_Train_loss':mean(train_loss_list)}
                    logger.info("Training loss:{:.3f}".format(loss.item()))
            

            target_label_np=torch.cat(target_labels).detach().numpy()
            pred_label_np=torch.cat(pred_labels).detach().numpy()

            #train stats
            diag_pear_coef = [pearsonr(target_label_np[i, :], pred_label_np[i, :])[0] for i in range(target_label_np.shape[0])]
            train_coef_score=np.mean(diag_pear_coef)
            train_map_list=calculate_stats(pred_label_np,target_label_np)
            map_score_train=np.nanmean(train_map_list)

            logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
            logger.info('Epoch:{:d},Overall Training loss:{:.3f},Overall training MAP:{:.3f}, Overall correlation:{:.3f}'.format(epoch,mean(train_loss_list),map_score_train,train_coef_score))

            logger.info('Evaluating the dataset')
            val_loss,cont_loss,map_score_val,val_corr_score,map_score_list=gen_validate_score_multi_label_model_person_text_scene_fully_ft_discrete_cont(model,val_dl,device,discrete_criterion,continuous_criterion,weight_discrete_loss,weight_continuous_loss)
            logger.info('Epoch:{:d},Overall Validation loss:{:.3f},Overall validation MAP:{:.3f}, Overall correlation:{:.3f}'.format(epoch,val_loss,map_score_val,val_corr_score))

            model.train(True)

            if(map_score_val>best_map_score):
                best_map_score=map_score_val
                logger.info('Saving the best model')
                torch.save(model, os.path.join(sub_folder_model,timestr+'_'+config_data['model']['model_type']+'_best_model_'+str(seed_value)+'.pt'))
                early_stop_cnt=0
            else:
                early_stop_cnt=early_stop_cnt+1
                
            if(early_stop_cnt==early_stop_counter):
                print('Validation performance does not improve for %d iterations' %(early_stop_counter))
                break

    print('Training complete. Resuming testing with current seed')
    model.eval()
    test_loss,cont_test_loss,map_score_test,test_corr_score,map_score_list=gen_validate_score_multi_label_model_person_text_scene_fully_ft_discrete_cont(model,test_dl,device,discrete_criterion,continuous_criterion,weight_discrete_loss,weight_continuous_loss)

    print('Current seed: %d, Test continuous loss: %f, Test map: %f, Test correlation: %f' %(seed_value,cont_test_loss,map_score_test,test_corr_score))

    test_map_multiple_seeds_list.append(map_score_test)
    test_loss_multiple_seeds_list.append(cont_test_loss)
    test_corr_multiple_seeds_list.append(test_corr_score)
    val_map_multiple_seeds_list.append(map_score_val)


df=pd.DataFrame({'Random_seeds':seed_value_list,
            'Test_loss':test_loss_multiple_seeds_list,
            'Test_corr': test_corr_multiple_seeds_list,
             'Val_map':val_map_multiple_seeds_list,
            'Test_map':test_map_multiple_seeds_list})

df.to_csv(os.path.join(model_loc_subfolder,'Test_multiple_seeds_statistics.csv'))
    




