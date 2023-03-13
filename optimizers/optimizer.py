import torch
from transformers import AdamW, get_linear_schedule_with_warmup

def optimizer_adamW(model,lr,weight_decay):
    #optim_set=AdamW(model.parameters(),lr=lr)
    #default weight decay parameters added
    optim_set=AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    return(optim_set)

def optimizer_adam(model,lr,weight_decay=0):
    optim_set=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    return(optim_set)

def optimizer_SGD(model,lr,momentum=0.9):
    optim_set=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    return(optim_set)

def linear_schedule_with_warmup(optimizer,num_warmup_steps,num_training_steps):
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps, # Default value
                                                num_training_steps=num_training_steps)
    return(scheduler)

def exponential_scheduler(optimizer,gamma):
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
    return(scheduler)

def cosine_annealing_scheduler(optimizer, T_max, eta_min=0,last_epoch=20):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min,last_epoch=last_epoch)
    return(scheduler)