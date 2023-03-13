import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd 
from collections import Counter
import math

def binary_cross_entropy_loss(device,pos_weights,reduction='mean'):
    """Binary cross entropy loss."""
    return nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weights).to(device)

def multilabel_softmargin_loss(device,reduction='mean'):
    """ multi label soft margin loss """
    return(nn.MultiLabelSoftMarginLoss(reduction=reduction).to(device))

def mean_square_error_loss(device,reduction='mean'):
    """ mean square error loss """
    return(nn.MSELoss(reduction=reduction).to(device))

def cross_entropy_loss(device,reduction='mean'):
    """ cross entropy loss """
    return(nn.CrossEntropyLoss(reduction=reduction).to(device))
    
    