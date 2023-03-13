import os 
import numpy as np
import pandas as pd 

class Cfgs:
    def __init__(self):
        super(Cfgs, self).__init__()

        #num layers
        self.LAYER=3

        #hidden size
        self.HIDDEN_SIZE = 768

        #multiple heads
        self.MULTI_HEAD = 8

        #dropout rate for all layers
        self.DROPOUT_R = 0.1

        #MLP size in flatten layers
        self.FLAT_MLP_SIZE = 768

        # ------------ Networks setup
        # FeedForwardNet size in every MCA layer
        self.FF_SIZE = int(self.HIDDEN_SIZE * 4)

        # A pipe line hidden size in attention compute
        assert self.HIDDEN_SIZE % self.MULTI_HEAD == 0
        self.HIDDEN_SIZE_HEAD = int(self.HIDDEN_SIZE / self.MULTI_HEAD)

# cfgs=Cfgs()
# print(cfgs.MULTI_HEAD)

    