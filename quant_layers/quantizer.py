import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm
from .utils import fp4_121_scaled,update_scale


class BaseQuantizer(nn.Module):
    def __init__(self, format='fp'):
        super().__init__()
        self.format = format


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x

class MaxMinQuantizer(BaseQuantizer):
    def __init__(self, 
                format='nvfp4',
                stochastic_rounding=False):
        assert format in ['nvfp4', 'mxfp4']
        self.format = format
        if format == 'nvfp4':
            self.block_size = 16
            self.scale_format = 'e4m3'
        self.stochastic_rounding = stochastic_rounding
        self.calibrated = False
        self.dynamic_mode = False
        
        self.scaling_factor = 0
    
    def forward(self, x):
        assert self.calibrated == True
        shape = x.shape
        x = x.reshape(-1, self.block_size)
        x = fp4_121_scaled(x, self.stochastic_rounding, self.scale_format,self.scaling_factor)
        x = x.reshape(shape)
        return x
    
    def calibrate(self, x):
        self.scaling_factor = update_scale(x, self.stochastic_rounding, self.scale_format,self.scaling_factor)
        self.calibrated = True
    

QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "MaxMinQuantizer": MaxMinQuantizer
}

