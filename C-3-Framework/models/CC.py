import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from misc import layer
from . import SCC_Model

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()

        ccnet = getattr(getattr(SCC_Model, model_name), model_name)
        gs_layer = getattr(layer, 'Gaussianlayer')

        # if model_name == 'AlexNet':
        #     from .SCC_Model.AlexNet import AlexNet as net        
        # elif model_name == 'VGG':
        #     from .SCC_Model.VGG import VGG as net
        # elif model_name == 'VGG_DECODER':
        #     from .SCC_Model.VGG_decoder import VGG_decoder as net
        # elif model_name == 'MCNN':
        #     from .SCC_Model.MCNN import MCNN as net
        # elif model_name == 'CSRNet':
        #     from .SCC_Model.CSRNet import CSRNet as net
        # elif model_name == 'Res50':
        #     from .SCC_Model.Res50 import Res50 as net
        # elif model_name == 'Res101':
        #     from .SCC_Model.Res101 import Res101 as net            
        # elif model_name == 'Res101_SFCN':
        #     from .SCC_Model.Res101_SFCN import Res101_SFCN as net

        self.CCN = ccnet()
        self.gs = gs_layer()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            self.gs = torch.nn.DataParallel(self.gs, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
            self.gs = self.gs.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):                               
        density_map = self.CCN(img)                          
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    
    # NWPU Uses dotmap
    # def forward(self, img, dot_map):
    #     density_map = self.CCN(img)
    #     gt_map = self.gs(dot_map)
    #     self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
    #     return density_map, gt_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

