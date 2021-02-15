

from numpy.lib.function_base import select
import torch.nn as nn
import torch
import copy
import numpy as np
from models.vgg_cifar import MaskVGG,VGG



class Pruning:
    def __init__(self,origin_model) -> None:
        super().__init__()
        self.origin_model = copy.deepcopy(origin_model)
        self.get_layer_information()

    def get_masked(self,select):
        self.all_mask = self.preprocess_get_mask(select)
        masked_vgg = MaskVGG('vgg16',select).cuda()
        self.pruned_model(masked_vgg)
        return masked_vgg

    def get_layer_information(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_dict = {}
        self.orgin_channel = []
        self.conv_buffer_dict = {} # layer after the conv
        self.all_idx = []
        self.buffer_conv_map = {}

        i=0
        buffer_temp_idx = []
        model = self.origin_model
        modules = list(model.modules())
        n = len(modules)
        while i < n :
            m = modules[i]

            if type(m) not in [nn.Conv2d,nn.BatchNorm2d,nn.Linear,nn.AvgPool2d,nn.MaxPool2d,nn.ReLU]:
                i+=1
                continue
            else:
                assert type(m) == torch.nn.modules.conv.Conv2d
                self.prunable_ops.append(m)
                self.prunable_idx.append(i)
                self.layer_type_dict[i] = type(m)
                self.orgin_channel.append(m.out_channels) 
                self.all_idx.append(i)
                buffer_temp_idx = []
                while i != n-1:
                    i+=1
                    bu = modules[i]
                    if type(bu) is torch.nn.modules.conv.Conv2d:
                        i-=1
                        break
                    buffer_temp_idx.append(i)
                    self.all_idx.append(i)
                self.conv_buffer_dict[self.prunable_idx[-1]] = copy.deepcopy(buffer_temp_idx)
                for j in buffer_temp_idx:
                    self.buffer_conv_map[j] = self.prunable_idx[-1]
            i+=1
        return None
    # def vgg_masked(self,strategy):
    #     masked_vgg = MaskVGG('vgg16',strategy)
    #     orimo_ls = list(self.orgin_model.modules())
        
    #     for i,m in enumerate(masked_vgg.modules()):
    #         if type(m) in [nn.Conv2d,nn.BatchNorm2d,nn.Linear,nn.AvgPool2d,nn.MaxPool2d,nn.ReLU]:
    #             # type
    #             ty = type(m)

    #             # conv
    #             if ty == nn.Conv2d:
    #                 m.weight.data.copy_(orimo_ls[i].weight.data)
    #                 m.bias.data.copy_(orimo_ls[i].bias.data)
    #             # bn2d
    #             elif ty == nn.BatchNorm2d:
    #                 m.weight.data.copy_(orimo_ls[i].weight.data)
    #                 m.bias.data.copy_(orimo_ls[i].bias.data)
    #                 m.running_mean.data.copy_(orimo_ls[i].running_mean.data)
    #                 m.running_var.data.copy_(orimo_ls[i].running_var.data)
    #             elif ty == nn.Linear:
    #                 # linear
    #                 m.weight.data.copy_(orimo_ls[i].weight.data)
    #             else:# maxpool,avgpool,relu don't need params
    #                 pass
    #     masked_vgg = masked_vgg.cuda()
    #     return masked_vgg


    def pruned_model(self,masked_vgg):
        m_list = list(self.origin_model.modules())
        mp_list = list(masked_vgg.modules())
        for idx,idxx in enumerate(self.prunable_idx):
            
            # replace conv first
            mask = self.all_mask[idx]
            weight = m_list[idxx].weight.data.cpu().numpy()
            bias = m_list[idxx].bias.data.cpu().numpy()
            
            mask_weight = None
            if idx == 0:
                mask_weight = weight[mask,:,:,:]
            else:
                input_mask = self.all_mask[idx-1]
                # select input
                mask_weight = weight[:,input_mask,:,:].reshape(weight.shape[0],-1,weight.shape[2],weight.shape[3])
                # select output
                mask_weight = mask_weight[mask,:,:,:].reshape(-1,mask_weight.shape[1],mask_weight.shape[2],mask_weight.shape[3])
            mask_bias = bias[mask]
            mp = mp_list[idxx]
            mp.weight.data.copy_(torch.from_numpy(mask_weight).cuda())
            mp.bias.data.copy_(torch.from_numpy(mask_bias).cuda())

            # replace other layers
            buffer = self.conv_buffer_dict[idxx]
            for buffer_idx in buffer:
                m = m_list[buffer_idx]
                mp = mp_list[buffer_idx]
                if type(m) == nn.BatchNorm2d:
                    mp.weight.data.copy_(torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda())
                    mp.bias.data.copy_(torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda())
                    mp.running_mean.data.copy_(torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda())
                    mp.running_var.data.copy_(torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda())
                elif type(m) == nn.Linear:
                    mp.weight.data.copy_(torch.from_numpy(m.weight.data.cpu().numpy()[:,mask]).cuda())
    #     print(f'replace cost {ed-st}s')
    def preprocess_get_mask(self,select,method = 'l1'):
        mask = []
        for i,a in enumerate(select):
            
            c = self.orgin_channel[i]
            # if a == 1.0:
            #     mask.append(np.ones(c,bool))
            #     continue
            d = int(c * a)
            mask_ = np.zeros(c,bool)
            weight = self.prunable_ops[i].weight.data.cpu().numpy()
            if method == 'l1':
                importance = np.abs(weight).sum((1, 2, 3))
                sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
                preserve_idx = sorted_idx[:d]  # to preserve index
                mask_[preserve_idx] = True
            mask.append(mask_)
        return mask

    def idx2idxx(self,idx):
        return self.prunable_idx[idx]

    def idxx2idx(self,idxx):
        return self.prunable_idx.index(idxx)


# if __name__ == "__main__":
#     origin_model = VGG('vgg16')
#     origin_model.load_state_dict(torch.load('C:\\Users\\lenovo\Desktop\\cacp\\cacp_vgg\\checkpoints\\vgg16_cifar10.pt')['state_dict'])
#     pruner = Pruning(origin_model)
#     select=[1.0]*13
#     mask_model = pruner.get_masked(select=select)
#     # print(mask_model)
#     from cver import Cifar10_Valider
#     cv = Cifar10_Valider()
#     cv.validate(mask_model)
