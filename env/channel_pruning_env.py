# Code for "Conditional Automated Channel Pruning for Deep Neural Networks"
# implement by Yixin Liu
# seyixinliu@mail.scut.edu.cn

from logging import raiseExceptions
import time
import torch
from torch._C import Value
import torch.nn as nn
from lib.utils import AverageMeter, accuracy, prGreen,prRed
from lib.data import get_split_dataset
from env.rewards import *
import math

import numpy as np
import copy

from lib.thop.profile import register_hooks


class ChannelPruningEnv:
    """
    Env for channel pruning search
    """
    def __init__(self, model, checkpoint, data, compression_targets, args, n_data_worker=4,
                 batch_size=256, export_model=False, use_new_input=False):
        # default setting
        self.prunable_layer_types = [torch.nn.modules.conv.Conv2d]

        # save options
        self.model = model
        self.model_backup = copy.deepcopy(model)
        self.checkpoint = checkpoint
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data

        # options from args
        self.args = args
        self.lbound = args.lbound
        self.rbound = args.rbound

        # self.use_real_val = args.use_real_val

        # self.n_calibration_batches = args.n_calibration_batches
        # self.n_points_per_layer = args.n_points_per_layer
        self.channel_round = args.channel_round
        self.acc_metric = args.acc_metric
        self.data_root = args.data_root

        # self.export_model = export_model
        # self.use_new_input = use_new_input

        # sanity check
        # assert self.preserve_ratio > self.lbound, 'Error! You can make achieve preserve_ratio smaller than lbound!'

        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        self.n_prunable_layer = len(self.prunable_idx)
        
        
        # extract information for preparing
        self._extract_layer_information()

        
        # extract the X,Y for fm-reconstruction
        self.repair_points = self.args.repair_points
        self.repair_batchs = self.args.repair_batchs
        self.data_saver = []
        self._collect_XY()

        # build embedding (static part)
        self._build_state_embedding()

        
       # multiple compression rate
       
        self.compression_targets = compression_targets
        self.beta = 1.0
        self.cur_beta_idx = -1
        self.beta_total = len(self.compression_targets)

        # build reward
        self.reset()  # restore weight
        self.org_acc = self._validate(self.val_loader, self.model)
        print('=> original acc: {:.3f}%'.format(self.org_acc))
        self.org_model_size = sum(self.params_list)
        print('=> original params size: {:.4f} M '.format(self.org_model_size * 1. / 1e6))
        self.org_flops = sum(self.flops_list)
        print('=> FLOPs:')
        print([self.flops_dict[idx]/1e6 for idx in sorted(self.flops_dict.keys())])
        print('=> original FLOPs: {:.4f} M'.format(self.org_flops * 1. / 1e6))

        self.expected_preserve_computation = [i* self.org_flops for i in self.compression_targets] 

        self.reward = eval(args.reward)

        self.best_reward = -math.inf
        self.best_strategy = None
        self.best_d_prime_list = None

        self.oristate_dict = self.model.state_dict()

        



    def _collect_XY(self):
        # pass
        # with torch.no_grad():
        #     for i_b, (input, target) in enumerate(self.train_loader):  # use image from train set
        #         if i_b == self.repair_batchs:
        #             break
        #         self.data_saver.append((input.clone(), target.clone()))
        #         input_var = torch.autograd.Variable(input).cuda()
        handler_collection = {}
        types_collection = set()
        # if custom_ops is None:
        custom_ops = {}
        model = self.model

        def add_hooks(m: nn.Module):

            m_type = type(m)

            fn = None
            verbose = True
            if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
                fn = custom_ops[m_type]
                if m_type not in types_collection and verbose:
                    print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
            elif m_type in register_hooks:
                fn = register_hooks[m_type]
                if m_type not in types_collection and verbose:
                    print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
            else:
                if m_type not in types_collection and verbose:
                    prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)


            def collect_x_y(m,x,y):

                if type(m) != nn.Conv2d:
                    return
                if  m.weight.size()[1] == 3 : # first layer
                    return
                
                x = x[0]
                x = torch.nn.functional.pad(x,pad=(1,1,1,1),Value = 0)
                B,c,HI,WI = x.shape
                B,n,HO,WO = y.shape
                n,c,kh,kw = m.weight.size()

                # 选出几个点
                points = []
                for i in range(self.repair_points):
                    rand_x = np.random.choice(list(range(HI-kh+1)))
                    rand_y = np.random.choice(list(range(WI-kw+1)))
                    points.append([rand_x,rand_y])
                
                chosen_X = None
                chosen_Y = None
                for i in range(self.repair_points):
                    rand_x,rand_y = points[i]
                    if chosen_X == None:
                        chosen_X = x.clone()[:,:,rand_x:rand_x+kh,rand_y:rand_y+kw]
                        chosen_Y = y.clone()[:,:,rand_x,rand_y]
                    else:
                        chosen_X = torch.cat([chosen_X,x.clone()[:,:,rand_x:rand_x+kh,rand_y:rand_y+kw]])
                        chosen_Y = torch.cat([chosen_Y,y.clone()[:,:,rand_x,rand_y]])

                if m.input_features == None and m.output_features == None:
                    m.input_features = x.clone()
                    m.output_features = y.clone()
                    m.sample_X = chosen_X
                    m.sample_Y = chosen_Y

                else:
                    m.input_features = torch.cat([m.input_features,x.clone()],0)
                    m.output_features = torch.cat([m.output_features,y.clone()],0)
                    m.sample_X = torch.cat([m.sample_X,chosen_X],0)
                    m.sample_Y = torch.cat([m.sample_Y,chosen_Y],0)

                
            if fn is not None: 
                m.register_buffer('input_features', None,persistent=False)
                m.register_buffer('output_features', None,persistent=False)
                m.register_buffer('sample_X', None,persistent=False)
                m.register_buffer('sample_Y', None,persistent=False)
                
                handler_collection[m] = (m.register_forward_hook(collect_x_y))
            types_collection.add(m_type)

        prev_training_status = model.training

        model.eval()
        model.apply(add_hooks)

        with torch.no_grad():
            for i_b ,(input,target) in enumerate(self.train_loader):
                if i_b == self.repair_points:
                    break
                input_var = torch.autograd.Variable(input).cuda()
                _ = self.model(input_var)


        self.op_input = {}
        self.op_output = {}
        self.op_randX = {}
        self.op_randY = {}
        for op in  self.prunable_ops:
            self.op_input[op] = op.input_features
            self.op_output[op] = op.output_features
            self.op_randX[op] = op.sample_X
            self.op_randY[op] = op.sample_Y

        # # collecting X and Y
        # for i,m in enumerate(self.model.modules()):
        #     if i in self.all_idx:
        #         self.params_dict[i] = m.total_params.item()
        #         self.flops_dict[i] = m.total_ops.item()
        #         self.params_list.append(m.total_params.item())
        #         self.flops_list.append(m.total_ops.item())

        model.train(prev_training_status)
        for m, (xy_handler) in handler_collection.items():
            xy_handler.remove()
            m._buffers.pop("input_features")
            m._buffers.pop("output_features")
            m._buffers.pop("sample_X")
            m._buffers.pop("sample_Y")



    def _flops_preprocessed(self):
        self.conv_related_flops = {}
        for i in self.prunable_idx:
            flops = 0
            for j in self.conv_buffer_dict[i]:
                flops += self.flops_dict[j]
            self.conv_related_flops[i] = flops
    
    def vgg_masked(self,strategy):
        # print('action taken:')
        # print(strategy)
        def preprocess_get_mask(select,method = 'l1'):
            mask = []
            for i,a in enumerate(select):
                c = self.org_Inchannels[i]
                d = int(c * a)
                mask_ = np.zeros(c,bool)
                weight = self.prunable_ops[i].weight.data.cpu().numpy()
                if method == 'l1':
                    importance = np.abs(weight).sum((0, 2, 3))
                    sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
                    preserve_idx = sorted_idx[:d]  # to preserve index
                    mask_[preserve_idx] = True
                mask.append(mask_)
            return mask
        mask = preprocess_get_mask(strategy)
        from models.vgg_cifar import MaskVGG
        masked_vgg = MaskVGG('vgg16',strategy).cuda()


        def idx2idxx(idx):
            return self.prunable_idx[idx]
        def idxx2idx(idxx):
            return self.prunable_idx.index(idxx)

        def pruned_model(origin_model,pruned_model,all_mask):
            m_list = list(origin_model.modules())
            mp_list = list(pruned_model.modules())
            st = time.time()
            for idx,idxx in enumerate(self.prunable_idx):
                mi = m_list[idxx]
                # replace conv first
                if idx != len(self.prunable_idx)-1:
                    mask = all_mask[idx]
                    mask_output = all_mask[idx+1]
                X = self.op_randX[mi][:,mask,:,:]
                Y = self.op_randY[mi][:,mask_output]
                from lib.utils import least_square_sklearn
                W = least_square_sklearn(X,Y)
                mp = mp_list[idxx]
                mp.weight.data.copy_(torch.from_numpy(W).cuda())
                # weight = m_list[idxx].weight.data.cpu().numpy()
                # bias = m_list[idxx].bias.data.cpu().numpy()
                
                # mask_weight = None
                # if idx == 0:
                #     mask_weight = weight[mask,:,:,:]
                # else:
                #     input_mask = all_mask[idx-1]
                #     # select input
                #     mask_weight = weight[:,input_mask,:,:].reshape(weight.shape[0],-1,weight.shape[2],weight.shape[3])
                #     # select output
                #     mask_weight = mask_weight[mask,:,:,:].reshape(-1,mask_weight.shape[1],mask_weight.shape[2],mask_weight.shape[3])
                # mask_bias = bias[mask]
                # 
                # mp.bias.data.copy_(torch.from_numpy(mask_bias).cuda())

                # replace other layers
                if idx != 0 :
                    j = idxx - 1 
                    while True:
                        mj = mp_list[j]
                        if type(mj) == nn.BatchNorm2d:
                            mj.weight.data.copy_(torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda())
                            mj.bias.data.copy_(torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda())
                            # mj.running_mean.data.copy_(torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda())
                            # mj.running_var.data.copy_(torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda())
                            break

                        else:
                            j-=1

                        
                # buffer = self.conv_buffer_dict[idxx]
                # for buffer_idx in buffer:
                #     m = m_list[buffer_idx]
                #     mp = mp_list[buffer_idx]
                #     if type(m) == nn.BatchNorm2d:
                        
                        
                        
                        
                #     elif type(m) == nn.Linear:
                #         mp.weight.data.copy_(torch.from_numpy(m.weight.data.cpu().numpy()[:,mask]).cuda())
            ed = time.time()
            # print(f'replace cost {ed-st}s')

        pruned_model(self.model,masked_vgg,mask)
        return masked_vgg
        # from models.vgg_cifar import MaskVGG
        # masked_vgg = MaskVGG('vgg16',strategy)
        
        # for i,m in enumerate(masked_vgg.modules()):
        #     if i in self.all_idx:# is child layer
        #         # type
        #         ty = type(m)

        #         # conv
        #         if ty == nn.Conv2d:
        #             m.weight.data.copy_(self.pruned_weight[i][0])
        #             m.bias.data.copy_(self.pruned_weight[i][1])
        #         # bn2d
        #         elif ty == nn.BatchNorm2d:
        #             m.weight.data.copy_(self.pruned_weight[i][0])
        #             m.bias.data.copy_(self.pruned_weight[i][1])
        #             m.running_mean.data.copy_(self.pruned_weight[i][2])
        #             m.running_var.data.copy_(self.pruned_weight[i][3])
        #         elif ty == nn.Linear:
        #             # linear
        #             m.weight.data.copy_(self.pruned_weight[i][0])
        #         else:# maxpool,avgpool,relu don't need params
        #             pass
        # masked_vgg = masked_vgg.cuda()
        # if self.args.n_gpu > 1:
        #     masked_vgg = torch.nn.DataParallel(masked_vgg, range(self.args.n_gpu))
        # return masked_vgg


    def step(self, action):

        action = self._action_wall(action)  # percentage to preserve
        
        # viturally conduct the pruning process
        action = self.prune_kernel(action,self.cur_ind)
        
        self.strategy.append(action)
        self.strategy_dict[self.prunable_idx[self.cur_ind]] = action

        
        if self._is_final_layer():
            assert len(self.strategy) == len(self.prunable_idx)
            current_flops = self._cur_flops()
            acc_t1 = time.time()
            # mask_t1 = time
            self.model_masked = self.vgg_masked(self.strategy)
            acc = self._validate(self.val_loader, self.model_masked)
            acc_t2 = time.time()
            self.val_time = acc_t2 - acc_t1
            compress_ratio = current_flops * 1. / self.org_flops
            info_set = {'compress_ratio': compress_ratio, 'accuracy': acc, 'strategy': self.strategy.copy(),'d_prime':self.d_prime_list.copy()}
            reward = self.reward(self, acc, current_flops)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy.copy()
                self.best_d_prime_list = self.d_prime_list.copy()
                prGreen('New best reward: {:.4f}, acc: {:.4f}, compress: {:.4f}'.format(self.best_reward, acc, compress_ratio))
                prGreen('New best policy: {}'.format(self.best_strategy))
                prGreen('New best d primes: {}'.format(self.best_d_prime_list))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            # if self.export_model:  # export state dict
            #     torch.save(self.model.state_dict(), self.export_path)
            #     return None, None, None, None
            return obs, reward, done, info_set
        info_set = None
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer

        # build next state (in-place modify)
        self.layer_embedding[self.cur_ind][-4] = self._cur_reduced() * 1. / self.org_flops  # reduced
        if self._is_final_layer:
            self.layer_embedding[self.cur_ind][-3] = 0.0  # rest
        else:
            self.layer_embedding[self.cur_ind][-3] = sum(self.flops_list[self.prunable_idx[self.cur_ind + 1]:]) * 1. / self.org_flops  # rest
        self.layer_embedding[self.cur_ind][-2] = self.strategy[-1]  # last action
        obs = self.layer_embedding[self.cur_ind, :].copy()

        return obs, reward, done, info_set
    

    def reset_model(self):
        self.model = copy.deepcopy(self.model_backup)
        self.model.load_state_dict(self.checkpoint)

    def reset(self):
        self.pruned_weight = {}
        for i in self.all_idx:
            self.pruned_weight[i] = []
        self.d_prime_list = [] # chanel number for each conv
        self.chan_mask = [] # preserve mask for each conv
        self.channel_preseve = [] # preserve channel index for each conv layer
        self.reset_model() # reset model for next epoch
        self.cur_ind = 0
        self.strategy = []  # pruning strategy seq
        self.d_prime_list = []
        self.strategy_dict = copy.deepcopy(self.identity_strategy_dict)
        
        # sampling one compression rate beta
        self.cur_beta_idx = np.random.choice(range(self.beta_total),1)[0]
        self.beta = self.compression_targets[self.cur_beta_idx]
        print(f'compression rate {self.beta} is selected!')
        
        

        # reset layer embeddings
        self.layer_embedding[:, -2] = 1. # a t-1
        self.layer_embedding[:, -3] = 0.# rest
        self.layer_embedding[:, -4] = 0. # reduced
        self.layer_embedding[:,-1] = self.beta  # beta

        obs = self.layer_embedding[0].copy()
        obs[-2] = sum(self.flops_list[1:]) * 1. / sum(self.flops_list)

        self.extract_time = 0
        self.fit_time = 0
        self.val_time = 0

        return obs




    def set_export_path(self, path):
        self.export_path = path

    def prune_kernel(self,preserve_ratio,cur_idx):
        '''make channel belong to [1,c] and k times self.channel_round'''
        # m_list = list(self.model.modules())
        op = self.prunable_ops[cur_idx]
        # idxx = self.prunable_idx[cur_idx]
        
        assert (preserve_ratio <= 1.)

        def format_rank(x):
            rank = int(np.around(x))
            return max(rank, 1)

        # n, c = op.weight.size(0), op.weight.size(1)
        c = op.out_channels 
        
        # 规定通道必须为某参数的整数倍，缩小解空间, 可以问去掉试试效果怎样
        d_prime = format_rank(c * preserve_ratio)
        d_prime = int(np.ceil(d_prime * 1. / self.channel_round) * self.channel_round)
        if d_prime > c:
            d_prime = int(np.floor(c * 1. / self.channel_round) * self.channel_round)

        action = d_prime / c  # calculate the ratio

        # # conduct the pruning
        # weight = op.weight.data.cpu().numpy()
        # bias = op.bias.data.cpu().numpy()
        # # 采用L1范数来筛选通道
        # importance = np.abs(weight).sum((1, 2, 3))
        # sorted_idx = np.argsort(-importance)  # sum magnitude along C_in, sort descend
        # preserve_idx = sorted_idx[:d_prime]  # to preserve index
        # assert len(preserve_idx) == d_prime
        # mask = np.zeros(c, bool)
        # mask[preserve_idx] = True

        # # the weight related to input channel change
        # mask_weight = None
        # if self.cur_ind == 0:
        #     mask_weight = weight[mask,:,:,:]
        # else:
        #     input_mask = self.chan_mask[-1]
        #     # select input
        #     mask_weight = weight[:,input_mask,:,:].reshape(-1,self.d_prime_list[-1],weight.shape[2],weight.shape[3])
        #     # select output
        #     mask_weight = mask_weight[mask,:,:,:].reshape(d_prime,-1,weight.shape[2],weight.shape[3])
        
        # # op.weight.data = torch.from_numpy(mask_weight).cuda()
        # # weight change only related to the output channel
        # mask_bias = bias[mask]
        # # op.bias.data = torch.from_numpy(mask_bias).cuda()

        # # self.pruned_weight[idxx].append(copy.deepcopy(op.weight.data))
        # # self.pruned_weight[idxx].append(copy.deepcopy(op.bias.data))
        # self.pruned_weight[idxx].append(torch.from_numpy(mask_weight).cuda())
        # self.pruned_weight[idxx].append(torch.from_numpy(mask_bias).cuda())

        # # layer after the conv layer
        # buffer = self.conv_buffer_dict[self.prunable_idx[cur_idx]]
        # for buffer_idx in buffer:
        #     m = m_list[buffer_idx]

        #     if type(m) == nn.BatchNorm2d:
        #         # m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda()
        #         # m.bias.data = torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda()
        #         # m.running_mean.data = torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda()
        #         # m.running_var.data = torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda()
        #         self.pruned_weight[buffer_idx].append(torch.from_numpy(m.weight.data.cpu().numpy()[mask]).cuda())
        #         self.pruned_weight[buffer_idx].append(torch.from_numpy(m.bias.data.cpu().numpy()[mask]).cuda())
        #         self.pruned_weight[buffer_idx].append(torch.from_numpy(m.running_mean.data.cpu().numpy()[mask]).cuda())
        #         self.pruned_weight[buffer_idx].append(torch.from_numpy(m.running_var.data.cpu().numpy()[mask]).cuda())
        #     elif type(m) == nn.Linear:
        #         # m.weight.data = torch.from_numpy(m.weight.data.cpu().numpy()[:,mask]).cuda()
        #         self.pruned_weight[buffer_idx].append(torch.from_numpy(m.weight.data.cpu().numpy()[:,mask]).cuda())
            
        # record the preserve idx for the next layer
        # self.channel_preseve.append(copy.deepcopy(preserve_idx))
        # self.chan_mask.append(copy.deepcopy(mask))
        self.d_prime_list.append(d_prime)

        return action

    def _is_final_layer(self):
        return self.cur_ind == len(self.prunable_idx) - 1

    def _is_not_final_layer(self):
        return not  self.cur_ind == len(self.prunable_idx) - 1

    def _conv_pruned_flops(self,idx,last_prev=None,curr_prev=None):
        curr_idx = self.prunable_idx.index(idx)
        if curr_prev is None:
            curr_prev = self.strategy_dict[idx]
        if curr_idx == 0:
            last_prev = 1.0
        else:
            last_idx = self.prunable_idx[curr_idx - 1]

            if last_prev is None:
                last_prev = self.strategy_dict[last_idx]
            
        last_prev_ratio =last_prev
        curr_prev_ratio = curr_prev
        rate = last_prev_ratio*curr_prev_ratio
        return self.flops_dict[idx]*rate

    def _conv_related_pruned_flops(self,idx,curr_prev=None):
        if curr_prev is None:
            curr_prev = self.strategy_dict[idx]
        return self.conv_related_flops[idx] * curr_prev

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind

        action = float(action)
        action = np.clip(action, 0, 1)

        other_comp = 0
        this_comp = 0
        for i, idx in enumerate(self.prunable_idx):

            if i < self.cur_ind:
                other_comp += self._conv_pruned_flops(idx) + self._conv_related_pruned_flops(idx)

            elif i == self.cur_ind:
                this_comp += self._conv_pruned_flops(idx,curr_prev=1.0) + self._conv_related_pruned_flops(idx,curr_prev=1.0)

            elif i == self.cur_ind+1:
                other_comp += self._conv_pruned_flops(idx,last_prev=1.0,curr_prev=self.lbound) + self._conv_related_pruned_flops(idx,curr_prev=self.lbound)
            else:
                other_comp += self._conv_pruned_flops(idx,last_prev=self.lbound,curr_prev=self.lbound) + self._conv_related_pruned_flops(idx,curr_prev=self.lbound)
            
        max_preserve_ratio = (self.expected_preserve_computation[self.cur_beta_idx] - other_comp) * 1. / this_comp

        action = np.minimum(action, max_preserve_ratio)
        action = np.maximum(action,self.lbound)
        # action = np.maximum(action, self.strategy_dict[self.prunable_idx[self.cur_ind]][0])  # impossible (should be)

        return action

    def _get_buffer_flops(self, idx):
        buffer_idx = self.buffer_dict[idx]
        buffer_flop = sum([self.layer_info_dict[_]['flops'] for _ in buffer_idx])
        return buffer_flop

    def _cur_flops(self):
        flops = 0
        for i, idx in enumerate(self.prunable_idx):
            flops += self._conv_pruned_flops(idx) + self._conv_related_pruned_flops(idx)
            # if i < self.cur_ind:
            #     flops += self._conv_pruned_flops(idx) + self._conv_related_pruned_flops(idx)
            # elif i == self.cur_ind:
            #     flops += self._conv_pruned_flops(idx,curr_prev=1.0) + self._conv_related_pruned_flops(idx,curr_prev=1.0)
            # else:
            #     flops += self._conv_pruned_flops(idx,last_prev=1.0,curr_prev=1.0) + self._conv_related_pruned_flops(idx,curr_prev=1.0)
            
        return flops

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_flops - self._cur_flops()
        return reduced

    def _init_data(self):
        # valset picking:
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in self.data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(self.data_type, self.batch_size,
                                                                        self.n_data_worker, val_size,
                                                                        data_root=self.data_root,
                                                                        shuffle=False)  # same sampling
        # if self.use_real_val:  # use the real val set for eval, which is actually wrong
        #     print('*** USE REAL VALIDATION SET!')

    def _build_index(self):
        self.prunable_idx = []
        self.prunable_ops = []
        self.layer_type_dict = {}
        self.org_channels = []
        self.conv_buffer_dict = {} # layer after the conv
        self.all_idx = []
        self.buffer_conv_map = {}
        self.strategy_dict = {}
        self.op2idx = {}
        self.idx2op = {}

        modules = list(self.model.modules())
        self.m_list = list(self.model.modules())
        for i,mi in enumerate(modules):
            if type(mi) not in list(register_hooks):
                continue
            self.all_idx.append(i)
            if type(mi) == nn.Conv2d:
                self.prunable_ops.append(mi)
                self.prunable_idx.append(i)
                self.op2idx[mi] = i
                self.idx2op[i] = mi
                self.org_Outchannels.append(mi.out_channels)
                self.org_Inchannels.append(mi.in_channels)
                self.strategy_dict[i] = 1.0
            self.layer_type_dict[i] = type(mi)
        self.identity_strategy_dict = copy.deepcopy(self.strategy_dict)
        
        

    def _add_hook_and_collect(self,model: nn.Module, inputs, custom_ops=None, verbose=True):
        handler_collection = {}
        types_collection = set()
        if custom_ops is None:
            custom_ops = {}

        def add_hooks(m: nn.Module):

            m_type = type(m)

            fn = None
            if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
                fn = custom_ops[m_type]
                if m_type not in types_collection and verbose:
                    print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
            elif m_type in register_hooks:
                fn = register_hooks[m_type]
                if m_type not in types_collection and verbose:
                    print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
            else:
                if m_type not in types_collection and verbose:
                    prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

            
            def count_parameters(m, x, y):
                total_params = 0
                for p in m.parameters():
                    total_params += torch.DoubleTensor([p.numel()])
                m.total_params[0] = total_params

            # def collect_x_y(m,x,y):
            #     m.input_features = x.clone()
            #     m.output_features = y.clone()
            
            handler_collection_xy = {}
            if fn is not None: 
                m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
                m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
                # m.register_buffer('input_features', None,persistent=False)
                # m.register_buffer('output_features', None,persistent=False)
                
                handler_collection[m] = (m.register_forward_hook(fn), m.register_forward_hook(count_parameters))
            types_collection.add(m_type)

        prev_training_status = model.training

        model.eval()
        model.apply(add_hooks)

        with torch.no_grad():
            model(*inputs)

        # collecting flops and params
        for i,m in enumerate(self.model.modules()):
            if i in self.all_idx:
                self.params_dict[i] = m.total_params.item()
                self.flops_dict[i] = m.total_ops.item()
                self.params_list.append(m.total_params.item())
                self.flops_list.append(m.total_ops.item())

        model.train(prev_training_status)
        for m, (op_handler, params_handler) in handler_collection.items():
            op_handler.remove()
            params_handler.remove()
            m._buffers.pop("total_ops")
            m._buffers.pop("total_params")
        

    def _extract_layer_information(self):

        self.layer_info_dict = dict()
        self.flops_dict = {}
        self.params_dict = {}
        self.flops_list = []
        self.params_list = []

        print('=> Extracting information...')
        input = torch.randn(1, 3, 32, 32).cuda()
        self._add_hook_and_collect(self.model,(input, ))

        self._flops_preprocessed()


    def _build_state_embedding(self):
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model.modules())
        for i, ind in enumerate(self.prunable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                this_state.append(i)  # index
                this_state.append(0)  # layer type, 0 for conv
                this_state.append(m.in_channels)  # in channels
                this_state.append(m.out_channels)  # out channels
                this_state.append(m.stride[0])  # stride
                this_state.append(m.kernel_size[0])  # kernel size
                this_state.append(np.prod(m.weight.size()))  # weight size
            # elif type(m) == nn.Linear:
            #     this_state.append(i)  # index
            #     this_state.append(1)  # layer type, 1 for fc
            #     this_state.append(m.in_features)  # in channels
            #     this_state.append(m.out_features)  # out channels
            #     this_state.append(0)  # stride
            #     this_state.append(1)  # kernel size
                # this_state.append(np.prod(m.weight.size()))  # weight size

            # this 4 features need to be changed later
            this_state.append(0.)  # reduced
            this_state.append(0.)  # rest
            this_state.append(1.)  # a_{t-1}
            this_state.append(1.0) # beta 
            layer_embedding.append(np.array(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape


        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _validate(self, val_loader, model, verbose=False):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        criterion = nn.CrossEntropyLoss().cuda()
        # switch to evaluate mode
        model.eval()
        end = time.time()

        t1 = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(non_blocking=True)
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target).cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f    top1: %.3f    top5: %.3f    time: %.3f' %
                  (losses.avg, top1.avg, top5.avg, t2 - t1))
        if self.acc_metric == 'acc1':
            return top1.avg
        elif self.acc_metric == 'acc5':
            return top5.avg
        else:
            raise NotImplementedError
