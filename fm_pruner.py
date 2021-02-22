import torch.nn as nn
from lib.thop.profile import register_hooks
import torch
from lib.utils import AverageMeter, accuracy, prGreen,prRed
import numpy as np
from models.vgg_cifar import MaskVGG_IN,MaskVGG,VGG
import time
import copy 
from lib.data import get_split_dataset



class FM_reconstruct:
    def __init__(self,model,repair_points,train_loader,val_loader):
        self.model = model
        self.compressed = model
        self.repair_points = repair_points
        self.train_loader, self.val_loader =  train_loader, val_loader
        self._init_data()
        self._build_index()
        self._extract_layer_information()
        self._collect_XY()

    def copy_pruned(self,pruned):
        m_list = list(self.model.modules())
        mp_list = list(pruned.modules())
        for idxx in self.all_idx:
            mp = mp_list[idxx]
            mo = m_list[idxx]
            
            if type(mp) == nn.BatchNorm2d:
                mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()).cuda())
                mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())
                mp.running_mean.data.copy_(torch.from_numpy(mo.running_mean.data.cpu().numpy()).cuda())
                mp.running_var.data.copy_(torch.from_numpy(mo.running_var.data.cpu().numpy()).cuda())
            if type(mp) == nn.Conv2d:
                mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()).cuda())
                mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())
            if type(mp) == nn.Linear:
                mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()).cuda())
                mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())
        self.compressed = pruned
    
    def _init_data(self):
        # valset picking:
        # for CIFAR, split 5k for val
        # for ImageNet, split 3k for val
        val_size = 5000 if 'cifar' in data_type else 3000
        self.train_loader, self.val_loader, n_class = get_split_dataset(data_type, batch_size,
                                                                        n_data_worker, val_size,
                                                                        data_root,
                                                                        shuffle=False)  # same sampling

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
            
            handler_collection_xy = {}
            if fn is not None: 
                m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
                m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))
                
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
        

    def _flops_preprocessed(self):
        self.conv_related_flops = {}
        for idx,i in enumerate(self.prunable_idx):
            flops = 0
            if idx == 0:
                pass
            else:
                j = i - 1
                while self.layer_type_dict[j] != nn.Conv2d:
                    flops+= self.flops_dict[j]
                    j-=1
            self.conv_related_flops[i] = flops

    def _extract_layer_information(self):

        self.layer_info_dict = dict()
        self.flops_dict = {}
        self.params_dict = {}
        self.flops_list = []
        self.params_list = []

        print('=> Extracting information...')
        input = torch.randn(1, 3, 32, 32).cuda()
        self._add_hook_and_collect(self.model,(input, ))

        # self._flops_preprocessed()
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
        self.org_Outchannels = []
        self.org_Inchannels = []

        modules = list(self.model.modules())
        self.m_list = list(self.model.modules())
        self.prunable_idx = [5, 12,  19, 26, 32,  39]
        for i,mi in enumerate(modules):
            if type(mi) not in list(register_hooks):
                continue
            self.all_idx.append(i)
            # if type(mi) == nn.Conv2d:
            if i in self.prunable_idx:
                self.prunable_ops.append(mi)
                # self.prunable_idx.append(i)
                self.op2idx[mi] = i
                self.idx2op[i] = mi
                self.org_Outchannels.append(mi.out_channels)
                self.org_Inchannels.append(mi.in_channels)
                self.strategy_dict[i] = 1.0
            self.layer_type_dict[i] = type(mi)
        self.identity_strategy_dict = copy.deepcopy(self.strategy_dict)
        

    def validate(self, verbose=True):
        '''
        Validate the performance on validation set
        :param val_loader:
        :param model:
        :param verbose:
        :return:
        '''
        val_loader = self.val_loader
        model = self.compressed
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
        # if acc_metric == 'acc1':
        #     return top1.avg
        # elif acc_metric == 'acc5':
        #     return top5.avg
        # else:
        #     raise NotImplementedError
        
    def compress(self,pruned,strategy):
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
        self.pruned_model(self.model,pruned,mask)
        self.compressed = pruned
        return self.compressed

    def pruned_model(self,origin_model,pruned_model,all_mask):
        m_list = list(origin_model.modules())
        mp_list = list(pruned_model.modules())
        for idx,idxx in enumerate(self.prunable_idx):
            mi = m_list[idxx]
            # replace conv
            mask = all_mask[idx]
            X = self.op_randX[mi][:,mask,:,:].data.cpu().numpy()

            Y = self.op_randY[mi].data.cpu().numpy()
            
            from lib.utils import least_square_sklearn
            N,c,h,w = X.shape
            N,o = Y.shape
            X = X.reshape((N,-1))
            
            W = least_square_sklearn(X,Y)
            mp = mp_list[idxx]
            mo = m_list[idxx]
            W = W.reshape((o,c,h,w))
            mp.weight.data.copy_(torch.from_numpy(W).cuda())
            mp.bias.data.copy_(torch.zeros_like(mp.bias.data).cuda())
            
            # mp = mp_list[idxx]
            # mo = m_list[idxx]
            # mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()[:,mask]).cuda())
            # mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())

            # if idx != 0 :
            j = idxx - 1 
            while True:
                mj = mp_list[j]
                mo = m_list[j]
                if type(mj) == nn.BatchNorm2d:
                    mj.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()[mask]).cuda())
                    mj.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()[mask]).cuda())
                    mj.running_mean.data.copy_(torch.from_numpy(mo.running_mean.data.cpu().numpy()[mask]).cuda())
                    mj.running_var.data.copy_(torch.from_numpy(mo.running_var.data.cpu().numpy()[mask]).cuda())
                if type(mj) == nn.Conv2d:
                    mj.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()[mask]).cuda())
                    mj.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()[mask]).cuda())
                    break
                j-=1
            

            bna = mp_list[idxx+1]
            bnb = m_list[idxx+1]
            bna.weight.data.copy_(torch.from_numpy(bnb.weight.data.cpu().numpy()).cuda())
            bna.bias.data.copy_(torch.from_numpy(bnb.bias.data.cpu().numpy()).cuda())
            bna.running_mean.data.copy_(torch.from_numpy(bnb.running_mean.data.cpu().numpy()).cuda())
            bna.running_var.data.copy_(torch.from_numpy(bnb.running_var.data.cpu().numpy()).cuda())
                
            
        for idx,idxx in enumerate(self.all_idx):
            if idxx > self.prunable_idx[-1]:
                mo = m_list[idxx]
                mp = mp_list[idxx]
                if type(mp) == nn.BatchNorm2d:
                    mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()).cuda())
                    mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())
                    mp.running_mean.data.copy_(torch.from_numpy(mo.running_mean.data.cpu().numpy()).cuda())
                    mp.running_var.data.copy_(torch.from_numpy(mo.running_var.data.cpu().numpy()).cuda())
                elif type(mp) == nn.Linear:
                    mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()).cuda())
                    mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())
                elif type(mp) == nn.Conv2d:
                    mp.weight.data.copy_(torch.from_numpy(mo.weight.data.cpu().numpy()).cuda())
                    mp.bias.data.copy_(torch.from_numpy(mo.bias.data.cpu().numpy()).cuda())
                
    def _collect_XY(self):
        model = self.model
        handler_collection = {}
        types_collection = set()
        # if custom_ops is None:
        custom_ops = {}

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

                # if type(m) != nn.Conv2d:
                #     return
                if m not in self.prunable_ops:
                    return
                
                x = x[0]
                x = torch.nn.functional.pad(x,pad=(1,1,1,1),value = 0)
                B,c,HI,WI = x.shape
                B,n,HO,WO = y.shape
                n,c,kh,kw = m.weight.size()

                # 选出几个点
                points = []
                for i in range(repair_points):
                    rand_x = np.random.choice(list(range(HI-kh+1)))
                    rand_y = np.random.choice(list(range(WI-kw+1)))
                    points.append([rand_x,rand_y])
                
                chosen_X = None
                chosen_Y = None
                for i in range(repair_points):
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
        repair_points = self.repair_points
        train_loader = self.train_loader
        with torch.no_grad():
            for i_b ,(input,target) in enumerate(train_loader):
                if i_b == repair_points:
                    break
                input_var = torch.autograd.Variable(input).cuda()
                _ = model(input_var)


        self.op_input = {}
        self.op_output = {}
        self.op_randX = {}
        self.op_randY = {}
        for op in self.prunable_ops:
            self.op_input[op] = op.input_features
            self.op_output[op] = op.output_features
            self.op_randX[op] = op.sample_X
            self.op_randY[op] = op.sample_Y

        model.train(prev_training_status)
        for m, (xy_handler) in handler_collection.items():
            xy_handler.remove()
            m._buffers.pop("input_features")
            m._buffers.pop("output_features")
            m._buffers.pop("sample_X")
            m._buffers.pop("sample_Y")

if __name__ == '__main__':
    batch_size = 50
    n_data_worker= 1
    data_type = 'cifar10'
    data_root = 'C:\\Users\\lenovo\\dataset\\cifar'
    repair_points = 6

    vgg = VGG('vgg16').cuda()
    vgg.load_state_dict(torch.load(r'C:\Users\lenovo\Desktop\cacp\cacp_vgg\checkpoints\vgg16_cifar10.pt')['state_dict'])
    fm = FM_reconstruct(vgg)
    prun = 0.2
    cfg = [1.0]+[prun,1.0]*6
    action = [prun] * len([5, 12,  19, 26, 32,  39])
    # cfg = [1.0]*13
    pruned = MaskVGG_IN('vgg16',cfg).cuda()
    fm.compress(pruned,action)
    # fm.copy_pruned(pruned)
    fm.validate()
