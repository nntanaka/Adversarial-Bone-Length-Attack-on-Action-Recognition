#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import pickle
import numpy as np
from numpy.lib.format import open_memmap
import os
import time

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum((self.meta_info['epoch']*self.arg.iteration)>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def L_2_const(self, beta, thres):
        norm = torch.norm(beta-1)
        if norm > thres:
            beta *= torch.sqrt(thres)/norm
        return beta

    def L_inf_const(self, beta, thres, two_human):
        beta =  torch.clamp(beta, 1-thres, 1+thres)
        return beta

    def adam_attack(self, data, label, thres, alpha, ind, two_humans):
        niter = self.arg.iteration
        beta = torch.ones(2, 24, dtype=torch.float, requires_grad=True, device='cuda')
        opt = optim.Adam([beta], lr=alpha, weight_decay=self.arg.ad_weight_decay)
        clean_data = data.clone()
        if self.arg.constraint == 'L_2':
            const = self.L_2_const
        else:
            const = self.L_inf_const
        fool = 0

        
        for i in range(niter):
            for k, bp in enumerate(self.ntu_skeleton_bone_pairs):
                data[0, :,:,bp[1], :] = beta[:, k]*(clean_data[0, :, :, bp[1], :] - clean_data[0, :, :, bp[0], :]) + data[0, :, :, bp[0], :]
            output = self.model(data)

            pred = output.max(1, keepdim=True)[1].item()
            
            if pred != label.item() and fool==0:
                fool = 1
                print('Adversarial attack was successful!')
                return (pred, data.detach().clone(), fool)
            loss_ad = -self.loss(output, label)
            opt.zero_grad()
            loss_ad.backward()
            opt.step()
            beta.data = const(beta.detach(), thres, two_humans)
            data = data.detach()
        
        for k, bp in enumerate(self.ntu_skeleton_bone_pairs):
            data[0, :,:,bp[1], :] = beta[:, k]*(clean_data[0, :, :, bp[1], :] - clean_data[0, :, :, bp[0], :]) + data[0, :, :, bp[0], :]
        output = self.model(data)

        if fool == 0:
            pred = output.max(1, keepdim=True)[1].item()
            
            if pred != label.item():
                fool=1
                print('Adversarial attack was successful!')
            else:
                fool=0
                print('Adversarial attack failed.')

        return (pred, data.detach().clone(), fool)
    
    def pgd_attack(self, data, label, thres, alpha, ind, two_humans):
        niter = self.arg.iteration
        beta = torch.ones(2, 24, dtype=torch.float, requires_grad=True, device='cuda')
        clean_data = data.clone()
        if self.arg.constraint == 'L_2':
            const = self.L_2_const
        else:
            const = self.L_inf_const
        fool = 0
        
        for i in range(niter):
            for k, bp in enumerate(self.ntu_skeleton_bone_pairs):
                data[0, :,:,bp[1], :] = beta[:, k]*(clean_data[0, :, :, bp[1], :] - clean_data[0, :, :, bp[0], :]) + data[0, :, :, bp[0], :]
            output = self.model(data)

            pred = output.max(1, keepdim=True)[1].item()
            
            if pred != label.item() and fool==0:
                fool = 1
                print('Adversarial attack was successful!')
                return (pred, data.detach().clone(), fool)
            loss_ad = self.loss(output, label)
            loss_ad.backward()
            beta.data = beta + alpha * torch.sign(beta.grad)
            beta.grad.zero_()
            beta.data = const(beta.detach(), thres, two_humans)
            data = data.detach()

        
        for k, bp in enumerate(self.ntu_skeleton_bone_pairs):
            data[0, :,:,bp[1], :] = beta[:, k]*(clean_data[0, :, :, bp[1], :] - clean_data[0, :, :, bp[0], :]) + data[0, :, :, bp[0], :]
        output = self.model(data)

        if fool == 0:
            pred = output.max(1, keepdim=True)[1].item()
            
            if pred != label.item():
                fool=1
                print('Adversarial attack was successful!')
            else:
                fool=0
                print('Adversarial attack failed.')
                
        return  (pred, data.detach().clone(), fool)
            
    def adversarial_bone_length_attack(self):
        self.ntu_skeleton_bone_pairs = list((i-1, j-1) for (i,j) in (
            (1, 2), (2, 21), (21, 3), (21, 9),
            (21, 5), (1, 13), (1, 17), (3, 4),
            (13, 14), (17, 18), (5, 6), (9, 10),
            (14, 15), (18, 19), (6, 7), (10, 11),
            (7, 8), (11, 12), (8, 23), (12, 25),
            (15, 16), (19, 20), (8, 22), (12, 24)))
        
        self.model.eval()
        loader = self.data_loader['test']
        size = len(loader.dataset)
        
        method_name = self.arg.constraint + '_' + self.arg.adversarial_op
        if self.arg.adversarial_op == 'pgd':
            adversarial_attack = self.pgd_attack
        else:
            adversarial_attack = self.adam_attack
        for alpha in [self.arg.step_size]:
            for thres in self.arg.threshold:
                start_time = time.time()
                fools = 0
                work_dir = './{:s}/results_ad/{}/alpha{:0>5d}'.format(self.arg.xs_or_xv, method_name, int(alpha*10000))
                log_dir = './{:s}/logs_ad/{}/alpha{:0>5d}'.format(self.arg.xs_or_xv, method_name, int(alpha*10000))
                if not os.path.exists(work_dir):
                    os.makedirs(work_dir)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                ind = 0
                fp = open_memmap(os.path.join(work_dir, 'iter_{:d}_thres{:.2f}.npy'.format(self.arg.iteration, thres)), dtype='float32', mode='w+', shape=(size, 3, 300, 25, 2))
                
                labels = list()
                fool_list = list()
                for data, label in loader:
                    if ind % 500 == 0:
                        print('{} / {}'.format(ind, size))
                    
                    two_humans = True
                    if torch.equal(data[0,:,:,:,1], torch.zeros((3,300,25), dtype=torch.float)):
                        two_humans = False
                    
                    data = data.float().to(self.dev)
                    label = label.long().to(self.dev)
                    
                    
                    pred_label, ad_skeleton, fool = adversarial_attack(data, label, thres, alpha, ind, two_humans)
                    fp[ind, :,:,:,:] = ad_skeleton.cpu().data.numpy()
                    if fool==1:
                        fool_list.append(ind)
                    labels.append(pred_label)
                    fools += fool
                    ind += 1
                
                fl_rate = fools/ind
                with open(os.path.join(work_dir, 'iter_{:d}_thres{:.2f}label.pkl'.format(self.arg.iteration, thres)), 'wb') as f:
                    pickle.dump(labels, f)
                with open(os.path.join(work_dir, 'iter_{:d}_thres{:.2f}fool_index.txt'.format(self.arg.iteration, thres)), 'w') as f:
                    [f.write('{}\n'.format(fool_i)) for fool_i in fool_list]
                fin_time = time.time()
                total_time = int((fin_time - start_time)/60)
                total_hour = total_time//(60)
                total_minute = total_time-60*total_hour
                with open(os.path.join(log_dir, 'log.txt'), 'a') as f:
                    f.write('step size: {}, threshold: {}, iteration: {}, optimizer: {}, constraint: {}'.format(alpha, thres, self.arg.iteration, self.arg.adversarial_op, self.arg.constraint))
                    if self.arg.adversarial_op == 'adam':
                        f.write(', weight_decay: {}'.format(self.arg.ad_weight_decay))
                    f.write('\n{} hours. {} minutes., fooling rate: {}\n\n'.format(total_hour, total_minute, fl_rate))
                with open(os.path.join(work_dir, 'fooling_rate.txt'), 'a') as f:
                    f.write('{} '.format(fl_rate))   
            
    def get_correctly_classified_data(self):
        self.model.eval()
        loader = self.data_loader['test']
        data_list = list()
        label_list = list()
        p = 0
        f = 0
        i = 0
        size = len(loader.dataset)
        for data, label in loader:
            cpu_data = data.cpu()
            cpu_label = label.cpu()
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            output = torch.nn.Softmax(dim=1)(output)
            result = torch.max(output, dim=1).item()
            if result[1][0] == label:
                data_list.append(data)
                label_list.appeend(label)
            
        fp = open_memmap('./data/NTU-RGB-D/{:s}/correctly_classified_data.npy'.format(self.arg.xs_or_xv), dtype='float32', mode='w+', shape=(len(label_list), 3, 300, 25, 2))
        fp[:,:,:,:,:] = np.stack(data_list)

        with open('./data/NTU-RGB-D/{:s}/correctly_classified_label.pkl'.format(self.arg.xs_or_xv),'wb') as f:
            pickle.dump(label_list, f)

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
            return np.mean(loss_value)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--iteration', type=int, help='iteration number for adversarial attack')
        parser.add_argument('--adversarial_op', choices=['adam', 'pgd'], default='pgd', help='use Adam or PGD')
        parser.add_argument('--xs_or_xv', choices=['xsub', 'xview'], default='xsub', help='use X-subject or X-view')
        parser.add_argument('--ad_weight_decay', type=float, default=0.0, help='weight decay for adam in adversarial attack')
        parser.add_argument('--constraint', choices=['L_2', 'L_inf'], default='L_inf', help='use L_inf norm or L_2 norm')
        parser.add_argument('--step_size', type=float, default=0.05, help='step size for adversarial attack')
        parser.add_argument('--threshold', type=float, default=[], nargs='+', help='threshold for L_x constraint')
        # endregion yapf: enable

        return parser
