##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Jitesh N. Joshi
## University College London
## jitesh.joshi.20@ucl.ac.uk
## Copyright (c) 2022
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import base_exec_prefix

import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank, is_distributed
# from mmcv.cnn import get_model_complexity_info


class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.train_losses_critic = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.critic_net = None
        self.train_loader = None
        self.val_loader = None

        self.optimizer = None
        self.scheduler = None

        self.running_score = None        
        self._init_model()

    def _init_model(self):

        self.seg_net = self.model_manager.semantic_segmentor()

        # try:
        #     flops, params = get_model_complexity_info(self.seg_net, (3, 512, 512))
        #     split_line = '=' * 30
        #     print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        #         split_line, (3, 512, 512), flops, params))
        #     print('!!!Please be cautious if you use the results in papers. '
        #           'You may need to check if all ops are supported and verify that the '
        #           'flops computation is correct.')
        # except:
        #     pass

        self.seg_net = self.module_runner.load_net(self.seg_net)

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters(self.seg_net)

        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss()

        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

        self.with_gcl = True if self.configer.exists("gcl") else False

        if self.with_gcl:

            self.with_gcl_input = False
            if self.configer.exists('data', 'use_gcl_input'):
                self.with_gcl_input = bool(self.configer.get('data', 'use_gcl_input'))

            self.critic_net = self.model_manager.critic_network()
            self.critic_net = self.module_runner.load_net(self.critic_net)

            Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
            if self.configer.get('optim', 'group_method') == 'decay':
                params_group_critic = self.group_weight(self.critic_net)
            else:
                assert self.configer.get('optim', 'group_method') is None
                params_group_critic = self._get_parameters(self.critic_net)

            self.critic_loss_func = self.loss_manager.get_critic_loss()
            if is_distributed():
                self.critic_loss_func = self.module_runner.to_device(self.critic_loss_func)

            self.gcl_loss_weight = self.configer.get('gcl', 'loss_weight')

            if self.configer.exists("gcl", "warmup_iters"):
                self.gcl_warmup_iters = self.configer.get("gcl", "warmup_iters")
            else:
                self.gcl_warmup_iters = 0

            self.seg_act = nn.LogSoftmax(dim=1)
            self.num_classes = self.configer.get('data', 'num_classes')

            self.tc_cls_ord = torch.arange(0, self.num_classes)
    
            self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group + params_group_critic)

        else:
            self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self, net):
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        params_dict = dict(net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' in key:
                bb_lr.append(value)
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            else:
                nbb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def _generate_fake_segmenation_mask(self, one_hot_target_mask):

        # one_hot_fake_mask = 1 - one_hot_target_mask
        
        # if torch.randn(1) > 0:
        #     one_hot_fake_mask = 1 - one_hot_target_mask
        # else:
        #     tc_rnd_cls = torch.randperm(self.num_classes)
        #     while((tc_rnd_cls == self.tc_cls_ord).any()):
        #         tc_rnd_cls = torch.randperm(self.num_classes)
        #     one_hot_fake_mask = torch.zeros_like(one_hot_target_mask)
        #     one_hot_fake_mask = one_hot_fake_mask[:, tc_rnd_cls, :, :]

        tc_rnd_cls = torch.randperm(self.num_classes)
        while((tc_rnd_cls == self.tc_cls_ord).any()):
            tc_rnd_cls = torch.randperm(self.num_classes)
        one_hot_fake_mask = torch.zeros_like(one_hot_target_mask)
        one_hot_fake_mask = one_hot_fake_mask[:, tc_rnd_cls, :, :]

        return one_hot_fake_mask
        
    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()

        self.critic_net.train()
        self.critic_loss_func.train()

        start_time = time.time()
        
        scaler = torch.cuda.amp.GradScaler()

        if "swa" in self.configer.get('lr', 'lr_policy'):
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()

            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(self.configer.get('iters'), self.scheduler, self.optimizer, backbone_list=[0, ])

            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()

            outputs = self.seg_net(*inputs)
            
            if self.with_gcl_input:
                targets, gcl_input = targets

            # with torch.autograd.set_detect_anomaly(True):            
            # Log.info('targets.shape, min, max: {}, {}, {}'.format(targets.shape, targets.min(), targets.max()))
            targets[targets < 0] = 0  # Resizing and Crop is causing this to -1 - need to resolve
            one_hot_target_mask = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).to(dtype=torch.float32)
            one_hot_fake_mask = self._generate_fake_segmenation_mask(one_hot_target_mask)
            pred_seg_mask = self.seg_act(outputs).exp()
            one_hot_pred_seg_mask = F.one_hot(torch.argmax(pred_seg_mask, dim=1), num_classes=self.num_classes).permute(0, 3, 1, 2).to(dtype=torch.float32)

            if self.with_gcl_input:
                critic_outputs_pred = self.critic_net(one_hot_pred_seg_mask, gcl_input=gcl_input)
                critic_outputs_real = self.critic_net(one_hot_target_mask, gcl_input=gcl_input)
                critic_outputs_fake = self.critic_net(one_hot_fake_mask, gcl_input=gcl_input)
            else:
                critic_outputs_pred = self.critic_net(one_hot_pred_seg_mask)
                critic_outputs_real = self.critic_net(one_hot_target_mask)
                critic_outputs_fake = self.critic_net(one_hot_fake_mask)

            self.foward_time.update(time.time() - foward_start_time)

            loss_start_time = time.time()

            if is_distributed():
                import torch.distributed as dist

                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that 
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp

                with torch.cuda.amp.autocast():
                    loss = self.pixel_loss(outputs, targets, is_eval=False)
                    critic_loss = self.critic_loss_func(critic_outputs_real, critic_outputs_fake, critic_outputs_pred)
                    
                    backward_loss = loss + self.gcl_loss_weight * critic_loss
                    display_loss = reduce_tensor(backward_loss) / get_world_size()
                    display_loss_critic = reduce_tensor(critic_loss) / get_world_size()

            else:
                with torch.cuda.amp.autocast():
                    loss = self.pixel_loss(outputs, targets, is_eval=False, gathered=self.configer.get('network', 'gathered')) 
                    critic_loss = self.critic_loss_func(
                        critic_outputs_real, critic_outputs_fake, critic_outputs_pred, gathered=self.configer.get('network', 'gathered'))
                    display_loss = backward_loss = loss + self.gcl_loss_weight * critic_loss
                    display_loss_critic = critic_loss

            self.train_losses_critic.update(display_loss_critic.item(), batch_size)
            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            backward_start_time = time.time()

            scaler.scale(backward_loss).backward()
            # nn.utils.clip_grad_value_(self.seg_net.parameters(), 0.1)
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and \
                    (not is_distributed() or get_rank() == 0):
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                         'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                         'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\t'
                         'Critic_Loss = {critic_loss.val:.8f} (ave = {critic_loss.avg:.8f})\n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'),
                    self.configer.get('solver', 'display_iter'),
                    self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                    foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                             data_time=self.data_time, loss=self.train_losses, critic_loss=self.train_losses_critic))

                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # save checkpoints for swa
            if 'swa' in self.configer.get('lr', 'lr_policy') and \
                    self.configer.get('iters') > normal_max_iters and \
                    ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or \
                     self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
                self.optimizer.update_swa()

            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            # if self.configer.get('epoch') % self.configer.get('solver', 'test_interval') == 0:
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

        self.configer.plus_one('epoch')

    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                if is_distributed(): dist.barrier()  # Synchronize all processes
                Log.info('{} images processed\n'.format(j))


            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
            if self.with_gcl_input:
                targets, _ = targets

            with torch.no_grad():
                if self.data_helper.conditions.diverse_size:
                    if is_distributed():
                        outputs = [self.seg_net(inputs[i]) for i in range(len(inputs))]
                    else:
                        outputs = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)

                    for i in range(len(outputs)):
                        loss = self.pixel_loss(outputs[i], targets[i].unsqueeze(0))
                        self.val_losses.update(loss.item(), 1)
                        outputs_i = outputs[i]
                        if isinstance(outputs_i, torch.Tensor):
                            outputs_i = [outputs_i]
                        self.evaluator.update_score(outputs_i, data_dict['meta'][i:i + 1])

                else:
                    outputs = self.seg_net(*inputs)

                    try:
                        loss = self.pixel_loss(
                            outputs, targets,
                            gathered=self.configer.get('network', 'gathered')
                        )
                    except AssertionError as e:
                        print(len(outputs), len(targets))

                    if not is_distributed():
                        outputs = self.module_runner.gather(outputs)
                    self.val_losses.update(loss.item(), batch_size)
                    if isinstance(outputs, dict):
                        try:
                            outputs = outputs['seg']
                        except:
                            outputs = outputs['pred']
                    self.evaluator.update_score(outputs, data_dict['meta'])

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.evaluator.update_performance()

        self.configer.update(['val_loss'], self.val_losses.avg)
        self.module_runner.save_net(self.seg_net, save_mode='performance')
        self.module_runner.save_net(self.seg_net, save_mode='val_loss')
        cudnn.benchmark = True

        # Print the log info & reset the states.
        self.evaluator.reduce_scores()
        if not is_distributed() or get_rank() == 0:
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            self.evaluator.print_scores()

        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()
        self.seg_net.train()
        self.pixel_loss.train()

    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
                return
            elif self.configer.get('network', 'resume_train'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
                return
            # return

        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
            return

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train()

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))

    def summary(self):
        from lib.utils.summary import get_model_summary
        import torch.nn.functional as F
        self.seg_net.eval()

        for j, data_dict in enumerate(self.train_loader):
            print(get_model_summary(self.seg_net, data_dict['img'][0:1]))
            return


if __name__ == "__main__":
    pass
