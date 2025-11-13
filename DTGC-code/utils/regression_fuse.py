from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.DTGC import DTGC
from datasets.crowd_sh import Crowd
from glob import glob
import cv2
import random


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'venice' in args.data_dir or 'dronecrowd' in args.data_dir:
            self.datasets = {x: [Crowd(args.data_dir+'/'+x+'/'+file, args.is_gray, x, args.frame_number,
                                       args.crop_height, args.crop_width, args.roi_path)
                                 for file in sorted(os.listdir(os.path.join(args.data_dir, x)), key=int)]
                             for x in ['train', 'val']}
            self.dataloaders = {x: [DataLoader(self.datasets[x][file],
                                               batch_size=(args.batch_size
                                               if x == 'train' else 1),
                                               shuffle=(True if x == 'train' else False),
                                               num_workers=args.num_workers * self.device_count,
                                               pin_memory=(True if x == 'train' else False))
                                    for file in range(len(os.listdir(os.path.join(args.data_dir, x))))]
                                for x in ['train', 'val']}
        else:
            self.datasets = {x: Crowd(os.path.join(args.data_dir, x), args.is_gray, x, args.frame_number, args.crop_height,
                                      args.crop_width, args.roi_path) for x in ['train', 'val']}
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                              batch_size=(args.batch_size
                                              if x == 'train' else 1),
                                              shuffle=(True if x == 'train' else False),
                                              num_workers=args.num_workers*self.device_count,
                                              pin_memory=(True if x == 'train' else False))
                                for x in ['train', 'val']}
        self.model = DTGC()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)
        self.save_all = args.save_all
        self.num = -1

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % 50 == 0:
                self.num += 1
                self.best_mae = np.inf
                self.best_mse = np.inf
            elif epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        args = self.args
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'venice' in args.data_dir or 'dronecrowd' in args.data_dir:
            file_list = list(range(len(os.listdir(os.path.join(args.data_dir, 'train')))))
            random.shuffle(file_list)
            for file in file_list:
                for step, (imgs, targets, keypoints, mask) in enumerate(self.dataloaders['train'][file]):
                    b0, f0, c0, h0, w0 = imgs.shape
                    assert b0 == 1
                    imgs = imgs.to(self.device).squeeze(0)
                    targets = targets.to(self.device).squeeze(0)
                    mask = mask.to(self.device).squeeze(0)

                    with torch.set_grad_enabled(True):
                        output_1, output_2, output_3 = self.model(imgs)
                        output_1 = output_1 * mask
                        output_2 = output_2 * mask
                        output_3 = output_3 * mask

                        # MSE
                        loss = 1 / (1 * b0 * f0) * (self.criterion(output_1, targets) + self.criterion(output_2, targets) +
                                                    10*self.criterion(output_3, targets))

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        pre_count = torch.sum(output_3.view(f0, -1), dim=1).detach().cpu().numpy()
                        res = pre_count - keypoints[0].numpy()
                        epoch_loss.update(loss.item(), f0)
                        epoch_mse.update(np.mean(res * res), f0)
                        epoch_mae.update(np.mean(abs(res)), f0)

            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time() - epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models
        else:
            for step, (imgs, targets, keypoints, mask) in enumerate(self.dataloaders['train']):
                b0, f0, c0, h0, w0 = imgs.shape
                assert b0 == 1
                imgs = imgs.to(self.device).squeeze(0)
                targets = targets.to(self.device).squeeze(0)
                mask = mask.to(self.device).squeeze(0)

                with torch.set_grad_enabled(True):
                    output_1, output_2, output_3 = self.model(imgs)
                    output_1 = output_1 * mask
                    output_2 = output_2 * mask
                    output_3 = output_3 * mask

                    # MSE
                    loss = 1 / (1 * b0 * f0) * (self.criterion(output_1, targets) + self.criterion(output_2, targets) +
                                                10*self.criterion(output_3, targets))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pre_count = torch.sum(output_3.view(f0, -1), dim=1).detach().cpu().numpy()
                    res = pre_count - keypoints[0].numpy()
                    epoch_loss.update(loss.item(), f0)
                    epoch_mse.update(np.mean(res * res), f0)
                    epoch_mae.update(np.mean(abs(res)), f0)

            logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                 time.time()-epoch_start))
            model_state_dic = self.model.state_dict()
            save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
            torch.save({
                'epoch': self.epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_state_dict': model_state_dic
            }, save_path)
            self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        # Iterate over data.
        if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
            sum_res = []
            file_list = sorted(os.listdir(os.path.join(args.data_dir, 'val')), key=int)
            for file in range(len(file_list)):
                epoch_res = []
                for imgs, keypoints in self.dataloaders['val'][file]:
                    b, f, c, h, w = imgs.shape
                    assert b == 1, 'the batch size should equal to 1 in validation mode'
                    imgs = imgs.to(self.device).squeeze(0)

                    with torch.set_grad_enabled(False):
                        output_1, output_2, output_3 = self.model(imgs)
                        if args.roi_path:
                            mask = np.load(args.roi_path)
                            mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                            mask = torch.tensor(mask).to(self.device)
                            output_3 = output_3 * mask
                        res = keypoints[0].numpy() - torch.sum(output_3.view(f, -1), dim=1).detach().cpu().numpy()
                        for r in res:
                            epoch_res.append(r)
                epoch_res = np.array(epoch_res)
                if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                    val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                          key=lambda x: int(x.split('/')[-1].split('.')[0]))
                else:
                    val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                          key=lambda x: int(x.split('_')[-1].split('.')[0]))
                if len(val_img_list) % args.frame_number != 0:
                    remain = len(val_img_list) % args.frame_number
                    epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
                for e in epoch_res:
                    sum_res.append(e)
            sum_res = np.array(sum_res)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))

        elif 'venice' in args.data_dir:
            sum_res = []
            file_list = sorted(os.listdir(os.path.join(args.data_dir, 'val')), key=int)
            for file in range(len(file_list)):
                epoch_res = []
                for imgs, keypoints, masks in self.dataloaders['val'][file]:
                    b, f, c, h, w = imgs.shape
                    assert b == 1, 'the batch size should equal to 1 in validation mode'
                    imgs = imgs.to(self.device).squeeze(0)
                    masks = masks.to(self.device).squeeze(0)

                    with torch.set_grad_enabled(False):
                        output_1, output_2, output_3 = self.model(imgs)
                        output_3 = output_3 * masks
                        res = keypoints[0].numpy() - torch.sum(output_3.view(f, -1), dim=1).detach().cpu().numpy()
                        for r in res:
                            epoch_res.append(r)
                epoch_res = np.array(epoch_res)
                val_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'val'+'/'+file_list[file], '*.jpg')),
                                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
                if len(val_img_list) % args.frame_number != 0:
                    remain = len(val_img_list) % args.frame_number
                    epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
                for e in epoch_res:
                    sum_res.append(e)
            sum_res = np.array(sum_res)
            mse = np.sqrt(np.mean(np.square(sum_res)))
            mae = np.mean(np.abs(sum_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))

        else:
            epoch_res = []
            for imgs, keypoints in self.dataloaders['val']:
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(self.device).squeeze(0)

                with torch.set_grad_enabled(False):
                    output_1, output_2, output_3 = self.model(imgs)
                    if args.roi_path:
                        mask = np.load(args.roi_path)
                        mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                        mask = torch.tensor(mask).to(self.device)
                        output_3 = output_3 * mask
                    res = keypoints[0].numpy() - torch.sum(output_3.view(f, -1), dim=1).detach().cpu().numpy()
                    for r in res:
                        epoch_res.append(r)
            epoch_res = np.array(epoch_res)
            val_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'val'), '*.jpg')),
                                  key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(val_img_list) % args.frame_number != 0:
                remain = len(val_img_list) % args.frame_number
                epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
            mse = np.sqrt(np.mean(np.square(epoch_res)))
            mae = np.mean(np.abs(epoch_res))
            logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time()-epoch_start))

            model_state_dic = self.model.state_dict()
            if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
                self.best_mse = mse
                self.best_mae = mae
                logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
                if self.save_all:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                    self.best_count += 1
                else:
                    torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.num)))