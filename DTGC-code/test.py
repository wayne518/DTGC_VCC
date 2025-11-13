import torch
import os
import numpy as np
from datasets.crowd_sh import Crowd
from models.dsti import dsti
import argparse
from glob import glob
import cv2
from torch.utils.data import DataLoader
import h5py

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--roi-path', default='',
                        help='roi path')
    parser.add_argument('--frame-number', type=int, default=4,
                        help='the number of input frames')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu··
    model = dsti()
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    if 'fdst' in args.data_dir or 'ucsd' in args.data_dir or 'dronecrowd' in args.data_dir:
        sum_res = []
        datasets = [Crowd(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path)
                    for file in sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                      for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in range(len(file_list)):
            epoch_res = []
            for imgs, keypoints in dataloader[file]:
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(device).squeeze(0)
                with torch.set_grad_enabled(False):
                    output_1, output_2, output_3 = model(imgs)
                    if args.roi_path:
                        mask = np.load(args.roi_path)
                        mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                        mask = torch.tensor(mask).to(device)
                        output_3 = output_3 * mask
                    res = keypoints[0].numpy() - torch.sum(output_3.view(f, -1), dim=1).detach().cpu().numpy()
                    for r in res:
                        epoch_res.append(r)
            epoch_res = np.array(epoch_res)
            if 'fdst' in args.data_dir or 'ucsd' in args.data_dir:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                       key=lambda x: int(x.split('/')[-1].split('.')[0]))
            else:
                test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(test_img_list) % args.frame_number != 0:
                remain = len(test_img_list) % args.frame_number
                epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
            for j, k in enumerate(test_img_list):

                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                if args.roi_path:
                    mask = np.load(args.roi_path)
                    h5_map = h5_map * mask
                count = np.sum(h5_map)

                print(k, epoch_res[j], count, count - epoch_res[j])
            for e in epoch_res:
                sum_res.append(e)
        sum_res = np.array(sum_res)
        mse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
        print(log_str)

    elif 'venice' in args.data_dir:
        sum_res = []
        datasets = [Crowd(args.data_dir+'/'+'test'+'/'+file, is_gray=args.is_gray, method='val',
                          frame_number=args.frame_number, roi_path=args.roi_path)
                    for file in sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)]
        dataloader = [DataLoader(datasets[file], 1, shuffle=False, num_workers=8, pin_memory=False)
                      for file in range(len(os.listdir(os.path.join(args.data_dir, 'test'))))]
        file_list = sorted(os.listdir(os.path.join(args.data_dir, 'test')), key=int)
        for file in range(len(file_list)):
            epoch_res = []
            for imgs, keypoints, masks in dataloader[file]:
                b, f, c, h, w = imgs.shape
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                imgs = imgs.to(device).squeeze(0)
                masks = masks.to(device).squeeze(0)
                with torch.set_grad_enabled(False):
                    output_1, output_2, output_3 = model(imgs)
                    output_3 = output_3 * masks
                    res = keypoints[0].numpy() - torch.sum(output_3.view(f, -1), dim=1).detach().cpu().numpy()
                    for r in res:
                        epoch_res.append(r)
            epoch_res = np.array(epoch_res)
            test_img_list = sorted(glob(os.path.join(args.data_dir+'/'+'test'+'/'+file_list[file], '*.jpg')),
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
            if len(test_img_list) % args.frame_number != 0:
                remain = len(test_img_list) % args.frame_number
                epoch_res = np.delete(epoch_res, slice(-1 * args.frame_number, -1 * remain))
            for j, k in enumerate(test_img_list):

                h5_path = k.replace('jpg', 'h5')
                h5_file = h5py.File(h5_path, mode='r')
                h5_map = np.asarray(h5_file['density'])
                if args.roi_path:
                    mask = np.load(args.roi_path)
                    h5_map = h5_map * mask
                count = np.sum(h5_map)

                print(k, epoch_res[j], count, count - epoch_res[j])
            for e in epoch_res:
                sum_res.append(e)
        sum_res = np.array(sum_res)
        mse = np.sqrt(np.mean(np.square(sum_res)))
        mae = np.mean(np.abs(sum_res))
        log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
        print(log_str)

    else:
        datasets = Crowd(os.path.join(args.data_dir, 'test'), is_gray=args.is_gray, method='val',
                         frame_number=args.frame_number, roi_path=args.roi_path)
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)
        epoch_res = []
        for imgs, keypoints in dataloader:
            b, f, c, h, w = imgs.shape
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            imgs = imgs.to(device).squeeze(0)
            with torch.set_grad_enabled(False):
                output_1, output_2, output_3 = model(imgs)
                if args.roi_path:
                    mask = np.load(args.roi_path)
                    mask = cv2.resize(mask, (mask.shape[1] // 32 * 4, mask.shape[0] // 32 * 4))
                    mask = torch.tensor(mask).to(device)
                    output_3 = output_3 * mask
                res = keypoints[0].numpy() - torch.sum(output_3.view(f, -1), dim=1).detach().cpu().numpy()
                for r in res:
                    epoch_res.append(r)
        epoch_res = np.array(epoch_res)
        test_img_list = sorted(glob(os.path.join(os.path.join(args.data_dir, 'test'), '*.jpg')),
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(test_img_list) % args.frame_number != 0:
            remain = len(test_img_list) % args.frame_number
            epoch_res = np.delete(epoch_res, slice(-1*args.frame_number, -1*remain))
        for j, k in enumerate(test_img_list):

            h5_path = k.replace('jpg', 'h5')
            h5_file = h5py.File(h5_path, mode='r')
            h5_map = np.asarray(h5_file['density'])
            if args.roi_path:
                mask = np.load(args.roi_path)
                h5_map = h5_map * mask
            count = np.sum(h5_map)

            print(os.path.basename(k).split('.')[0], epoch_res[j], count, count-epoch_res[j])
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
        print(log_str)
