import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import lmdb

from fire import Fire
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from global_config import config, modekey
from SiamFC import SiamOGEM
from Dataset.dataset_memory import ImagnetVIDDataset
from SiamFC.custom_transforms import Normalize, ToTensor, RandomStretch, \
    RandomCrop, CenterCrop, RandomBlur, ColorAug

torch.manual_seed(1234)

def train(gpu_id, data_dir):
    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data_memnet.pkl")
    meta_data = pickle.load(open(meta_data_path,'rb'))
    all_videos = [x[0] for x in meta_data]

    # split train/valid dataset
    train_videos, valid_videos = train_test_split(all_videos, 
            test_size=1-config.train_ratio, random_state=config.seed)

    # define transforms
    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
		# Normalize(),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        RandomCrop((config.instance_size, config.instance_size),
                    config.max_translate),
		# Normalize(),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
		# Normalize(),
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
		CenterCrop((config.instance_size, config.instance_size)),
        # Normalize(),
        ToTensor()
    ])

    # open lmdb
    # db = lmdb.open(data_dir+'.lmdb', readonly=True, map_size=int(50e9))

    # create dataset
    train_dataset = ImagnetVIDDataset(train_videos, data_dir,
            train_z_transforms, train_x_transforms)
    valid_dataset = ImagnetVIDDataset(valid_videos, data_dir,
            valid_z_transforms, valid_x_transforms, training=False)
    
    # create dataloader
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
            shuffle=True, pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
            shuffle=False, pin_memory=True, num_workers=config.valid_num_workers, drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)

    # start training
    with torch.cuda.device(gpu_id):
        model = SiamOGEM(gpu_id, mode=modekey.train)
        model.init_weights()
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                momentum=config.momentum, weight_decay=config.weight_decay)
        scheduler = StepLR(optimizer, step_size=config.step_size, 
                gamma=config.gamma)

        for epoch in range(config.epoch):
            train_loss = []
            model.train()
            for i, data in enumerate(tqdm(trainloader)):
                exemplar_img, instance_imgs, e_bbox, i_bboxs = data
                optimizer.zero_grad()
                outputs = model(exemplar_img.cuda(), instance_imgs.cuda(), e_bbox.cuda(), i_bboxs.cuda())
                loss = model.weighted_loss(outputs)
                loss.backward()
                optimizer.step()
                step = epoch * len(trainloader) + i
                summary_writer.add_scalar('train/loss', loss.data, step)
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)

            valid_loss = []
            model.eval()
            for i, data in enumerate(tqdm(validloader)):
                exemplar_img, instance_imgs, e_bbox, i_bboxs = data
                with torch.no_grad():
                    outputs = model(exemplar_img.cuda(), instance_imgs.cuda(), e_bbox.cuda(), i_bboxs.cuda())
                    loss = model.weighted_loss(outputs)
                valid_loss.append(loss.item())
            valid_loss = np.mean(valid_loss)
            print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" %
                    (epoch, valid_loss, train_loss))
            summary_writer.add_scalar('valid/loss', 
                    valid_loss, (epoch+1)*len(trainloader))
            torch.save(model.cpu().state_dict(), 
                    "../trained/SiamOGEM_{}.pth".format(epoch+1))
            model.cuda()
            scheduler.step()
if __name__ == "__main__":
    train(0, '/home/s03/gdh/data/ILSVRC2015_VID_CURRATION')