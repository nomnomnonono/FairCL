# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from data_handler.data_utils import split_ssl_data


class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, sens, target, ratio=None):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=str)
        sens_labels = np.loadtxt(attr_path, skiprows=2, usecols=att_list.index(sens) + 1, dtype=int)
        target_labels = np.loadtxt(attr_path, skiprows=2, usecols=att_list.index(target) + 1, dtype=int)
        sens_labels, target_labels = (sens_labels + 1) // 2, (target_labels + 1) // 2
        
        if mode == 'train':
            images = images[:162770]
            sens_labels = sens_labels[:162770]
            target_labels = target_labels[:162770]
        if mode == 'valid':
            images = images[162770:182637]
            sens_labels = sens_labels[162770:182637]
            target_labels = target_labels[162770:182637]
        if mode == 'test':
            images = images[182637:]
            sens_labels = sens_labels[182637:]
            target_labels = target_labels[182637:]
        
        if mode == 'train':
            self.images, self.sens, self.target, self.ulb_images, self.ulb_sens, self.ulb_target = split_ssl_data(images, sens_labels, target_labels, ratio, 2)
        else:
            self.images, self.sens, self.target = images, sens_labels, target_labels

        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])                                       
        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        sens = torch.tensor(self.sens[index])
        return img, sens

    def __len__(self):
        return self.length
