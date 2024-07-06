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
import torchvision.utils as vutils
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
            self.gen_ulb_images, self.gen_ulb_sens, self.clf_ulb_images, self.clf_ulb_sens = self.images, self.sens, self.images, self.sens
            self.images, self.ulb_images, self.gen_ulb_images, self.clf_ulb_images = self.images.astype(object), self.ulb_images.astype(object), self.gen_ulb_images.astype(object), self.clf_ulb_images.astype(object)
        else:
            self.images, self.sens, self.target = images, sens_labels, target_labels

        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])   
        self.gen_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])           

        self.mode = "labeled"

    def update_generater_training_data(self, indices, labels):
        self.gen_ulb_images = np.concatenate((self.gen_ulb_images, self.ulb_images[indices]))
        self.gen_ulb_sens = np.concatenate((self.gen_ulb_sens, labels))

        self.clf_ulb_images = self.gen_ulb_images
        self.clf_ulb_sens = self.gen_ulb_sens

        self.ulb_images = np.delete(self.ulb_images, indices)
        self.ulb_sens = np.delete(self.ulb_sens, indices)
    
    def update_classifier_training_data(self, images, labels, root):
        self.clf_ulb_sens = np.concatenate((self.clf_ulb_sens, labels))

        for i in range(len(images)):
            vutils.save_image(images[i], os.path.join(root, f"{i}.jpg"), nrow=1, normalize=True, value_range=(0., 1.))
            self.clf_ulb_images = np.append(self.clf_ulb_images, os.path.join(root, f"{i}.jpg"))

    def __getitem__(self, index):
        if self.mode == "labeled":
            img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
            sens = torch.tensor(self.sens[index])
            return img, sens
        elif self.mode == "unlabeled":
            img = self.tf(Image.open(os.path.join(self.data_path, self.ulb_images[index])))
            sens = torch.tensor(self.ulb_sens[index])
            return img, sens, index
        elif self.mode == "gen_semi":
            img = self.tf(Image.open(os.path.join(self.data_path, self.gen_ulb_images[index])))
            sens = torch.tensor(self.gen_ulb_sens[index])
            return img, sens, index
        elif self.mode == "clf_semi":
            if os.path.split(self.clf_ulb_images[index])[0] == "":
                img = self.tf(Image.open(os.path.join(self.data_path, self.clf_ulb_images[index])))
            else:
                img = self.gen_tf(Image.open(self.clf_ulb_images[index]))
            sens = torch.tensor(self.clf_ulb_sens[index])
            return img, sens, index

    def __len__(self):
        if self.mode == "labeled":
            return len(self.images)
        elif self.mode == "unlabeled":
            return len(self.ulb_images)
        elif self.mode == "gen_semi":
            return len(self.gen_ulb_images)
        elif self.mode == "clf_semi":
            return len(self.clf_ulb_images)
