import os
import pickle
import random

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from dataset.utils import MyWrap, Rotate, Bright, Noise

class ToTensorZones(object):
    def __call__(self, image, zones):
        image = F.to_tensor(image)
        zones = torch.as_tensor(np.array(zones))
        # value for NA area=0, stone=64, glacier=127, ocean with ice melange=254
        zones[zones == 0] = 0
        zones[zones == 64] = 1
        zones[zones == 127] = 2
        zones[zones == 254] = 3
        # class ids for NA area=0, stone=1, glacier=2, ocean with ice melange=3
        return image, zones


class GlacierDataset(Dataset):
    def __init__(self, model_name, mode, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        if mode == 'train':
            self.images_path = os.path.join(parent_dir, "data", "sar_images", "train")
            self.targets_path = os.path.join(parent_dir, "data", "zones", "train")
        elif mode == 'val':
            self.images_path = os.path.join(parent_dir, "data", "sar_images", "val")
            self.targets_path = os.path.join(parent_dir, "data", "zones", "val")
        elif mode == 'test':
            self.images_path = os.path.join(parent_dir, "data", "sar_images", "test")
            self.targets_path = os.path.join(parent_dir, "data", "zones", "test")


        self.imgs = os.listdir(self.images_path)
        self.labels = os.listdir(self.targets_path)

        # if mode == 'train' or mode == 'val':
        #     self.imgs = [img for img in self.imgs if "SI" in img]
        #     self.labels = [label for label in self.labels if "SI" in label]
        #
        # else:
        #     self.imgs = [img for img in self.imgs if "SI" not in img]
        #     self.labels = [label for label in self.labels if "SI" not in label]



        # Sort so images and labels fit together
        self.imgs.sort()
        self.labels.sort()

        if not os.path.exists(os.path.join("data_processing", "data_splits")):
            os.makedirs(os.path.join("data_processing", "data_splits"))
        if not os.path.isfile(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt")):
            shuffle = np.random.permutation(len(self.imgs))
            with open(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt"), "wb") as fp:
                pickle.dump(shuffle, fp)
        else:
            # use already existing shuffle
            with open(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt"), "rb") as fp:
                shuffle = pickle.load(fp)
                # if lengths do not match, we need to create a new permutation
                # 若之前生成的shuffle序列与patch数量不相等，则在线生成与patch数量相等版本的shuffle
                if len(shuffle) != len(self.imgs):
                    shuffle = np.random.permutation(len(self.imgs))
                    with open(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt"), "wb") as fp:
                        pickle.dump(shuffle, fp)

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        tmp = self.imgs[shuffle]
        self.imgs = tmp
        tmp = self.labels[shuffle]
        self.labels = tmp
        self.imgs = list(self.imgs)
        self.labels = list(self.labels)

        self.model_name = model_name
        self.mode = mode
        self.augmentation = augmentation

        self.bright = bright
        self.wrap = wrap
        self.noise = noise
        self.rotate = rotate
        self.flip = flip

        if self.model_name == "CISNet":
            # pseudo_idx
            self.pseudo_idx = [i for i in range(len(self.imgs))]
            random.shuffle(self.pseudo_idx)

        # assert both lists have the same length
        assert len(self.imgs) == len(self.labels), "You don't have the same number of images and masks"

    def custom_to_tensor(self, image, zones):
        to_tensor = ToTensorZones()
        image, mask = to_tensor(image=image, zones=zones)
        return image, mask

    def transform(self, image, mask):
        do_augmentation = self.augmentation
        # ToTensor automatically scales the input to [0, 1]
        image, mask = self.custom_to_tensor(image=image, zones=mask)

        if self.mode == 'train' and do_augmentation:
            if np.random.random() >= (1 - self.flip):
                image = torchvision.transforms.functional.hflip(image)
                mask = torchvision.transforms.functional.hflip(mask)
            if np.random.random() >= (1 - self.rotate):
                rot = Rotate()
                image, mask = rot(image=image, target=mask.unsqueeze(0))
                mask = mask.squeeze(0)
            if np.random.random() >= (1 - self.bright):
                bright = Bright()
                image, mask = bright(image=image, target=mask)
            if np.random.random() >= (1 - self.wrap):
                wrap_transform = MyWrap()
                image, mask = wrap_transform(image=image, target=mask)
            if np.random.random() >= (1 - self.noise):
                noise = Noise()
                image, mask = noise(image=image, target=mask)

        # Z-Score Normalization
        norm = torchvision.transforms.Normalize(mean=0.3047126829624176, std=0.32187142968177795)
        image = norm(image)
        return image, mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        # Assert that image and label name match
        assert img_name.split("__")[0] == label_name.split("__")[0][:-6], "image and label name don't match. Image name: " + img_name + ". Label name: " + label_name
        assert img_name.split("__")[1] == label_name.split("__")[1], "image and label name don't match. Image name: " + img_name + ". Label name: " + label_name
        image = cv2.imread(os.path.join(self.images_path, img_name).__str__(), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.targets_path, label_name).__str__(), cv2.IMREAD_GRAYSCALE)
        x, y = self.transform(image, mask)

        if self.model_name == "CISNet" and self.mode == "train":
            img_name_pseudo = self.imgs[self.pseudo_idx[idx]]
            label_name_pseudo = self.labels[self.pseudo_idx[idx]]
            image_pseudo = cv2.imread(os.path.join(self.images_path, img_name_pseudo).__str__(), cv2.IMREAD_GRAYSCALE)
            mask_pseudo = cv2.imread(os.path.join(self.targets_path, label_name_pseudo).__str__(), cv2.IMREAD_GRAYSCALE)
            x_pseudo, y_pseudo = self.transform(image_pseudo, mask_pseudo)
            return np.concatenate([x, x_pseudo], axis=0), np.concatenate([y[None, :, :], y_pseudo[None, :, :]], axis=0), [img_name, img_name_pseudo]

        return x, y, img_name


class GlacierDataModule(pl.LightningDataModule):

    def __init__(self, model_name, batch_size, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        """
        :param batch_size: batch size
        :param augmentation: Whether or not augmentation shall be performed
        """
        super(GlacierDataModule, self).__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.parent_dir = parent_dir

        self.bright = bright
        self.wrap = wrap
        self.noise = noise
        self.rotate = rotate
        self.flip = flip

        self.glacier_train = None
        self.glacier_val = None
        self.glacier_test = None

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.glacier_test = GlacierDataset(model_name=self.model_name,
                                               mode='test',
                                               augmentation=self.augmentation,
                                               parent_dir=self.parent_dir,
                                               bright=self.bright,
                                               wrap=self.wrap,
                                               noise=self.noise,
                                               rotate=self.rotate,
                                               flip=self.flip)


        if stage == 'fit' or stage is None:
            self.glacier_train = GlacierDataset(model_name=self.model_name,
                                                mode='train',
                                                augmentation=self.augmentation,
                                                parent_dir=self.parent_dir,
                                                bright=self.bright,
                                                wrap=self.wrap,
                                                noise=self.noise,
                                                rotate=self.rotate,
                                                flip=self.flip)

            self.glacier_val = GlacierDataset(model_name=self.model_name,
                                              mode='val',
                                              augmentation=self.augmentation,
                                              parent_dir=self.parent_dir,
                                              bright=self.bright,
                                              wrap=self.wrap,
                                              noise=self.noise,
                                              rotate=self.rotate,
                                              flip=self.flip)

    def train_dataloader(self):
        return DataLoader(self.glacier_train, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.glacier_val, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.glacier_test, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False)

    def prepare_data(self, *args, **kwargs):
        pass
