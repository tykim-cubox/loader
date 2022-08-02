import time
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import os


import cv2
import jpeg4py as jpeg

import albumentations as A
import kornia.augmentation as K


import cv2
from PIL import Image


from functools import partial


def pil_loader(path):
    """
    Returns:
        PIL Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def opencv_loader(path):
    """
    Returns:
        images(np.array [H, W, C])
    """
    image = cv2.imread(path)
    return image

def jpeg4py_loader(path):
    """
    Returns:
        images(np.array [H, W, C])
    """
    image = jpeg.JPEG(path).decode()
    return image

def jp4pil_loader(path):
    """
    Returns:
        images(np.array [H, W, C])
    """
    image = jpeg4py_loader(path)
    return Image.fromarray(image.astype('uint8'), 'RGB')

# def tfms_deco(original_func, image):
#     return original_func(image)

class CustomDataset(Dataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    
    def __init__(self, root_path, transform=None, loader_type='pil'):
        self.file_list = self._make_filelist(root_path)
        self.loader = self._make_loader(loader_type)
        self.loader_type = loader_type
        if transform is not None:
            self.transform = transform
        else:
            self._make_transforms()
        
    # def _setup_transform(self, transform):
    #     """
    #     albumentation은 사용할 때 지정된 signature 그대로 사용해야 함
    #     """
    #     if transform.__class__ == 'albumentations.core.composition.Compose':
    #         def albm_tfms(image):
    #             return transform(image=image)['image']
    #         self.transform = albm_tfms
    #     else:
    #         self.transform = transform
        
    
    def _make_transforms(self):
        if self.loader_type == 'pil':
            self.transform = transforms.Compose([transforms.ToTensor()])
        elif self.loader_type == 'opencv':
            def opencv_tfms(img):
                return torch.from_numpy(img.transpose(2, 0, 1))
            self.transform = opencv_tfms
        elif self.loader_type == 'jpeg4py':
            def jpeg4py_tfms(img):
                return torch.from_numpy(img.transpose(2, 0, 1))
            self.transform = jpeg4py_tfms
        else:
            self.transform = None
    @staticmethod 
    def _make_filelist(root_path):
        classes = [d.name for d in os.scandir(root_path) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        file_list = DatasetFolder.make_dataset(root_path, class_to_idx, CustomDataset.IMG_EXTENSIONS, None)
        return file_list
    
    def _make_loader(self, loader_type):
        if loader_type == 'opencv':
            return opencv_loader
        elif loader_type == 'jpeg4py':
            return jpeg4py_loader
        elif loader_type =='jp4pil':
            return jp4pil_loader
        else:
            return pil_loader

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        
        # File Loading
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class CustomAlbDataset(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        
        # File Loading
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        return img, label


# Kornia and torch vision
class CustomKorDataset(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        
        # File Loading
        img = self.loader(img_path)
        img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float32)
        if self.transform is not None:
            img = self.transform(img)
            
        # image = torch.from_numpy(image['image'].transpose(2,0,1).astype(np.float32))
        return img, label


import numbers
import os
import queue as Queue
import threading
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch