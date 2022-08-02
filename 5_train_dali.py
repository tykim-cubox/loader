from argparse import ArgumentParser
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed import Backend

import timm

import time
import random
import numpy as np
import os

import albumentations as A
import albumentations.pytorch #<- 이걸 안하면 AttributeError: module 'albumentations' has no attribute 'pytorch'

from set_loader import CustomAlbDataset, DataLoaderX
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

###################### DALI ##################################
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
###############################################################

@pipeline_def(batch_size=128, num_threads=8)
def get_dali_pipeline(data_dir, shard_id, num_shards, dali_cpu=False, crop=256):
  dali_device = 'cpu' if dali_cpu else 'gpu'
  decoder_device = 'cpu' if dali_cpu else 'mixed'

  img_files, labels = fn.readers.file(file_root=data_dir, random_shuffle=False, name="Reader", 
                                      shard_id=shard_id, num_shards=num_shards)
  
  # Load and Crop
  # 이미지 사이즈 힌트
  preallocate_width_hint = 512 if decoder_device == 'mixed' else 0
  preallocate_height_hint = 512 if decoder_device == 'mixed' else 0
  
  # images = fn.decoders.image(img_files, device="mixed")
  # Decode and Random Crop
  images = fn.decoders.image_random_crop(img_files, device=decoder_device, preallocate_width_hint=preallocate_width_hint,
                                         preallocate_height_hint=preallocate_height_hint,
                                         output_type=types.RGB, random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0])
  
  # Resize
  images = fn.resize(images,device=dali_device,resize_x=crop, resize_y=crop,interp_type=types.INTERP_TRIANGULAR)
  # Horizontal F
  images = fn.flip(images, device=dali_device, horizontal=1)


  # Cutout
  axis_names="WH"
  nregions=8
  ndims = len(axis_names)
  args_shape=(ndims*nregions,)
  random_anchor = fn.random.uniform(range=(0., 1.), shape=args_shape)
  random_shape = fn.random.uniform(range=(20., 50), shape=args_shape)
  fn.erase(images, device=dali_device, anchor=random_anchor, shape=random_shape,
            axis_names=axis_names, normalized_anchor=True,
            normalized_shape=False)
  # Normalization
  images = fn.crop_mirror_normalize(images, device=dali_device,
                                    dtype=types.FLOAT,
                                    mean = [0.5023*255, 0.4599*255, 0.3993*255],
                                    std = [0.2553*255, 0.2457*255, 0.2503*255])
  return images, labels.gpu()



def create_dali_loader(data_path, rank, world_size, seed):
    pipe = get_dali_pipeline(data_dir=data_path, device_id=rank, shard_id=rank, num_shards=world_size)
    pipe.build()
    train_loader = DALIGenericIterator(pipe, ['data', 'label'],reader_name='Reader')
    return train_loader
    
    
def create_torch_loader(data_path, rank, world_size, batch_size, p, seed):
    
    albumentations_transform = A.Compose([
        A.RandomCrop(256, 256, p=p),
        A.HorizontalFlip(p=p),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), p=p),
        A.dropout.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=p),
        A.pytorch.ToTensorV2(),])
    
    train_dataset = CustomAlbDataset(data_path, loader_type='jpeg4py', transform=albumentations_transform)
    
    sampler = DistributedSampler(train_dataset, 
                               num_replicas=world_size, # <-- world size만큼
                               rank=rank, # <-- 보통 0번째 device의 rank가 들어감
                               shuffle=True, # <-Must be True
                               seed=seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=8, sampler=sampler, pin_memory=True)
    # train_loader = DataLoaderX(dataset=train_dataset, batch_size=batch_size, 
    #                          shuffle=False, num_workers=8, local_rank=rank, #<--DataLoaderX에는 설정해야함
    #                          sampler=sampler, pin_memory=True)
    return train_loader


def create_model(rank):
    device = torch.device(f'cuda:{rank}')
    
    model = timm.create_model('resnet34', num_classes=3)
    model = model.to(memory_format=torch.channels_last)        
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    
    return model



def main(rank, epochs, model, train_loader) -> nn.Module:
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss().cuda()
    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
  
        for i, data in enumerate(train_loader):

            x, y = data[0]['data'], data[0]['label'].squeeze(-1).long() #<-nn.Cross-entropy를 위해 
            optimizer.zero_grad()
            y_hat = model(x)

            batch_loss = loss(y_hat, y)
            batch_loss.backward()

            optimizer.step()

            batch_loss_scalar = batch_loss.item()
            epoch_loss += batch_loss_scalar / x.shape[0]

        if rank == 0:
            print(f"Epoch={epoch}, train_loss={epoch_loss:.4f}")
    
    if rank == 0:
        end = time.time() - start_time
        print(f"It took {end:0.4f} seconds to train")
    return model.module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=10)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=128)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=0)
    parser.add_argument("--cut_prob", type=float, default=0.5)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.")
    parser.add_argument("--model_filename", type=str, help="Model filename.")
    parser.add_argument("--data_root", type=str, help="Model filename.", default='/home/aiteam/tykim/dataset/afhq/train')
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    
    
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    random_seed = args.random_seed
    model_dir = args.model_dir
    model_filename = args.model_filename
    data_path = args.data_root
    resume = args.resume
    p = args.cut_prob
    seed = 33
    
    rank = local_rank
    world_size = torch.cuda.device_count()
    
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                        init_method='env://')
    torch.cuda.set_device(rank)

    train_loader = create_dali_loader(data_path, rank, world_size, seed)

    model = create_model(rank)
            
    model = main(rank=rank, epochs=epochs, model=model, 
                train_loader=train_loader)
    if rank == 0:
        torch.save(model.state_dict(), 'model.pt')