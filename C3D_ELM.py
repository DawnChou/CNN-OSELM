import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from opts import parse_opts
from models import resnet_cut
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset.UCF101 import UCF101
from oselm import OSELM
from utils import Logger, AverageMeter, calculate_accuracy
from training import train_epoch
from validation import val_epoch
import inference


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3

    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    return opt


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    new_dict = {k:v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_dict)
    else:
        model.load_state_dict(new_dict)

    return model

def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())

    spatial_transform.append(ToTensor())

    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = UCF101(opt.root_path,
                        opt.split_path,
                        opt.split_name,
                        spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle= True)

    return train_loader

def get_val_utils(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(LoopPadding(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    val_data = UCF101(opt.root_path,
                        opt.split_path,
                        split_name = "testlist01",
                        n_samples_for_each_video = opt.n_val_samples,
                        spatial_transform = spatial_transform,
                        temporal_transform = temporal_transform)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size= 64, shuffle=False)

    return val_loader

def main(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    #generate model
    model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)

    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)

    train_loader = get_train_utils(opt)
    val_loader = get_val_utils(opt)

    OSELM = OSELM(n_input_nodes = 512, n_hidden_nodes=opt.n_hidden_nodes, n_output_nodes = opt.n_classes)
    num_initialize = opt.num_initialize
    inputs_list = []
    targets_list = []
    for i, (inputs, targets) in train_loader:
        if i < int(num_initialize/64):
            inputs_list.append(inputs)
            targets_list.append(inputs)
        if i == int(num_initialize/64):
            ini_inputs = torch.cat(inputs_list, 0)
            ini_targets = torch.cat(targets_list, 0)
            OSELM.initialize(ini_inputs, ini_targets)
            OSELM.predict(ini_inputs, ini_targets)
        ELM.seq_train(inputs, targets)
        OSELM.predict(inputs, targets)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)
