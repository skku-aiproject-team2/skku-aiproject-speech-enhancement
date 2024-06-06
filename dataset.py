# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio

class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, file_id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0, sample_rate=48000):
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["training", "testing", "validation"]
        self.crop_length_sec = crop_length_sec
        self.sample_rate = sample_rate
        self.subset = subset
        
        N_clean = len(os.listdir(os.path.join(root, 'train/clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'train/noisy')))
        assert N_clean == N_noisy
        
        if subset == "training":
            _p = os.path.join(root, 'train/')
        elif subset == "testing":
            _p = os.path.join(root, 'test/')
        elif subset == "validation":
            _p = os.path.join(root, 'valid/')
        else:
            raise NotImplementedError

        clean_files = sorted(os.listdir(os.path.join(_p, 'clean')))
        noisy_files = sorted(os.listdir(os.path.join(_p, 'noisy')))
        
        self.files = [(os.path.join(_p, 'clean', f), os.path.join(_p, 'noisy', f)) for f in clean_files]

    def __getitem__(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate_clean = torchaudio.load(fileid[0])
        noisy_audio, sample_rate_noisy = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio) and sample_rate_clean == sample_rate_noisy

        # resample audios to self.sample_rate
        if sample_rate_clean != self.sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate_clean, new_freq=self.sample_rate)
            clean_audio = resample(clean_audio)
            noisy_audio = resample(noisy_audio)

        crop_length = int(self.crop_length_sec * self.sample_rate)
        
        if crop_length > len(clean_audio):
            # repeat the audio to match the crop length
            n_repeats = crop_length // len(clean_audio) + 1
            clean_audio = clean_audio.repeat(n_repeats)
            noisy_audio = noisy_audio.repeat(n_repeats)

        # random crop
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]
        
        clean_audio, noisy_audio = clean_audio.unsqueeze(0), noisy_audio.unsqueeze(0)
        return (clean_audio, noisy_audio, fileid)

    def __len__(self):
        return len(self.files)


def load_CleanNoisyPairDataset(root, subset, crop_length_sec, batch_size, sample_rate, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec, sample_rate=sample_rate)                                                       
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
    return dataloader


if __name__ == '__main__':
    import json
    with open('./configs/DNS-large-full.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config["trainset_config"]

    trainloader = load_CleanNoisyPairDataset(**trainset_config, subset='training', batch_size=2, num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config, subset='testing', batch_size=2, num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_audio, noisy_audio, fileid in trainloader: 
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break
    