# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import sys
from collections import defaultdict
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile 

from pesq import pesq
from pystoi import stoi
import torch
import torchaudio


def evaluate(clean_path, denoised_path):
    clean_files = os.listdir(clean_path)
    denoised_files = os.listdir(denoised_path)

    result = defaultdict(int)

    # metric : peq_wb, pesq_nb, stoi
    for i in tqdm(range(len(clean_files))):
        rate, clean = wavfile.read(os.path.join(clean_path, clean_files[i]))
        _, denoised = wavfile.read(os.path.join(denoised_path, denoised_files[i]))

        clean = torch.tensor(clean, dtype=torch.float32)
        denoised = torch.tensor(denoised, dtype=torch.float32)
        
        resample = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        clean = resample(clean)
        denoised = resample(denoised)

        length = clean.shape[-1]

        result['pesq_wb'] += pesq(16000, clean, denoised, 'wb') * length
        result['pesq_nb'] += pesq(16000, clean, denoised, 'nb') * length
        result['stoi'] += stoi(clean, denoised, rate) * length
        result['count'] += 1 * length
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', type=str, help='clean audio path')
    parser.add_argument('-d', '--denoised', type=str, help='denoised audio path')

    args = parser.parse_args()

    clean_path = args.clean
    denoised_path = args.denoised

    result = evaluate(clean_path, denoised_path)

    print("Evaluation results:")
    for key in result:
        if key != 'count':
            print('{} = {:.3f}'.format(key, result[key]/result['count']), end=", ")