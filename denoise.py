import os
import argparse
import json
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
# from torch.utils.tensorboard import SummaryWriter

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread

from dataset import load_CleanNoisyPairDataset
from util import rescale, find_max_epoch, print_size, sampling
from network import CleanUNet, CleanUNet_bilinear


def denoise(output_directory, ckpt_iter, subset, num, gpu, dump=False):
    """
    Denoise audio

    Parameters:
    output_directory (str):         save generated speeches to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    subset (str):                   training, testing, validation
    num (int):                      number of samples to use in inference, use all if 0.
    gpu (bool):                     whether to run on gpu
    dump (bool):                    whether save enhanced (denoised) audio
    """

    # setup local experiment path
    exp_path = train_config["exp_path"]
    print('exp_path:', exp_path)

    # load data
    loader_config = deepcopy(trainset_config)
    loader_config["crop_length_sec"] = 0
    dataloader = load_CleanNoisyPairDataset(
        **loader_config, 
        subset=subset,
        batch_size=1, 
        num_gpus=1
    )
    if num == 0:
        num = len(dataloader)

    # predefine model
    device = 'cuda' if gpu else 'cpu'
    if(gpu):
        assert torch.cuda.is_available()
    print(opt_config)
    if("bilinear" in opt_config.keys() and opt_config["bilinear"] == True):
        net = CleanUNet_bilinear(**network_config).to(device)
    else:
        net = CleanUNet(**network_config).to(device)
    print_size(net)

    # load checkpoint
    ckpt_directory = os.path.join(train_config["log"]["directory"], exp_path, 'checkpoint')
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    if ckpt_iter != 'pretrained':
        ckpt_iter = int(ckpt_iter)
    model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
    print('model_path:', model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # get output directory ready
    if ckpt_iter == "pretrained":
        speech_directory = os.path.join(output_directory, exp_path, 'speech', ckpt_iter) 
    else:
        speech_directory = os.path.join(output_directory, exp_path, 'speech', '{}k'.format(ckpt_iter//1000))
    if dump and not os.path.isdir(speech_directory):
        os.makedirs(speech_directory)
        os.chmod(speech_directory, 0o775)
    print("speech_directory: ", speech_directory, flush=True)

    # inference
    all_generated_audio = []
    all_clean_audio = []
    sortkey = lambda name: '_'.join(name.split('/')[-1].split('_')[1:])

    avg_time = 0

    iter = 1
    with tqdm(total = num) as pbar:
        for clean_audio, noisy_audio, fileid in dataloader:
            clean_audio, noisy_audio = clean_audio.to(device), noisy_audio.to(device)

            filename = fileid[0][0].split('/')[-1]
    
            LENGTH = len(noisy_audio[0].squeeze())
            start_time = time.time()
            generated_audio = sampling(net, noisy_audio)
            end_time = time.time()
            elapsed_time = end_time - start_time            
            if dump:
                wavwrite(os.path.join(speech_directory, filename),
                        trainset_config["sample_rate"],
                        generated_audio[0].squeeze().cpu().numpy())
            else:
                all_clean_audio.append(clean_audio[0].squeeze().cpu().numpy())
                all_generated_audio.append(generated_audio[0].squeeze().cpu().numpy())
                
            avg_time += elapsed_time
            pbar.set_postfix({"Average Time": f"{avg_time / iter:.6f}"})
            pbar.update(1)
  
            
            if iter == num:
                break
            iter+=1

    print("Average time: ", avg_time / iter)
    print("Total time: ", avg_time)
    return all_clean_audio, all_generated_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max" or "pretrained"')     
    parser.add_argument('-subset', '--subset', type=str, choices=['training', 'testing', 'validation'],
                        default='testing', help='subset for denoising')
    parser.add_argument('-n','--num', type=int, default=0, help='number of samples to use in inference')
    parser.add_argument('-cpu', '--cpu', action='store_true', help='Use CPU instead of GPU')
    

    args = parser.parse_args()
    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    gen_config              = config["gen_config"]
    global network_config
    network_config          = config["network_config"]      # to define wavenet
    global train_config
    train_config            = config["train_config"]        # train config
    global trainset_config
    trainset_config         = config["trainset_config"]     # to read trainset configurations
    global opt_config
    opt_config              = config["opt_config"] 

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    gpu = not args.cpu
    
    if args.subset == "testing":
        with torch.no_grad():
            denoise(gen_config["output_directory"],
                    subset=args.subset,
                    ckpt_iter=args.ckpt_iter,
                    num=args.num,
                    gpu=gpu,
                    dump=True)
    
