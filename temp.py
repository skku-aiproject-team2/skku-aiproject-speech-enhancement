

# -*- coding: utf-8 -*-
"""compare_metric.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1soNoGu-90KSnKZnMoEkBecuJqDNcnUnx

### 테스트 데이터셋 설정

**필요한 경우만 실행할 것**

dns/datasets/...에 있는 테스트 데이터를 eval_data로 복사한 후,

noisy의 데이터들의 이름을 noisy_fileid_12.wav와 같이 변경합니다.
"""

import os
from collections import defaultdict
from tqdm import tqdm
import time
import warnings
import json
import traceback

warnings.filterwarnings("ignore")

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from pesq import pesq
from pystoi import stoi

from pathlib import Path
import wave

def prettier(obj):
    print(json.dumps(obj, indent=4, sort_keys=True))
    
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def result_to_metric(result):
    metric = defaultdict(float)

    prettier(result)
    metric["pesq_wb"] = result["pesq_wb"] / result["count"]
    metric["pesq_nb"] = result["pesq_nb"] / result["count"]
    metric["stoi"] = result["stoi"] / result["count"]
    metric["rtf"] = result["infer_time"] / result["length"]

    return metric


def save_enhanced_audio(rate, enhanced_audio, output_dir, filename):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert floating point audio to int16
    if enhanced_audio.dtype != np.int16:
        enhanced_audio = np.int16(enhanced_audio / np.max(np.abs(enhanced_audio)) * 32767)
    
    # Save the enhanced audio using wave module
    output_path = os.path.join(output_dir, filename)
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(1)  # Assuming mono audio
    wf.setsampwidth(2)  # Assuming 16-bit audio
    wf.setframerate(rate)
    wf.writeframes(enhanced_audio.tostring())
    wf.close()

def eval_metric(infer, target_name, testset_path="eval", output_base_dir="enhanced_output"):
    result = defaultdict(int)

    cleans = os.listdir(os.path.join(testset_path, "clean"))
    noises = os.listdir(os.path.join(testset_path, "noisy"))

    output_dir = os.path.join(output_base_dir, target_name)
    
    for i in tqdm(range(len(cleans))):
        duration = 0
        try:
            rate, clean = wavfile.read(
                os.path.join(testset_path, "clean", cleans[i])
            )
            rate, noisy = wavfile.read(
                os.path.join(testset_path, "noisy", noises[i])
            )
            # As we infer on the CPU device, we don't need to sync with GPU.
            # So, we can utilize time.
            start_time = time.time()
            rate, target_wav = infer(rate, noisy, i)
            duration = time.time() - start_time

            # Save the enhanced audio
            save_enhanced_audio(rate, target_wav, output_dir, noises[i])
        except Exception as e: 
            traceback.print_exc()
            return
            continue

        n_samples = target_wav.shape[-1]
        length = n_samples / rate

        result["pesq_wb"] += (
            pesq(16000, clean, target_wav, "wb") * n_samples
        )  # wide band
        result["pesq_nb"] += (
            pesq(16000, clean, target_wav, "nb") * n_samples
        )  # narrow band
        result["stoi"] += stoi(clean, target_wav, rate) * n_samples
        result["count"] += 1 * n_samples
        result["length"] += length
        result["infer_time"] += duration

    if result["count"] is None:
        return None
    metric = result_to_metric(result)
    return metric

"""infer에 들어갈 spectral_subtraction 함수입니다.

메트릭을 구하기 위해 길이가 같아야 해서 코드를 적절히 변경하였습니다.

해당 코드를 비교해보시면 변경된 부분을 쉽게 구하실 수 있을 거에요.
"""

def spectral_subtraction(rate: int, noisy: NDArray, i: int):
    fft = abs(np.fft.fft(noisy))
    len_ = 20 * rate // 1000  # frame size in samples
    PERC = 50  # window overlap in percent of frame
    len1 = len_ * PERC // 100  # overlap'length
    len2 = len_ - len1  # window'length - overlap'length

    # setting default parameters
    Thres = 3  # VAD threshold in dB SNRseg
    Expnt = 1.0  # exp(Expnt)
    G = 0.9

    # initial Hamming window
    win = np.hamming(len_)
    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)

    # nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))
    nFFT = 2 * next_power_of_2(len_)
    noise_mean = np.zeros(nFFT)
    j = 1
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * noisy[j : j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    # initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(noisy) // len2 - 1
    xfinal = np.zeros(noisy.shape[0])

    # === Start Processing === #
    for n in range(0, Nframes):
        # Windowing
        insign = win * noisy[k - 1 : k + len_ - 1]
        # compute fourier transform of a frame
        spec = np.fft.fft(insign, nFFT)
        # compute the magnitude
        sig = abs(spec)
        # save the noisy phase information
        theta = np.angle(spec)
        # SNR
        SNRseg = 10 * np.log10(
            np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2
        )

        # --- spectral subtraction --- #
        sub_speech = sig**Expnt - noise_mu**Expnt
        # the pure signal is less than the noise signal power
        diffw = sig**Expnt - noise_mu**Expnt

        # beta negative components
        def find_index(x_list):
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list

        z = find_index(diffw)
        if len(z) > 0:
            sub_speech[z] = 0

        # --- implement a simple VAD detector --- #
        if SNRseg < Thres:  # Update noise spectrum
            noise_temp = (
                G * noise_mu**Expnt + (1 - G) * sig**Expnt
            )  # Smoothing processing noise power spectrum
            noise_mu = noise_temp ** (1 / Expnt)  # New noise amplitude spectrum

        # add phase
        x_phase = (sub_speech ** (1 / Expnt)) * np.exp(img * theta)
        # take the IFFT
        xi = np.fft.ifft(x_phase).real

        # --- Overlap and add --- #
        xfinal[k - 1 : k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1 : len_]

        k = k + len2

    xfinal[k - 1 :k + len2 - 1] = x_old

    return rate, winGain * xfinal.astype(noisy.dtype)

"""mmse"""
import scipy.special as sp

def mmse(rate: int, noisy: NDArray, i: int):
    len_ = 20 * rate // 1000  # frame size in samples
    PERC = 50  # window overlap in percent of frame
    len1 = len_ * PERC // 100  # overlap'length
    len2 = len_ - len1  # window'length - overlop'length

    # setting default parameters
    aa = 0.98
    eta = 0.15
    Thres = 3
    mu = 0.98
    c = np.sqrt(np.pi) / 2
    ksi_min = 10 ** (-25 / 10)

    # hamming window
    win = np.hamming(len_)
    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)

    # setting initial noise
    nFFT = 2 * next_power_of_2(len_)
    j = 1
    noise_mean = np.zeros(nFFT)
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * noisy[j : j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5
    noise_mu2 = noise_mu ** 2

    # initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len2)
    Nframes = len(noisy) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # === Start Processing ==== #
    for n in range(0, Nframes):

        # Windowing
        insign = win * noisy[k - 1 : k + len_ - 1]

        # Take fourier transform of frame
        spec = np.fft.fft(insign , nFFT)
        sig = abs(spec)
        sig2 = sig ** 2
        # save the noisy phase information
        theta = np.angle(spec)  

        SNRpos = 10 * np.log10(
            np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2
        )

        # posteriori SNR
        gammak = np.minimum(sig2 / noise_mu2 , 40) 
        
        # decision-direct estimate of a priori SNR  P231 [7.75]
        if n == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1 , 0)
        else:
            ksi = aa * Xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1 , 0)
            # limit ksi to -25 dB 
            ksi = np.maximum(ksi_min , ksi)  

        # --- implement a simple VAD detector --- #
        if SNRpos < Thres:  # Update noise spectrum
            noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2  # Smoothing processing noise power spectrum
            noise_mu = np.sqrt(noise_mu2)

        # [7.40]
        vk = gammak * ksi / (1 + ksi)
        j_0 = sp.iv(0 , vk/2) #modified bessel function of the first kind of real order 
        j_1 = sp.iv(1 , vk/2)    
        C = np.exp(-0.5 * vk)
        A = ((c * (vk ** 0.5)) * C) / gammak      # [7.40] A
        B = (1 + vk) * j_0 + vk * j_1             # [7.40] B
        hw = A * B                                # [7.40]

        # get X(w)
        mmse_speech = hw * sig

        # save for estimation of a priori SNR in next frame
        Xk_prev = mmse_speech ** 2  

        # IFFT
        x_phase = mmse_speech * np.exp(img * theta)
        xi_w = np.fft.ifft(x_phase , nFFT).real

        # overlap add
        xfinal[k - 1 : k + len2 - 1] = x_old + xi_w[0 : len1]
        x_old = xi_w[len1 + 0 : len_]

        k = k + len2

    xfinal = winGain * xfinal.astype(noisy.dtype)
    
    # Overlap으로 인해 크기 차이 발생
    if len(xfinal) < len(noisy):
        xfinal = np.pad(xfinal, (0, len(noisy) - len(xfinal)), 'constant')
    else:
        xfinal = xfinal[:len(noisy)]

    return rate, xfinal

"""wiener infernence"""
def wiener_filtering(rate: int, noisy: NDArray, i: int):
    len_ = 20 * rate // 1000  # frame size in samples
    PERC = 50  # window overlap in percent of frame
    len1 = len_ * PERC // 100  # overlap'length
    len2 = len_ - len1  # window'length - overlop'length

    # setting default parameters
    Thres = 3       # VAD threshold in dB SNRseg
    Expnt = 1.0
    G = 0.9

    # sine window
    i = np.linspace(0, len_ - 1, len_)
    win = np.sqrt(2 / (len_ + 1)) * np.sin(np.pi * (i + 1) / (len_ + 1))

    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)

    # setting initial noise
    nFFT = 2 * next_power_of_2(len_)
    j = 1
    noise_mean = np.zeros(nFFT)
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * noisy[j:j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5

    # initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(noisy) // len2 - 1
    xfinal = np.zeros(Nframes * len2)

    # === Start Processing ==== #
    for n in range(0, Nframes):

        # Windowing
        insign = win * noisy[k-1:k + len_ - 1]    
        # compute fourier transform of a frame
        spec = np.fft.fft(insign, nFFT)    
        # compute the magnitude
        sig = abs(spec)     
        # save the noisy phase information
        theta = np.angle(spec)  
        # Posterior SNR
        SNRpos = 10 * np.log10(
            np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2
        )

        # --- wiener filtering --- #
        sub_speech = sig ** Expnt - noise_mu ** Expnt
        diffw = sig ** Expnt - noise_mu ** Expnt   
        
        def find_index(x_list):
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list

        z = find_index(diffw)
        if len(z) > 0:
            sub_speech[z] = 0

        SNRpri = 10 * np.log10(np.linalg.norm(sub_speech, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
        mel_max = 10
        mel_0 = (1 + 4 * mel_max) / 5
        s = 25 / (mel_max - 1)

        def get_mel(SNR):
            if -5.0 <= SNR <= 20.0:
                a = mel_0 - SNR / s
            else:
                if SNR < -5.0:
                    a = mel_max
                if SNR > 20:
                    a = 1
            return a

        mel = get_mel(SNRpri) 
        G_k = sub_speech ** 2 / (sub_speech ** 2 + mel * noise_mu ** 2)
        wf_speech = G_k * sig
        
        if SNRpos < Thres:
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  
            noise_mu = noise_temp ** (1 / Expnt)  

        x_phase = wf_speech * np.exp(img * theta)
        xi = np.fft.ifft(x_phase).real

        xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1:len_]

        k = k + len2

    xfinal = winGain * xfinal.astype(noisy.dtype)
    
    # Overlap으로 인해 크기 차이 발생
    if len(xfinal) < len(noisy):
        xfinal = np.pad(xfinal, (0, len(noisy) - len(xfinal)), 'constant')
    else:
        xfinal = xfinal[:len(noisy)]

    return rate, xfinal

"""메트릭을 측정합니다."""

testset_path = "./dataset/"


def noisy(_rate, noisy, i):
    return _rate, noisy


targets = [
    {"name": "noisy", "infer": noisy},
    {"name": "spectral_subtraction", "infer": spectral_subtraction},
    {"name": "mmse", "infer": mmse},
    {"name": "wiener_filtering", "infer": wiener_filtering}
]

metrics = {}
for target in targets:
    metric = eval_metric(target["infer"], target["name"], testset_path)
    metrics[target["name"]] = metric
prettier(metrics)
