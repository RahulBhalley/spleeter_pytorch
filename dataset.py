import csv
import math
import random
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as Func
from numpy import random


def load_audio(filename, start=0, stop=None, resample=True):
    """
    Load wav file.

    Args:
        filename (str): Path to audio file
        start (int): Start frame (default 0)
        stop (int): End frame (default None)
        resample (bool): Whether to resample (default True)

    Returns:
        tuple: Tuple containing:
            - torch.Tensor: Audio waveform (L x 2)
            - int: Sample rate
    """
    wav, sr = torchaudio.load_wav(filename)
    
    wav_torch = wav / (wav.max() + 1e-8)
    if start is not None:
        wav_torch = wav_torch[:, start:stop]
    
    return wav_torch, sr


def stft_feature(waveform, sample_rate=44100, frame_length=2048, frame_step=512,
                 spec_exponent=1., F=1024, T=512, separate=False):
    """
    Compute STFT feature from waveform.

    Args:
        waveform (torch.Tensor): Audio waveform (L x 2)
        sample_rate (int): Sample rate (default 44100)
        frame_length (int): STFT frame length (default 2048)
        frame_step (int): STFT frame step (default 512)
        spec_exponent (float): Spectrogram exponent (default 1.)
        F (int): Number of frequency bins (default 1024)
        T (int): Number of time frames (default 512)
        separate (bool): Whether to return separate components (default False)

    Returns:
        tuple: Tuple containing:
            - torch.Tensor: STFT (F x T x 2)
            - torch.Tensor: Magnitude spectrogram (F x T)
    """
    stft = torch.stft(
        waveform, frame_length, hop_length=frame_step, window=torch.hann_window(frame_length))

    # Only keep frequencies up to F
    stft = stft[:, :F, :, :]
    real = stft[:, :, :, 0]
    im = stft[:, :, :, 1]
    mag = torch.sqrt(real ** 2 + im ** 2)

    return stft, mag


def pad_and_partition(tensor, T):
    """
    Pad and partition tensor into segments of length T.

    Args:
        tensor (torch.Tensor): Input tensor (B x C x F x L)
        T (int): Segment length

    Returns:
        torch.Tensor: Padded and partitioned tensor (B*[L/T] x C x F x T)
    """
    old_size = tensor.size(3)
    new_size = math.ceil(old_size / T) * T
    tensor = Func.pad(tensor, [0, new_size - old_size])
    [b, c, t, f] = tensor.shape
    split = new_size // T
    
    return torch.cat(torch.split(tensor, T, dim=3), dim=0)


class TrainDataset(Dataset):
    """Dataset for training."""

    def __init__(self, params):
        self.datasets = []
        self.count = 0
        self.MARGIN = params['margin']
        self.chunk_duration = params['chunk_duration']
        self.n_chunks_per_song = params['n_chunks_per_song']
        self.frame_length = params['frame_length']
        self.frame_step = params['frame_step']
        self.T = params['T']
        self.F = params['F']

        with open(params['train_manifest'], 'r') as f:
            reader = csv.reader(f)
            for mix_path, vocal_path, instrumental_path, duration, samplerate in reader:
                duration = float(duration)
                for k in range(self.n_chunks_per_song):
                    if self.n_chunks_per_song > 1:
                        start_time = k * (duration - self.chunk_duration - 2 * self.MARGIN) / (self.n_chunks_per_song - 1) + self.MARGIN
                        if start_time > 0.0:
                            self.datasets.append((mix_path, vocal_path, instrumental_path, duration, samplerate, start_time))
                            self.count += 1
                    elif self.n_chunks_per_song == 1:
                        start_time = duration / 2 - self.chunk_duration / 2
                        if start_time > 0.0:
                            self.datasets.append((mix_path, vocal_path, instrumental_path, duration, samplerate, start_time))
                            self.count += 1
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, chunk_id):
        chunk_id %= self.count
        pair = self.datasets[chunk_id]
        mix_chunk, vocal_chunk, instru_chunk, _, samplerate, start_time = pair
        
        samplerate = float(samplerate)
        start_time = float(start_time)
                
        # Load audio chunks
        mix_audio, mix_sr = load_audio(mix_chunk, start=int(start_time * samplerate), stop=int((start_time + self.chunk_duration) * samplerate))
        vocal_audio, vocal_sr = load_audio(vocal_chunk, start=int(start_time * samplerate), stop=int((start_time + self.chunk_duration) * samplerate))
        instru_audio, instru_sr = load_audio(instru_chunk, start=int(start_time * samplerate), stop=int((start_time + self.chunk_duration) * samplerate))
       
        # Resample if necessary
        if int(samplerate) != 44100:
            resample = torchaudio.transforms.Resample(int(samplerate), 44100)
            mix_audio = resample(mix_audio)
            vocal_audio = resample(vocal_audio)
            instru_audio = resample(instru_audio)
            samplerate = 44100
        
        # Ensure 2 channels
        if mix_audio.shape[0] == 1: 
            mix_audio = torch.cat((mix_audio, mix_audio), dim=0)
            vocal_audio = torch.cat((vocal_audio, vocal_audio), dim=0)
            instru_audio = torch.cat((instru_audio, instru_audio), dim=0)
        elif mix_audio.shape[0] > 2:
            mix_audio = mix_audio[:2, :]
            vocal_audio = vocal_audio[:2, :]
            instru_audio = instru_audio[:2, :]

        # Compute STFT
        mix_stft, mix_stft_mag = stft_feature(mix_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        vocal_stft, vocal_stft_mag = stft_feature(vocal_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        instru_stft, instru_stft_mag = stft_feature(instru_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        
        # Random time crop
        num_frame = mix_stft_mag.shape[2]
        start = random.randint(low=1, high=(num_frame - self.T))
        end = start + self.T
        mix_stft_mag = mix_stft_mag[:, :, start:end]
        vocal_stft_mag = vocal_stft_mag[:, :, start:end]
        instru_stft_mag = instru_stft_mag[:, :, start:end]

        return mix_stft_mag, vocal_stft_mag, instru_stft_mag


class SeparateDataset(Dataset):
    """Dataset for separation."""

    def __init__(self, params):
        self.datasets = []
        self.count = 0
        self.chunk_duration = params['chunk_duration']
        self.n_chunks_per_song = params['n_chunks_per_song']
        self.frame_length = params['frame_length']
        self.frame_step = params['frame_step']
        self.T = params['T']
        self.F = params['F']

        with open(params['separate_manifest'], 'r') as f:
            reader = csv.reader(f)
            for path, duration, samplerate in reader:
                self.datasets.append((path, duration, samplerate))
                self.count += 1
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, audio_id):
        audio_id %= self.count
        audio = self.datasets[audio_id]

        path, duration, samplerate = audio
        duration = float(duration)
        samplerate = float(samplerate)
        wav_name = path.split('/')[-1].split('.')[0]

        source_audio, _ = load_audio(path)
        stft, stft_mag = stft_feature(source_audio, sample_rate=samplerate, frame_length=self.frame_length, frame_step=self.frame_step, spec_exponent=1., F=self.F, T=self.T)
        stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
            
        L = stft.size(2)

        stft_mag = pad_and_partition(stft_mag, self.T)

        return stft, stft_mag.transpose(2, 3), L, wav_name, samplerate
