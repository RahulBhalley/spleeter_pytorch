import csv
import random
import torchaudio
import numpy as np
import torch
from torch.utils.data import Dataset
import ffmpeg
from scipy.signal.windows import hann
from librosa.core import stft, istft


def _load_audio(
        path, offset=None, duration=None,
        sample_rate=None, dtype=np.float32):
    """
    Load audio file and return its data as a waveform.

    Args:
        path (str): Path of the audio file to load data from.
        offset (float, optional): Start offset to load from in seconds.
        duration (float, optional): Duration to load in seconds.
        sample_rate (int, optional): Sample rate to load audio with.
        dtype (numpy.dtype, optional): Numpy data type to use, default to float32.

    Returns:
        tuple: Loaded data as (waveform, sample_rate).

    Raises:
        SpleeterError: If any error occurs while loading audio.
    """
    if not isinstance(path, str):
        path = path.decode()
        
    probe = ffmpeg.probe(path)

    metadata = next(
        stream
        for stream in probe['streams']
        if stream['codec_type'] == 'audio')
    n_channels = metadata['channels']
    if sample_rate is None:
        sample_rate = metadata['sample_rate']
    output_kwargs = {'format': 'f32le', 'ar': sample_rate}
    if duration is not None:
        output_kwargs['t'] = _to_ffmpeg_time(duration)
    if offset is not None:
        output_kwargs['ss'] = _to_ffmpeg_time(offset)
    process = (
        ffmpeg
        .input(path)
        .output('pipe:', **output_kwargs)
        .run_async(pipe_stdout=True, pipe_stderr=True))
    buffer, _ = process.communicate()
    waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
    if not waveform.dtype == np.dtype(dtype):
        waveform = waveform.astype(dtype)
    return waveform, sample_rate
 
def _to_ffmpeg_time(n):
    """
    Format number of seconds to time expected by FFMPEG.

    Args:
        n (float): Time in seconds to format.

    Returns:
        str: Formatted time in FFMPEG format.
    """
    m, s = divmod(n, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%09.6f' % (h, m, s)


def _stft(data, inverse=False, frame_length=4096, frame_step=1024, length=None):
    """
    Compute STFT or ISTFT on stereo data.

    Args:
        data (numpy.ndarray): Input data, shape (n_samples, 2) for STFT or (T, F, 2) for ISTFT.
        inverse (bool): If True, compute ISTFT; otherwise, compute STFT.
        frame_length (int): Length of each frame.
        frame_step (int): Step size between frames.
        length (int, optional): Output length for ISTFT.

    Returns:
        numpy.ndarray: STFT or ISTFT result with channels in the last dimension.
    """
    assert not (inverse and length is None)
    data = np.asfortranarray(data)
    N = frame_length
    H = frame_step
    win = hann(N, sym=False)
    fstft = istft if inverse else stft
    win_len_arg = {"win_length": None, "length": length} if inverse else {"n_fft": N}
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = data[:, :, c].T if inverse else data[:, c]
        s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
        s = np.expand_dims(s.T, 2-inverse)
        out.append(s)
    if len(out) == 1:
        return out[0]
    return np.concatenate(out, axis=2-inverse)


class TrainDataset(Dataset):
    def __init__(self, params):
        """
        Initialize the TrainDataset.

        Args:
            params (dict): Parameters for dataset initialization.
        """
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
        """
        Get the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return self.count
    
    def __getitem__(self, chunk_id):
        """
        Get an item from the dataset.

        Args:
            chunk_id (int): Index of the chunk to retrieve.

        Returns:
            tuple: Contains mix_stft_mag, vocal_stft_mag, instru_stft_mag.
        """
        chunk_id %= self.count
        pair = self.datasets[chunk_id]
        mix_chunk, vocal_chunk, instru_chunk = pair[:3]
        samplerate = float(pair[4])
        start_time = float(pair[5])
                
        # Load audio
        mix_audio, mix_sr = _load_audio(mix_chunk, offset=start_time, duration=self.chunk_duration) 
        vocal_audio, vocal_sr = _load_audio(vocal_chunk, offset=start_time, duration=self.chunk_duration)
        instru_audio, instru_sr = _load_audio(instru_chunk, offset=start_time, duration=self.chunk_duration)
        
        mix_audio = mix_audio.T
        vocal_audio = vocal_audio.T
        instru_audio = instru_audio.T

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
        mix_stft = _stft(mix_audio.T, frame_length=self.frame_length, frame_step=self.frame_step)
        mix_stft_mag = abs(mix_stft)
        mix_stft_mag = mix_stft_mag.transpose(2, 1, 0)
        
        vocal_stft = _stft(vocal_audio.T, frame_length=self.frame_length, frame_step=self.frame_step)
        vocal_stft_mag = abs(vocal_stft)
        vocal_stft_mag = vocal_stft_mag.transpose(2, 1, 0)

        instru_stft = _stft(instru_audio.T, frame_length=self.frame_length, frame_step=self.frame_step)
        instru_stft_mag = abs(instru_stft)
        instru_stft_mag = instru_stft_mag.transpose(2, 1, 0)

        # Random crop in time
        num_frame = mix_stft_mag.shape[2]
        start = random.randint(low=1, high=(num_frame - self.T))
        end = start + self.T
        mix_stft_mag = mix_stft_mag[:, :self.F, start: end]
        vocal_stft_mag = vocal_stft_mag[:, :self.F, start: end]
        instru_stft_mag = instru_stft_mag[:, :self.F, start: end]
                
        return mix_stft_mag, vocal_stft_mag, instru_stft_mag
