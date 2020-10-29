import torch
import collections
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import librosa

int16_max = (2 ** 15) - 1

## PREPROCESSING PARAMETERS
# Mel-filterbank
window_length = 25  # In milliseconds
window_step = 10    # In milliseconds
n_fft = 512

# Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 300     # 3000 ms

# Audio volume normalization
audio_norm_target_dBFS = -30


ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ToyASVDataset(Dataset):
    """
    Utility class to load a toy version of ASV2019 dataset.
	
	Params:
		LOGICAL_DATA_ROOT: path to data of logical partition of LA.
		LOGICAL_PROTOCOL_DIR: path to the protocol files. Names of the protocol files must be:
			- ASVspoof2019.LA.cm.train_short.trn.txt
			- ASVspoof2019.LA.cm.dev_short.trn.txt
			- ASVspoof2019.LA.cm.eval_short.trn
		feature_name: name to attach to the cached file. Has no effect on the data.
		CACHE_DIR: directory where the cached dataset is saved in/loaded from.
		transform: PyTorch transform to data
		target_transform: PyTorch transform to label
		partition: which partition of the dataset to load. Must be one of the following: 'train', 'dev', 'eval'
		sample_size: size of a random subset to extract from the dataset. If None (default value), all data of the partition is taken.
    """
    
    def __init__(self,
        LOGICAL_DATA_ROOT,
        LOGICAL_PROTOCOL_DIR,
		feature_name,
		partition = 'train',
        CACHE_DIR=None,
        transform=None,
        target_transform=None,
        # is_train=True,
        # is_eval=False,
        sample_size=None):
		
        assert partition in ['train', 'dev', 'eval'], "partition parameter must be either 'train', 'dev' or 'eval'"
		
        data_root = LOGICAL_DATA_ROOT
        track = 'LA'
		
        self.track = track
        self.prefix = 'ASVspoof2019_{}'.format(track)
        v1_suffix = ''

        self.sysid_dict = {
            'human': 0,  # bonafide speech
            'spoof': 1, # Spoofed signal
        }
        
		# Preparing strings of paths
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = partition
        self.protocols_fname = f'{partition}_short.trn'
        self.protocols_dir = LOGICAL_PROTOCOL_DIR
        
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, 'train')+v1_suffix, 'flac') # we just take from the train folder, sorry
        self.protocols_fname = os.path.join(self.protocols_dir,
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        self.cache_fname = 'cache_{}_{}_{}.npy'.format(self.dset_name, track, feature_name)
        
        if CACHE_DIR is not None:
            self.cache_fname = os.path.join(CACHE_DIR, self.cache_fname)
        
        self.transform = transform
		# Load from cache if present
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
			# Apply transforms, if any
            if self.transform:
                self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
            if target_transform:
                self.data_y = Parallel(n_jobs=4, prefer='threads')(delayed(target_transform)(y) for y in self.data_y)
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            print('Dataset saved to cache ', self.cache_fname)
			
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            self.files_meta= [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            self.data_sysid = [self.data_sysid[x] for x in select_idx]
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y #, self.files_meta[idx]

    def read_file(self, meta):
        data_x = self.read_and_preprocess_file(meta.path, sampling_rate=sampling_rate)
        data_y = meta.key
        # MODIFYING THIS TO INT 
        return data_x, int(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[3] == 'human'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)
    
    def adjust_duration(self, feature, partial_n_frames=10000):
        """
        Adapted from AutoSpeech (https://github.com/VITA-Group/AutoSpeech/blob/master/data_objects/DeepSpeakerDataset.py).
		If feature param is shorter than given number of frames, it is repeated.
		If it is longer, a random subset of consecutive frames is extracted to match the desired lenght.
        """
        if feature.shape[0] <= partial_n_frames:
            start = 0
            while feature.shape[0] < partial_n_frames:
                feature = np.repeat(feature, 2, axis=0)
        else:
            start = np.random.randint(0, feature.shape[0] - partial_n_frames)
        end = start + partial_n_frames
        return feature[start:end]
    
    
    def wav_to_spectrogram(self, wav, sampling_rate=16000, window_step=10, window_length=25, n_fft=512):
        """
        Converts audio signal to spectrogram.
		Returns the spectrogram in the shape (freq, time).
        """
        frames = np.abs(librosa.core.stft(
            wav,
            n_fft=n_fft,
            hop_length=int(sampling_rate * window_step / 1000),
            win_length=int(sampling_rate * window_length / 1000),
        ))

        return frames.astype(np.float32)
    
    def normalize_volume(self, wav, target_dBFS=-30, increase_only=False, decrease_only=False):
        """
        Normalizes volume of supplied wav.
		...or at least I think, I kinda have no idea what I'm doing. :D
        Adapted from AutoSpeech (https://github.com/VITA-Group/AutoSpeech/blob/master/data_objects/audio.py).
        """
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))
    
    def read_and_preprocess_file(self, path: str, sampling_rate=16000) -> np.array:
        """
        Read audio file to spectrogram.

        Params:
            path: file location
            sampling_rate: desired sampling rate (will resample the supplied wav if needed)

        Retruns:
            spectrogram of supplied file in shape (freq, time)
        """
        wav, sr = librosa.load(path, sr=sampling_rate)
        wav = self.normalize_volume(wav, target_dBFS=audio_norm_target_dBFS, increase_only=True)
        spectrogram = self.wav_to_spectrogram(wav, window_length=window_length, window_step=window_step, n_fft=n_fft, sampling_rate=sampling_rate) # obtain (freq, time) spectrogram
        # Resample the spectrogram to bound its duration.
        # It must be transposed to (time, freq) first
        spectrogram = self.adjust_duration(spectrogram.T, partial_n_frames=partials_n_frames)
        # Re-transpose to get it back to (freq, time) shape
        return spectrogram.T
		
def asv_toys(ROOT='ToyASV2019', download=True):
  """
  Download pre-processed cache of the three partitions of Toy ASV2019 dataset and load them into instances of ToyASVDataset class.
  
  Params:
	ROOT: where to download pre-processed cache files
	download: if False, do not attempt to download and look for the cache files directly into ROOT
	
  Returns:
	Train, Dev and Eval partitions as a triple
  """
  
  train_link_dropbox = 'https://www.dropbox.com/s/6i1b9gtilsvydml/cache_train_LA_toy.npy?dl=0'
  dev_link_dropbox = 'https://www.dropbox.com/s/mqmv4uoap8jqw1n/cache_dev_LA_toy.npy?dl=0'
  eval_link_dropbox = 'https://www.dropbox.com/s/zgxam6t7ki1i6f3/cache_eval_LA_toy.npy?dl=0'

  if not os.path.isdir(ROOT):
    os.mkdir(ROOT)

  if download:
    links = [train_link_dropbox, dev_link_dropbox, eval_link_dropbox]
    names = ['cache_train_LA_toy.npy', 'cache_dev_LA_toy.npy', 'cache_eval_LA_toy.npy']

    for link, name in zip(links, names):
      full_path = os.path.join(ROOT, name)
      if not os.path.isfile(full_path):
        print(f'Downloading {name} to {full_path}...')
        os.system(f"wget {link} -P {ROOT} -O {full_path}")
    
  asv_train = ToyASVDataset(ROOT, ROOT, feature_name='toy', partition='train', CACHE_DIR=ROOT)
  asv_dev = ToyASVDataset(ROOT, ROOT, feature_name='toy', partition='dev', CACHE_DIR=ROOT)
  asv_eval = ToyASVDataset(ROOT, ROOT, feature_name='toy', partition='eval', CACHE_DIR=ROOT)

  return asv_train, asv_dev, asv_eval