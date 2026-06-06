import numpy as np
import time
import os
import glob
import bisect
import torch
import cv2
from datetime import datetime
from tqdm import tqdm

'''
datadir|
       ├── *.png / *.jpg / *.jpeg / *.bmp
       └── rgb
              └── *.png / *.jpg / *.jpeg / *.bmp

'''

class CustomDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.h_resize, self.w_resize = int(cfg['frontend']['image_size'][0]), int(cfg['frontend']['image_size'][1])
        # Load data/datainfo.
        self.dataset_dir = cfg['dataset']['root']
        self.preload_rgbinfo()
        # Load extrinsic/intrinsic parameters.
        self.c2i       = np.eye(4)
        self.intrinsic = None
        self.tqdm = tqdm(total=self.__len__())
    
    def __len__(self):
        return len(self.rgbinfo_dict['timestamp'])
    
    def convert_to_unix_timestamp(self, line_str):
        # Input: 2011-09-26 13:02:25.446243840
        date_str, raw_time_str = line_str.split(' ')
        time_str, mm_second = raw_time_str.split('.')
        mm_second = '.' + mm_second
        datetime_str = f"{date_str} {time_str}"
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(dt.timetuple()) + dt.microsecond / 1e6
        timestamp += float(mm_second)
        return timestamp  

    def preload_camtimestamp(self):
        return np.array(self.rgbinfo_dict['timestamp']).reshape(-1, 1)
    
    def preload_imu(self):
        all_imu = np.zeros((len(self.rgbinfo_dict['timestamp']), 7))
        all_imu[:, 0] = np.array(self.rgbinfo_dict['timestamp'])
        return all_imu
    
    def _timestamp_from_filename(self, filepath, fallback):
        name = os.path.splitext(os.path.basename(filepath))[0]
        if '_' in name:
            maybe_timestamp = name.rsplit('_', 1)[-1]
            try:
                return float(maybe_timestamp)
            except ValueError:
                pass
        return float(fallback)

    def _frame_id_from_filename(self, filepath):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        return stem.split('_', 1)[0]

    def _to_numpy_timestamps(self, timestamps):
        if torch.is_tensor(timestamps):
            return timestamps.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return np.asarray(timestamps, dtype=np.float64).reshape(-1)

    def get_frame_metadata_by_timestamps(self, timestamps):
        query_timestamps = self._to_numpy_timestamps(timestamps)
        source_timestamps = self._timestamp_array
        matched_indices = []
        for timestamp in query_timestamps:
            matched_indices.append(int(np.argmin(np.abs(source_timestamps - timestamp))))

        return {
            'frame_indices': matched_indices,
            'frame_ids': [self.rgbinfo_dict['frame_id'][idx] for idx in matched_indices],
            'image_names': [self.rgbinfo_dict['image_name'][idx] for idx in matched_indices],
            'image_stems': [self.rgbinfo_dict['image_stem'][idx] for idx in matched_indices],
            'image_paths': [self.rgbinfo_dict['filepath'][idx] for idx in matched_indices],
            'render_names': [f"{self.rgbinfo_dict['frame_id'][idx]}.png" for idx in matched_indices],
            'pose_names': [f"{self.rgbinfo_dict['frame_id'][idx]}.txt" for idx in matched_indices],
            'matched_timestamps': [self.rgbinfo_dict['timestamp'][idx] for idx in matched_indices],
        }

    def preload_rgbinfo(self):
        '''
        Prefer timestamps encoded in filenames like 000_1638358313.099999905.png.
        Fall back to 1s-per-frame indices for generic custom image folders.
        '''
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_dirs = [self.dataset_dir, os.path.join(self.dataset_dir, 'rgb')]
        rgb_files = []
        for image_dir in image_dirs:
            for pattern in patterns:
                rgb_files.extend(glob.glob(os.path.join(image_dir, pattern)))
                rgb_files.extend(glob.glob(os.path.join(image_dir, pattern.upper())))
        rgb_files = sorted(set(rgb_files))

        if len(rgb_files) == 0:
            searched = ', '.join(image_dirs)
            raise FileNotFoundError(f"No images found in {searched}. Supported extensions: {patterns}")

        rgbinfo_dict = {}
        rgbinfo_dict['timestamp'] = [self._timestamp_from_filename(path, idx) for idx, path in enumerate(rgb_files)]
        rgbinfo_dict['filepath']  = rgb_files # (N, ), list
        rgbinfo_dict['image_name'] = [os.path.basename(path) for path in rgb_files]
        rgbinfo_dict['image_stem'] = [os.path.splitext(os.path.basename(path))[0] for path in rgb_files]
        rgbinfo_dict['frame_id'] = [self._frame_id_from_filename(path) for path in rgb_files]
        self.rgbinfo_dict = rgbinfo_dict
        self._timestamp_array = np.asarray(rgbinfo_dict['timestamp'], dtype=np.float64)
    
    
    
    def __getitem__(self, idx):
        resized_h, resized_w = int(self.cfg['frontend']['image_size'][0]), int(self.cfg['frontend']['image_size'][1])
        rgb_raw = cv2.imread(self.rgbinfo_dict['filepath'][idx])
        if rgb_raw is None:
            raise FileNotFoundError(f"Failed to read image: {self.rgbinfo_dict['filepath'][idx]}")
        rgb = (torch.tensor(cv2.resize(rgb_raw, (resized_w, resized_h)))[...,[2,1,0]]).permute(2,0,1).unsqueeze(0).to(self.cfg['device']['tracker'])
        u_scale, v_scale = resized_h/self.cfg['intrinsic']['H'], resized_w/self.cfg['intrinsic']['W']
        intrinsic = torch.tensor([self.cfg['intrinsic']['fv']*v_scale, self.cfg['intrinsic']['fu']*u_scale, \
                                  self.cfg['intrinsic']['cv']*v_scale, self.cfg['intrinsic']['cu']*u_scale], dtype=torch.float32, device=self.cfg['device']['tracker'])
        data_packet = {}
        data_packet['timestamp'] = self.rgbinfo_dict['timestamp'][idx] # float 
        data_packet['rgb']       = rgb                                 # (1, 3, H, W)
        data_packet['intrinsic'] = intrinsic                           # (4, )
        data_packet['frame_index'] = idx
        data_packet['frame_id'] = self.rgbinfo_dict['frame_id'][idx]
        data_packet['image_name'] = self.rgbinfo_dict['image_name'][idx]
        data_packet['image_path'] = self.rgbinfo_dict['filepath'][idx]
        self.tqdm.update(1)
        return data_packet
    

def get_dataset(config):
    return CustomDataset(config)