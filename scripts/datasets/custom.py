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
       ├── rgb
              ├── *.png / *.jpg / *.jpeg / *.bmp

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
    
    def preload_rgbinfo(self):
        '''
        We don't need timestamp in vo setup, we set 1s perframe.
        '''
        rgb_dir = os.path.join(self.dataset_dir, 'rgb')
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        rgb_files = []
        for pattern in patterns:
            rgb_files.extend(glob.glob(os.path.join(rgb_dir, pattern)))
            rgb_files.extend(glob.glob(os.path.join(rgb_dir, pattern.upper())))
        rgb_files = sorted(set(rgb_files))

        if len(rgb_files) == 0:
            raise FileNotFoundError(f"No images found in {rgb_dir}. Supported extensions: {patterns}")

        rgbinfo_dict = {}
        rgbinfo_dict['timestamp'] = list(range(len(rgb_files)))    # (N, ), list
        rgbinfo_dict['filepath']  = rgb_files # (N, ), list
        self.rgbinfo_dict = rgbinfo_dict
    
    
    
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
        self.tqdm.update(1)
        return data_packet
    

def get_dataset(config):
    return CustomDataset(config)