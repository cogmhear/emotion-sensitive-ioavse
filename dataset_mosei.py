import json
import logging
import os
import glob
import random
from os.path import join, isfile

import imageio
import librosa
import numpy as np
import torch
import cv2
#torch.multiprocessing.set_start_method('spawn')# good solution !!!!

#import torch.multiprocessing as mp
#mp.set_start_method('spawn')

from pathlib import Path
from collections import Counter
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from decord import VideoReader, cpu, gpu
from pytorch_lightning import LightningDataModule
from skimage.transform import resize
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data import *
from config_mosei import *
from utils.generic import subsample_list
from face_detector import *

from label_space_mapping import index2emotion, emotion2index
from emo_models import test_transforms
import platform
from PIL import Image
import pandas as pd
from consts import *
import pdb

def get_images(mp4_file):
    data = [np.array(img)[np.newaxis, ...] for img in imageio.mimread(mp4_file)]
    return np.concatenate(data, axis=0)


class MOSEIDataset(Dataset):
    def __init__(self, scenes_root, fea_type="mag", emotion=False, shuffle=True, seed=SEED, subsample=1, mask_type="IRM",
                 add_channel_dim=True, a_only=True, return_stft=False,
                 clipped_batch=True, sample_items=True, full_face=True, dat="train"):
        
        #self.lips_root = lips_root
        self.full_face = full_face
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.fea_type = fea_type
        self.return_stft = return_stft
        self.a_only = a_only
        self.add_channel_dim = add_channel_dim
        self.dat = dat
        self.visual_root = visual_root

        files_list = self.build_files_list
        files_list = files_list[:40]

        if platform.system() == 'Windows':
            self.files_list = files_list  ## select only a few files
        else:
            self.files_list = files_list
        
        #self.mask_type = mask_type.lower()
        self.rgb = True if nb_channels == 3 else False
        #if shuffle:
        #   random.seed(SEED)
        #   random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False
        self.sample_items = sample_items
        print('feature type --> ', self.fea_type)

        ## for emotion
        self.emotion = emotion
        self.emotion_to_idx = emotion2index
        self.idx_to_emotion, self.classes = np.array(index2emotion),  index2emotion
        self.faces_per_segment = 64
        
    @property
    def build_files_list(self):
        files_list = []
        if self.full_face:
            #print(self.scenes_root)
            noisy_files = glob.glob(os.path.join(self.scenes_root, '*_noisy_snr-*.wav'))

            #for file in os.listdir(self.scenes_root):
            for file in noisy_files:
                noisy_filename = file.split("/")[-1] 
                fileparts = noisy_filename.split('_noisy_snr-')  ## extract '210618_6_1' from filename '210618_6_1_noisy_snr-n6.wav' 
                clean_filename = fileparts[0]+"_target.wav"
                video_filename = fileparts[0]+"_silent.mp4"

                files_list.append((file,
                                   join(self.scenes_root, clean_filename),
                                   join(self.visual_root, video_filename), 
                                   ))
                #print(files_list)
            
            return files_list
        else:
            for file in os.listdir(self.lips_root):
                if file.endswith("silent.mp4"):
                    files_list.append((join(self.scenes_root, file.replace("silent.mp4", "target.wav")),
                                       join(self.scenes_root, file.replace("silent.mp4", "interferer.wav")),
                                       join(self.scenes_root, file.replace("silent.mp4", "mixed.wav")),
                                       join(self.lips_root, file),
                                       ))
            return files_list
        
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        data = {}
        noisy_file, clean_file, mp4_file = self.files_list[idx]
    
        if self.fea_type=='mag':
            if self.a_only:
                if self.return_stft:
                    data["noisy_audio_spec"], data["clean_audio_spec"], data["noisy_stft"], data["clean"] = self.get_data(clean_file, noisy_file, mp4_file)
                else:
                    data["noisy_audio_spec"], data["clean_audio_spec"] = self.get_data(clean_file, noisy_file, mp4_file)
            else:
                if self.return_stft:
                    if self.emotion:
                        data["noisy_audio_spec"], data["clean_audio_spec"], data["noisy_stft"], data["clean"], data["lip_images"], data["emo_images"] = self.get_data(clean_file, noisy_file, mp4_file)
                    else:
                        data["noisy_audio_spec"], data["clean_audio_spec"], data["noisy_stft"], data["clean"], data["lip_images"] = self.get_data(clean_file, noisy_file, mp4_file)
                else:
                    if self.emotion:
                        data["noisy_audio_spec"], data["clean_audio_spec"], data["lip_images"], data["emo_images"] = self.get_data(clean_file, noisy_file, mp4_file)
                    else:
                        data["noisy_audio_spec"], data["clean_audio_spec"], data["lip_images"] = self.get_data(clean_file, noisy_file, mp4_file)

        elif self.fea_type=='lps' or self.fea_type=='pcs':
            data["noisy_audio_spec"], data["clean_audio_spec"], data["noisy_stft"], data["clean"], data["lip_images"], data["emo_images"] = self.get_data(clean_file, noisy_file, mp4_file)
            
            data["scene"] = clean_file.replace(self.scenes_root,"").replace("_target.wav","").replace("/","")
            data["ny_file"] = noisy_file
            data["cl_file"] = clean_file
        return data


    def wav2spectrum(self, y, Normalize):
        epsilon = float(np.finfo(float).eps)
        D = librosa.stft(y, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)        
        D = D + epsilon
        Sxx = np.log10(abs(D)**2)
        phase = np.exp(1j * np.angle(D))
        mean = np.mean(Sxx, axis=1).reshape(-1,1)
        std = np.std(Sxx, axis=1, dtype = np.float32, ddof=1).reshape(-1,1)

        if Normalize:
            Sxx = np.float32((Sxx - mean)/std)

        return Sxx[np.newaxis, ...]


    def get_noisy_features(self, noisy):
        audio_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                  window=self.window, center=True)
        if self.add_channel_dim:
            return np.abs(audio_stft).astype(np.float32)[np.newaxis, ...]
        else:
            return np.abs(audio_stft).astype(np.float32)

    def load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)


    def get_lip_images(self, image, rgb=True):
        #lip_image = np.zeros((64, img_rows, img_cols)).astype(np.float32)
        try:
            #img = get_lip_images(images_root, video_idx, rgb=rgb)
            img = image

            if img is not None:
                img = img.astype(np.float32)
                img = img / 255
                mean = [0.5]
                std = [0.5]
                img = (img - mean) / std
                if lip_image.shape[0] <= img.shape[0]:
                    lip_image = img[:lip_image.shape[0]]
                else:
                    lip_image[:img.shape[0]] = img
        except Exception as e:
            print(e)
        return lip_image[np.newaxis, ...]


    def get_data(self, clean_file, noisy_file, mp4_file):
        noisy = self.load_wav(noisy_file)
        if isfile(clean_file):
            clean = self.load_wav(clean_file)
        else:
            clean = np.zeros(noisy.shape)
        
        clean_filename = clean_file.split("/")[-1]
        noisy_filename = noisy_file.split("/")[-1]
        mp4_filename = mp4_file.split("/")[-1]

        if self.clipped_batch:
            if clean.shape[0] > 48000:
                diff =  clean.shape[0] - 48000
                
                if diff > 6000:
                    clip_idx = random.randint(0, 6000) 
                else:
                    clip_idx = random.randint(0, clean.shape[0] - 48000)
                
                video_idx = max(int((clip_idx / 16000) * 30) - 2, 0)  ##
                clean_clip = clean[clip_idx:clip_idx + 40900]
                noisy_clip = noisy[clip_idx:clip_idx + 40900]

                clean = clean_clip
                noisy = noisy_clip            
            else:
                #print(clean_file)
                video_idx = -1
                clean = np.pad(clean, pad_width=[0, 48000 - clean.shape[0]], mode="constant")
                noisy = np.pad(noisy, pad_width=[0, 48000 - noisy.shape[0]], mode="constant")
                
                clean = clean[:40900]
                noisy = noisy[:40900]        
        
        ### a video with 25fps
        if len(noisy)==40900:
            num_frames = 64
        elif len(noisy)==48000:
            num_frames = 74
        else:
            print('the utterance is not clipped --> clip utterance')

        if not self.a_only:
            if not self.emotion:
                vframes_tensors_clipped=None 
            else:
                ### if want to save bboxes and frames to dir
                bbox_save_dir = '../datasets/mosei/Raw/Videos/Full/bbox_extracted_mp4_segments_30fps/'
                face_images_dir = '../datasets/mosei/Raw/Videos/Full/face_images_extracted_mp4_segments_30fps/'

                emo_feat_path = Path(os.path.join(facial_feat_dir, mp4_filename+'.npz'))
                emo_file = np.load(emo_feat_path)
                emo_feat = emo_file['arr_0']
                
                vframes_tensors_clipped = torch.from_numpy(emo_feat)
                
        if not self.a_only:
            video_capture = cv2.VideoCapture(mp4_file)
            if not video_capture.isOpened():
                print(f'Failed to open MP4 file: {mp4_file}')
                exit()

            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < num_frames:
                frames = [np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for _ in range(total_frames)]  # Pad with zeros

                frame_difference = num_frames - total_frames
                frames.extend([np.zeros((frame_height, frame_width, 3), dtype=np.uint8) for _ in range(frame_difference)])
            elif total_frames > num_frames:
                start_frame = video_idx
                end_frame = start_frame + num_frames
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                frames = []
                for i in range(start_frame, end_frame):
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    resized_frame = cv2.resize(frame, (224, 224))
                    frames.append(resized_frame)
                    #frames.append(frame)
            else:
                video_idx = 0
                # Read all frames from the video
                frames = []
                for _ in range(total_frames):
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, (224, 224))
                    frames.append(resized_frame)
                    #frames.append(frame)
                    
            video_capture.release()

            frames = np.stack(frames)  ## (64, 360, 540, 3)
            #resized_frames = resize(frames, (frames.shape[0], 224, 224, frames.shape[3]), anti_aliasing=True)  ## (64, 224, 224, 3)
            #vframes = np.moveaxis(resized_frames, -1, 0)  ## (3, 64, 224, 224)
            vframes = np.moveaxis(frames, -1, 0)  ## (3, 64, 224, 224)
            #print('frames shape', frames.shape)
        if self.fea_type=='mag':
            if self.return_stft:
                clean_audio = clean
                noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                          window=self.window, center=True)
                if self.a_only:
                    return self.get_noisy_features(noisy), self.get_noisy_features(clean), noisy_stft, clean_audio, 
                else:
                    if self.emotion:
                        return self.get_noisy_features(noisy), self.get_noisy_features(clean), noisy_stft, clean_audio, vframes, vframes_tensors_clipped
                    else:
                        return self.get_noisy_features(noisy), self.get_noisy_features(clean), noisy_stft, clean_audio, vframes
            else:
                if self.a_only:
                    return self.get_noisy_features(noisy), self.get_noisy_features(clean)
                else:
                    return self.get_noisy_features(noisy), self.get_noisy_features(clean), vframes, vframes_tensors_clipped
        

        elif self.fea_type=='lps':
            noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                          window=self.window, center=True)
            clean_audio = clean

            if self.a_only:
                return self.wav2spectrum(noisy, True), self.wav2spectrum(clean, False), noisy_stft, clean_audio
            else:
                return self.wav2spectrum(noisy, True), self.wav2spectrum(clean, False), noisy_stft, clean_audio, vframes, vframes_tensors_clipped

        elif self.fea_type=='pcs':
            #print('fea_type --> ',self.fea_type)
            noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                          window=self.window, center=True)
            
            clean_stft = librosa.stft(clean, win_length=window_size, n_fft=stft_size, hop_length=window_shift,
                                          window=self.window, center=True)
            clean_audio = clean
            
            Lp_noisy = PCS[:256] * np.transpose(np.log1p(np.abs(noisy_stft)), (1, 0))
            Lp_clean = PCS[:256] * np.transpose(np.log1p(np.abs(clean_stft)), (1, 0))

            epsilon = np.finfo(float).eps
            #D = librosa.stft(y, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
            
            noisy_stft = noisy_stft + epsilon

            if self.a_only:
                return Lp_noisy[np.newaxis, ...], Lp_clean[np.newaxis, ...], noisy_stft, clean_audio
            else:
                return Lp_noisy[np.newaxis, ...], Lp_clean[np.newaxis, ...], noisy_stft, clean_audio, vframes, vframes_tensors_clipped
    


class MOSEIDataModule(LightningDataModule):
    def __init__(self, batch_size=16, mask="mag", fea_type="mag", emotion=False, a_only=False, full_face=True, stage="train", dat="train", add_channel_dim=True):
    #def __init__(self, args, add_channel_dim=True):
        super(MOSEIDataModule, self).__init__()

        ## script considering lip information only
        if platform.system() == 'Windows':      
            self.train_dataset_batch = MOSEIDataset(join(DATA_ROOT, "train_noisy_data"), fea_type=fea_type, emotion=emotion, mask_type=mask,
                                                  add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True, full_face=full_face, dat="train")
            self.dev_dataset_batch = MOSEIDataset(join(DATA_ROOT, "valid_noisy_data"), fea_type=fea_type, mask_type=mask,
                                                add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True, full_face=full_face, dat="dev")    
        else:
            self.train_dataset_batch = MOSEIDataset(join(DATA_ROOT, "train_noisy_data"), fea_type=fea_type, emotion=emotion, mask_type=mask,
                                                  add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True, full_face=full_face, dat="train")
            self.dev_dataset_batch = MOSEIDataset(join(DATA_ROOT, "valid_noisy_data"), fea_type=fea_type, emotion=emotion, mask_type=mask,
                                                add_channel_dim=add_channel_dim, a_only=a_only, return_stft=True, full_face=full_face, dat="dev")    
            
        self.batch_size = batch_size
        

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset_batch, drop_last=True, batch_size=self.batch_size, num_workers=2, pin_memory=True,
                                          persistent_workers=True)
    def val_dataloader(self):

        return torch.utils.data.DataLoader(self.dev_dataset_batch, drop_last=True, batch_size=self.batch_size, num_workers=2, pin_memory=True,
                                           persistent_workers=True)
    #def test_dataloader(self):
    #   return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4, drop_last=True)




if __name__ == '__main__':

    DATA_ROOT = "../datasets/mosei/Raw/Videos/Full/segmented_4s_30fps_noisy_audio/"

    #dataset = TEDDataset(scenes_root=join(DATA_ROOT, "train_limited/scenes"), lips_root=join(DATA_ROOT, "lips/lips_train"), mask_type="mag",
    #                     add_channel_dim=True, a_only=False, return_stft=True, full_face=True, dat="train")
    train_dataset = MOSEIDataset(join(DATA_ROOT, "train_noisy_data"), fea_type="mag", emotion=True, mask_type="mag",
                                          add_channel_dim=True, a_only=False, return_stft=True, full_face=True, dat="train")
    print(train_dataset.files_list[:2])
    for i in tqdm(range(len(train_dataset)), ascii=True):
        data = train_dataset[i]
