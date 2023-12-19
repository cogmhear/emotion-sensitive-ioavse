from argparse import ArgumentParser
from os import makedirs
from os.path import isfile, join

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import cv2
from decord import VideoReader
from decord import cpu
from skimage.transform import resize
from torch.nn import functional as F
from config_mosei import *
from dataset import TEDDataModule
from model_cnn_lstm import *
from face_detector import *
from emo_models import *
from utils.generic import str2bool
import matplotlib.pyplot as plt
import math, glob, pdb
from consts import DEVICE

window = "hann"

def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)
    out = F.pad(x, pads, "constant", 0)
    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]:-pad[1]]
    return x

def main(args):
    if args.a_only:
        enhanced_root = os.path.join(save_path, args.model_name, "ase", args.fea_type, args.loss, "EP{}".format(args.max_epochs))

    if args.emotion:
        enhanced_root = os.path.join(save_path, args.model_name, "emotion_avse", args.fea_type, args.loss, "EP{}".format(args.max_epochs))
    else:
        enhanced_root = os.path.join(save_path, args.model_name, "avse", args.fea_type, args.loss, "EP{}".format(args.max_epochs))

    makedirs(enhanced_root, exist_ok=True)

    if args.emotion:
        vis_feat_dim=1024+512
        emo_net = MTLModel(num_classes=8, forward_mode='4sum_pre_logits')
        emo_net.to(DEVICE)
        emo_net.eval()  
    else:
        vis_feat_dim=1024
        emo_net=None
    
    audio_unet = build_audio_unet(filters=64, a_only=args.a_only, visual_feat_dim=vis_feat_dim, model_name=args.model_name)
    visual_net = build_visualfeat_net(extract_feats=True) if not args.a_only else None

    if args.a_only:
        if args.loss=="l1":
            ckpt_path = "../Emotion_AVSE/AVSE_AttnUNet/logs/mosei_avse/model_unet/ASE/full_face/fea_mag/loss_l1/EP10/lightning_logs/version_10170482/checkpoints/epoch=4-step=10820.ckpt"
        elif args.loss=="stoi":
            ckpt_path = "../Emotion_AVSE/AVSE_AttnUNet/logs/mosei_avse/model_unet/ASE/full_face/fea_mag/loss_stoi/EP10/lightning_logs/version_10170482/checkpoints/epoch=9-step=21640.ckpt"
        else:
            print("invalid loss function for audio-only framework")
    else:
        if args.emotion:
            if args.loss=="l1":
                ckpt_path = None
            elif args.loss=="stoi":
                ckpt_path = None
            else:
                print("invalid loss function for emotion avse framework")
        else:
            if args.loss=="l1":
                ckpt_path = None
            elif args.loss=="stoi":
                ckpt_path = None
            else:
                print("invalid loss function for audio-only framework")

  
    if ckpt_path.endswith("ckpt") and isfile(ckpt_path):
        model = IO_AVSE_DNN.load_from_checkpoint(ckpt_path, nets=(visual_net, audio_unet, emo_net), args=args, strict=False,  map_location=DEVICE)
    else:
        raise FileNotFoundError("Cannot load model weights: {}".format(args.ckpt_path))
    
    model.eval()
    model.to(DEVICE)
    #£model.to('cpu')

    ts_files = glob.glob(os.path.join(test_root, '*_noisy_snr-*.wav'))

    chunk_size = 40900
    epsilon = np.finfo(float).eps

    with torch.no_grad():
        for test_file in ts_files:
            try:                
                noisy_filename = test_file.split("/")[-1] 
                fileparts = noisy_filename.split('_noisy_snr-')  ## extract '210618_6_1' from filename '210618_6_1_noisy_snr-n6.wav' 
                clean_filename = fileparts[0]+"_target.wav"
                video_filename = fileparts[0]+"_silent.mp4"

                mp4_file = join(visual_root, video_filename)
                enh_filename = noisy_filename[:-4]+'_enhanced.wav'
                
                ny_file = join(test_root, test_file)
                cl_file = join(test_root, clean_filename)

                noisy, _ = librosa.load(ny_file, sr=16000)
                clean, _ = librosa.load(cl_file, sr=16000)

                if clean.shape[0] > 48000:
                    diff =  clean.shape[0] - 48000                    
                    if diff > 6000:
                        clip_idx = random.randint(0, 6000) ## temp sol: limit the clip index to avoid out of bound issue for visual frames
                    else:
                        clip_idx = random.randint(0, clean.shape[0] - 48000)
                    
                    video_idx = max(int((clip_idx / 16000) * 30) - 2, 0)  ##

                    clean_clip = clean[clip_idx:clip_idx + 40900]
                    noisy_clip = noisy[clip_idx:clip_idx + 40900]
                    clean = clean_clip
                    noisy = noisy_clip            
                else:
                    video_idx = -1
                    clean = np.pad(clean, pad_width=[0, 48000 - clean.shape[0]], mode="constant")
                    noisy = np.pad(noisy, pad_width=[0, 48000 - noisy.shape[0]], mode="constant")
                    clean = clean[:40900]
                    noisy = noisy[:40900]        
                
                if not self.a_only:
                    vframes, face_imgs, bounding_boxes, bboxes, probs = detect_faces_from_video_without_saving(video_file_path=mp4_file, return_frames=True)
                
                    num_frames = 64                
                    vframes_tensors = torch.stack([test_transforms(img).to(DEVICE) for img in face_imgs], dim=0)
                    vframes_tensors_clipped = vframes_tensors[video_idx:video_idx + num_frames]

                    if vframes_tensors_clipped.size(0) < num_frames:
                        num_rows_to_add = num_frames - vframes_tensors_clipped.size(0)
                        zeros_to_add = torch.zeros((num_rows_to_add, 3, 224, 224), dtype=vframes_tensors_clipped.dtype).to(DEVICE)
                        vframes_tensors_clipped = torch.cat((vframes_tensors_clipped, zeros_to_add), dim=0)

                    with torch.no_grad():
                        emotion_feat, emotion_cat, emotion_linear, emotion_classifier = emo_net(vframes_tensors_clipped, num_frames)

                    vframes = vframes_tensors_clipped
                    vframes = torch.moveaxis(vframes, 1, 0)  ##

                restored_waveform = np.empty(len(noisy),)
                noisy_stft = librosa.stft(noisy, n_fft=stft_size, hop_length=window_shift, win_length=window_size, window=window, center=True)
                noisy_phase = np.exp(1j * np.angle(noisy_stft))

                if args.fea_type=='lps':
                    noisy_stft = noisy_stft + epsilon
                    noisy_lps = np.log10(abs(noisy_stft)**2)
                    mean = np.mean(noisy_lps, axis=1).reshape(-1,1)
                    std = np.std(noisy_lps, axis=1, dtype = np.float32, ddof=1).reshape(-1,1)
                    if Normalize:
                        noisy_spec = np.float32((noisy_lps - mean)/std)
                else:
                    noisy_spec = np.abs(noisy_stft)[np.newaxis, ...]

                inputs = {"noisy_audio_spec": torch.from_numpy(noisy_spec[np.newaxis, ...]).to(DEVICE)}

                if not self.a_only:
                    if self.emotion:
                        inputs["emo_images"] = torch.unsqueeze(emotion_feat, dim=0).to(DEVICE)
                    if isinstance(vframes, torch.Tensor):
                        inputs["lip_images"] = torch.unsqueeze(vframes, dim=0).to(DEVICE)
                    else:
                        inputs["lip_images"] = torch.from_numpy(vframes[np.newaxis, ...]).to(DEVICE)

                pred_mag = model(inputs)[0][0].cpu().numpy()
                noisy_phase = np.angle(noisy_spec)
                estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                estimated_audio = librosa.istft(estimated, win_length=window_size, hop_length=window_shift, window="hann")
 
                enh_path= enhanced_root+'/'+enh_filename
                sf.write(enh_path, estimated_audio, 16000)

                cln_path= enhanced_root+'/'+clean_filename
                sf.write(cln_path, clean, 16000)

                ny_path= enhanced_root+'/'+noisy_filename
                sf.write(ny_path, noisy, 16000)

                

                """    
                pred = model(inputs).cpu()
                estimated_spec = pred.numpy()[0][0] * noisy_spec.squeeze()

                estimated_sig = estimated_spec * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                #estimated_audio = librosa.istft(estimated_sig.T, win_length=window_size, hop_length=window_shift, window="hann", length=chunk_size)
                estimated_audio = librosa.istft(estimated_sig, win_length=window_size, hop_length=window_shift, window="hann", length=chunk_size)

                # Restore the processed chunk to the full waveform
                restored_waveform[start_idx:end_idx] = estimated_audio

                '''
                ### modify this part when considering full-length audio-visual test data                
                ## video frame rate and chunks
                video_capture = cv2.VideoCapture(mp4_file)
                frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
                frames_per_chunk = int(np.ceil(frame_rate / 16000 * chunk_size))

                # Divide the test data into chunks of size 40900
                #num_audio_chunks = int(np.ceil(len(noisy) / chunk_size)) ## 2.45-->3 
                num_audio_chunks = int(np.floor(len(noisy) / chunk_size))  ## 2.45-->2

                restored_waveform = np.empty(len(noisy),)

                vr = VideoReader(mp4_file, ctx=cpu(0))
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()  ##(152, 224, 224, 3)
                #resized_frames = resize(frames, (64, 88, 88, 3), anti_aliasing=True)
                #frames = resized_frames
                frames = np.moveaxis(frames, -1, 0) ## (3, 152, 224, 224)
                frames = frames[np.newaxis, ...] ## (1, 3, 152, 224, 224) --> add batch axis
                #frames = np.mean(frames, axis=1)  ## (1, 152, 224, 224)
                for idx in range(num_audio_chunks):
                    start_idx = idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(noisy))

                    audio_chunk = noisy[start_idx:end_idx]

                    vis_start_idx = idx * frames_per_chunk
                    vis_end_idx = min(vis_start_idx + frames_per_chunk, frames.shape[2])

                    visual_chunk = frames[:,:,vis_start_idx:vis_end_idx,:,:]

                    noisy_stft = librosa.stft(audio_chunk, n_fft=stft_size, hop_length=window_shift, win_length=window_size, window=window, center=True)
                    noisy_phase = np.exp(1j * np.angle(noisy_stft))

                    if fea_type=='lps':
                        noisy_stft = noisy_stft + epsilon
                        noisy_lps = np.log10(abs(noisy_stft)**2)
                        mean = np.mean(noisy_lps, axis=1).reshape(-1,1)
                        std = np.std(noisy_lps, axis=1, dtype = np.float32, ddof=1).reshape(-1,1)
                        if Normalize:
                            noisy_spec = np.float32((noisy_lps - mean)/std)
                    else:
                        noisy_spec = np.abs(noisy_stft)

                    noisy_spec = noisy_spec[np.newaxis, ...] ## (1, 256, 256) --> add batch axis
                    #inputs = {"noisy_audio_spec": torch.from_numpy(noisy_spec[np.newaxis, ...]).to(model.device)}  
                    #inputs = {"lip_images": torch.from_numpy(visual_chunk).to(model.device)} 
                    test_noisy_chunk = torch.from_numpy(noisy_spec[np.newaxis, ...]).to(model.device)
                    lip_images_chunk = torch.from_numpy(visual_chunk).to(model.device)
                    inputs = {"noisy_audio_spec": test_noisy_chunk, "lip_images": lip_images_chunk}  
                
                    pred = model(inputs).cpu()
                    estimated_spec = pred.numpy()[0][0] * noisy_spec.squeeze()

                    estimated_sig = estimated_spec * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                    #estimated_audio = librosa.istft(estimated_sig.T, win_length=window_size, hop_length=window_shift, window="hann", length=chunk_size)
                    estimated_audio = librosa.istft(estimated_sig, win_length=window_size, hop_length=window_shift, window="hann", length=chunk_size)

                    # Restore the processed chunk to the full waveform
                    restored_waveform[start_idx:end_idx] = estimated_audio
                '''


                # Pad zeros to the last chunk if its size is less than 40900
                last_chunk_size = len(noisy) % chunk_size
                #pdb.set_trace()
                if last_chunk_size != 0:
                    last_chunk = np.zeros(chunk_size)
                    last_chunk[:last_chunk_size] = noisy[-last_chunk_size:]
                    audio_chunk = last_chunk

                    noisy_stft = librosa.stft(last_chunk, n_fft=stft_size, hop_length=window_shift, win_length=window_size, window=window, center=True)
                    noisy_phase = np.exp(1j * np.angle(noisy_stft))

                    if fea_type=='lps':
                        noisy_stft = noisy_stft + epsilon
                        noisy_lps = np.log10(abs(noisy_stft)**2)
                        mean = np.mean(noisy_lps, axis=1).reshape(-1,1)
                        std = np.std(noisy_lps, axis=1, dtype = np.float32, ddof=1).reshape(-1,1)
                        if Normalize:
                            noisy_spec = np.float32((noisy_lps - mean)/std)
                    else:
                        noisy_spec = np.abs(noisy_stft)[np.newaxis, ...]
                    
                    last_visual_chunk = frames[:,:,vis_end_idx:frames.shape[2],:,:]
                    padd_vis_frames = frames_per_chunk - last_visual_chunk.shape[2]
                    # Pad zeros to the last video chunk if its size is less than 40900
                    padding_vis_frames = np.zeros((1, 3, padd_vis_frames, frames.shape[3], frames.shape[3]), dtype=np.uint8)
                    last_vis_chunk_data = np.concatenate([last_visual_chunk, padding_vis_frames], 2)


                    test_noisy_chunk = torch.from_numpy(noisy_spec[np.newaxis, ...]).to(model.device)
                    lip_images_chunk = torch.from_numpy(last_vis_chunk_data).to(model.device)
                    inputs = {"noisy_audio_spec": test_noisy_chunk, "lip_images": lip_images_chunk}  

                    pred = model(inputs).cpu()

                    estimated_spec = pred.numpy()[0][0] * noisy_spec.squeeze()

                    estimated_sig = estimated_spec * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                    estimated_audio = librosa.istft(estimated_sig, win_length=window_size, hop_length=window_shift, window="hann", length=chunk_size)

                    # Remove the padded zeros from the processed last chunk
                    processed_last_chunk = estimated_audio[:last_chunk_size]

                    # Restore the processed last chunk to the full waveform
                    restored_waveform[-last_chunk_size:] = processed_last_chunk

                restored_waveform /= np.max(np.abs(restored_waveform))
                #pdb.set_trace()
                # Save the restored waveform for the current test file
                save_path= save_root+'/'+enh_filename
                sf.write(save_path, restored_waveform, 16000)
                """
            except Exception as e:
                print(e)

#python test_mosei.py --a_only False --emotion True --max_epochs 10 --batch_size 8 --loss l1 --full_face True --model_name unet --fea_type mag

##python test_avsec2.py --a_only False --full_face True
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, required=True)
    #parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--oracle", type=str2bool, required=False)
    parser.add_argument("--mask", type=str, default="mag")
    parser.add_argument("--model_name", type=str, default="unet")
    parser.add_argument("--emotion", type=str2bool, required=True)  ## unet, unet_lstm, unet_transformer
    parser.add_argument("--fea_type", type=str, default="mag")
    parser.add_argument("--batch_size", type=int, default=16)
    #parser.add_argument("--loss", type=str, default="bce")
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--lr", type=float, default=0.00158)
    parser.add_argument("--full_face", type=str2bool, required=True)
    parser.add_argument("--max_epochs", type=int, default=10)

    #parser.add_argument("--fusion", type=str, required=True, default="concat")
    #parser.add_argument("--test_data", type=str, required=True, default="devset")

    args = parser.parse_args()
    main(args)
