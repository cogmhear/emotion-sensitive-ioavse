import random

import librosa, os
import numpy as np
import torch
#torch.multiprocessing.set_start_method('spawn')# good solution !!!!

import torch.nn as nn
import soundfile as sf
from pytorch_lightning import LightningModule
from torch.nn import functional as F
import math
from torchaudio.transforms import MelScale
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config_mosei import *
from loss import STOILoss
from consts import DEVICE
from utils.models.resnet import BasicBlock, ResNet
from DeepCCAModels import MlpNet
from utils.nn import TCN, conv_block, threeD_to_2D_tensor, unet_conv, unet_upconv, up_conv, weights_init
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pdb

class VisualFeatNet(nn.Module):
    def __init__(self, tcn_options, hidden_dim=256, num_classes=500,
                 relu_type='prelu', extract_feats=False):
        super(VisualFeatNet, self).__init__()
        self.extract_feats = extract_feats
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(3, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.tcn = TCN(input_size=self.backend_out,
                       num_channels=[hidden_dim * len(tcn_options['kernel_size']) * tcn_options['width_mult']] *
                       tcn_options['num_layers'],
                       num_classes=num_classes,
                       tcn_options=tcn_options,
                       dropout=tcn_options['dropout'],
                       relu_type=relu_type,
                       dwpw=tcn_options['dwpw'],
                       )

    def forward(self, x, lengths):
        B, C, T, H, W = x.size()
        if type(lengths) == int:
            lengths = [lengths] * B
        x = self.frontend3D(x)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        return self.tcn(x, lengths, B, self.extract_feats)
        #return self.tcn(x, lengths, B, self.extract_feats).squeeze(2).permute(0,2,1)


class UNet(nn.Module):
    def __init__(self, filters=64, input_nc=2, output_nc=2, av_embedding=1024, a_only=True, emotion=False, activation='Sigmoid', model_name='unet'):
        super(UNet, self).__init__()
        self.a_only = a_only
        self.emotion = emotion
        self.model_name = model_name
        self.conv1 = unet_conv(input_nc, filters)
        self.conv2 = unet_conv(filters, filters * 2)
        self.conv3 = conv_block(filters * 2, filters * 4)
        self.conv4 = conv_block(filters * 4, filters * 8)
        self.conv5 = conv_block(filters * 8, filters * 8)
        self.frequency_pool = nn.MaxPool2d([2, 1])

        ## additional blocks for attention concat
        self.conv6 = conv_block(filters * 8, filters * 8)
        self.frequency_pool1 = nn.MaxPool2d([4, 4])
        self.conv7 = conv_block(filters * 8, filters * 8)
        self.frequency_pool2 = nn.MaxPool2d([2, 4])
        self.conv8 = conv_block(filters * 8, filters * 8)
        self.frequency_pool3 = nn.MaxPool2d([1, 4])


        if not a_only:
            self.upconv1 = up_conv(av_embedding, filters * 8)
        else:
            self.upconv1 = up_conv(filters * 8, filters * 8)
            
        self.upconv2 = up_conv(filters * 16, filters * 8, scale_factor=(2., 1.))
        self.upconv3 = up_conv(filters * 16, filters * 4, scale_factor=(2., 1.))
        self.upconv4 = up_conv(filters * 8, filters * 2, scale_factor=(2., 1.))
        self.upconv5 = unet_upconv(filters * 4, filters)
        self.upconv6 = unet_upconv(filters * 2, output_nc, True)
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'ReLu':
            self.activation = nn.ReLU()

    
    def forward(self, mix_spec, visual_feat=None, emo_model=None):

        print('model name -->', self.model_name)
        ## UNet-Encoder
        conv1feat = self.conv1(mix_spec)
        conv2feat = self.conv2(conv1feat)
        conv3feat = self.conv3(conv2feat)
        conv3feat = self.frequency_pool(conv3feat)
        conv4feat = self.conv4(conv3feat)
        conv4feat = self.frequency_pool(conv4feat)
        conv5feat = self.conv5(conv4feat)
        conv5feat = self.frequency_pool(conv5feat)    
        enc_feat = conv5feat

        if self.a_only:
            av_feat = enc_feat
        else:
            upsample_visuals = F.interpolate(visual_feat, (8, 64))
            #av_feat = torch.cat((conv5feat, upsample_visuals), dim=1)
            if self.emotion:
                #Batch, Frames, Channel = emo_model["emotion_feat"].shape
                #emotion_feat = emo_model["emotion_feat"].view(B, C, F)
                
                emotion_feat = torch.moveaxis(emo_model, -1, 1)  ##
                #emotion_feat = emotion_feat.unsqueeze(dim=2)
                upsample_emotion_feat = F.interpolate(emotion_feat, (8,64))
                av_feat = torch.cat((enc_feat, upsample_visuals, upsample_emotion_feat), dim=1)
            else:
                av_feat = torch.cat((enc_feat, upsample_visuals), dim=1)
        
        ## UNet-Decoder
        upconv1feat = self.upconv1(av_feat)
        upconv2feat = self.upconv3(torch.cat((upconv1feat, conv4feat), dim=1))
        upconv3feat = self.upconv4(torch.cat((upconv2feat, conv3feat), dim=1))
        upconv4feat = self.upconv5(torch.cat((upconv3feat, conv2feat), dim=1))
        predicted_mask = self.upconv6(torch.cat((upconv4feat, conv1feat), dim=1))
        pred_mask = self.activation(predicted_mask)
        
        return torch.mul(pred_mask, mix_spec)

def build_audio_unet(filters=64, input_nc=1, output_nc=1, visual_feat_dim=1280, weights='', a_only=False, emotion=False, activation="Sigmoid", model_name='unet'):
    net = UNet(filters, input_nc, output_nc, visual_feat_dim, a_only=a_only, emotion=emotion,activation=activation, model_name=model_name)
    net.apply(weights_init)

    if len(weights) > 0:
        print('Loading weights for UNet')
        net.load_state_dict(torch.load(weights))
    return net


def build_visualfeat_net(weights='', extract_feats=True):
    net = VisualFeatNet(tcn_options=dict(num_layers=4, kernel_size=[3], dropout=0.2, dwpw=False, width_mult=2),
                        relu_type="prelu",
                        extract_feats=extract_feats)
    if len(weights) > 0:
        print('Loading weights for lipreading stream')
        net.load_state_dict(torch.load(weights))
    return net


class IO_AVSE_DNN(LightningModule):
    def __init__(self, nets, args, val_dataset=None):
        super(IO_AVSE_DNN, self).__init__()
        #self.a_only = a_only    
        #self.lr = lr

        self.lr = args.lr
        ##self.args = args
        self.a_only = args.a_only
        self.model_name = args.model_name
        #self.net_visualfeat, self.net_audio_unet, self.dcca_net = nets
        self.loss = args.loss
        self.emotion=args.emotion

        #self.net_visualfeat, self.net_audio_unet, self.emotion_net = nets
        self.net_visualfeat, self.net_audio_unet = nets

        if self.loss.lower() == "l1":
            self.compute_loss = F.l1_loss
        elif self.loss.lower() == "l2":
            self.compute_loss = F.mse_loss
        elif self.loss.lower() == "stoi":
            self.compute_loss = STOILoss()
        else:
            raise NotImplementedError("{} is currently unavailable as loss function. Select one of l1, l2 and stoi".format(loss))

        self.val_dataset = val_dataset

    def forward(self, input):
        emotion_out = {}
        noisy_audio_spec = input['noisy_audio_spec']

        #if self.args.a_only:
        if self.a_only:
            pred_mask = self.net_audio_unet(noisy_audio_spec.float(), self.model_name)
        else:
            lip_images = input['lip_images']
            visual_feat = self.net_visualfeat(lip_images.float(), 64)
            if self.emotion:
                #inputs["emo_images"] = torch.unsqueeze(data["emo_images"], dim=0).to(DEVICE)
                emotion_out = input['emo_images']
            
            pred_mask = self.net_audio_unet(noisy_audio_spec.float(), visual_feat, emotion_out) 
        
        return pred_mask

    def training_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def training_epoch_end(self, outputs):
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                rand_int = random.randint(0, len(self.val_dataset))
                data = self.val_dataset[rand_int]

                inputs = {"noisy_audio_spec": torch.from_numpy(data["noisy_audio_spec"][np.newaxis, ...]).to(DEVICE)}

                #if not self.args.a_only:
                if not self.a_only:
                    if self.emotion:
                        inputs["emo_images"] = torch.unsqueeze(data["emo_images"], dim=0).to(DEVICE)

                    if isinstance(data["lip_images"], torch.Tensor):
                        #inputs["lip_images"] = data["lip_images"].to(DEVICE)
                        inputs["lip_images"] = torch.unsqueeze(data["lip_images"], dim=0).to(DEVICE)
                    else:
                        inputs["lip_images"] = torch.from_numpy(data["lip_images"][np.newaxis, ...]).to(DEVICE)

                pred_mag = self(inputs)[0][0].cpu().numpy()
                noisy_phase = np.angle(data["noisy_stft"])
                estimated = pred_mag * (np.cos(noisy_phase) + 1.j * np.sin(noisy_phase))
                estimated_audio = librosa.istft(estimated, win_length=window_size, hop_length=window_shift, window="hann")
                noisy = librosa.istft(data["noisy_stft"], win_length=window_size, hop_length=window_shift, window="hann")
                tensorboard.add_audio("{}/clean".format(self.current_epoch), data["clean"][np.newaxis, ...], sample_rate=16000)
                tensorboard.add_audio("{}/noisy".format(self.current_epoch), noisy[np.newaxis, ...], sample_rate=16000)
                tensorboard.add_audio("{}/enhanced".format(self.current_epoch), estimated_audio[np.newaxis, ...], sample_rate=16000)
                
    def cal_loss(self, batch_inp):
        mask = batch_inp["clean_audio_spec"].float()
        pred_mask  = self(batch_inp)
        loss = self.compute_loss(pred_mask, mask) ## ccstoi, stoi, l1, l2 as loss function

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.66, patience=2),
                "monitor": "val_loss_epoch",
            },
        }


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    
    #test_audio_data = torch.rand((1, 1, num_stft_frames, stft_size // 2 + 1))
    test_audio_data = torch.rand((1, 1, 256, 256))

    audio_net = build_audio_unet(a_only=True)
    pred_mask = audio_net(test_audio_data).detach().numpy()
    print("Audio-only UNet", pred_mask.shape)
    print(np.min(pred_mask), np.max(pred_mask))
    

    test_visual_data = torch.rand([1, 3, 64, 224, 224])
    visual_net = build_visualfeat_net(extract_feats=True)
    visual_feat = visual_net(test_visual_data, 64)
    print("Visual feat", visual_feat.shape)
    
    net = build_audio_unet(filters=64, a_only=False, visual_feat_dim=1024)
    print("Audio-visual UNet", net(test_audio_data, visual_feat).shape)
    pred_mask = net(test_audio_data, visual_feat)

    '''
    audiofeat_net = build_audio_unet(a_only=False)
    visual_net = build_visualfeat_net(extract_feats=True)
    #fusion_net = FusionNet(a_only=False)
    
    #net = IO_AVSE_DNN((visual_net, audiofeat_net), )
    net.eval()
    audiofeat_net.eval()
    visual_net.eval()
    # test_audio_data = torch.rand((1, num_stft_frames, stft_size // 2 + 1))
    # test_visual_data = torch.rand([1, 1, num_frames, 88, 88])
    test_audio_data = torch.rand((1, 1, num_stft_frames, stft_size // 2 + 1))
    test_visual_data = torch.rand([1, 3, num_frames, 224, 224])

    with torch.no_grad():
        start_time = time.time()
        pred_mask = audiofeat_net(test_audio_data).detach().numpy()
        print(time.time() - start_time)
        print("Audio-only Feat", pred_mask.shape)
        # print(np.min(pred_mask), np.max(pred_mask))
        start_time = time.time()
        visual_feat = visual_net(test_visual_data, 75)
        print(time.time() - start_time)
        print("Visual feat", visual_feat.shape)
        # print(net)
        warmup = net({'noisy_audio_spec': test_audio_data, "lip_images": test_visual_data})
        start_time = time.time()
        print("Audio-visual Net", net({'noisy_audio_spec': test_audio_data, "lip_images": test_visual_data}).shape)
        print(time.time() - start_time)
    '''