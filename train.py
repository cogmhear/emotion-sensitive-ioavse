import numpy as np
import torch
from dataset_mosei import MOSEIDataModule
from model_cnn_lstm import *  
import platform
from pytorch_lightning import Trainer


SEED = 1143
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.generic import str2bool
from emo_models import MTLModel
from consts import *


#DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main(args):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch")
    datamodule = MOSEIDataModule(batch_size=args.batch_size, mask=args.mask, fea_type=args.fea_type, emotion=args.emotion, a_only=args.a_only, stage=args.stage, dat=args.dat, full_face=args.full_face)
    if args.emotion:
        visual_feat_dim=1024+512
    else:
        visual_feat_dim=1024

    audio_unet = build_audio_unet(filters=64, a_only=args.a_only,  emotion=args.emotion, visual_feat_dim=visual_feat_dim, model_name=args.model_name)
    visualfeat_net = build_visualfeat_net(extract_feats=True) if not args.a_only else None
    

    save_dir = '../Emotion_AVSE/AVSE_AttnUNet/logs'

    if args.a_only:
        log_dir = save_dir+f'/mosei_avse/model_{args.model_name}/ASE/full_face/fea_{args.fea_type}/loss_{args.loss}/EP{args.max_epochs}'
    else:
        log_dir = save_dir+f'/mosei_avse/model_{args.model_name}/AVSE/full_face/fea_{args.fea_type}/loss_{args.loss}/EP{args.max_epochs}'


    #model = IO_AVSE_DNN((visualfeat_net, audio_unet, emo_net), args, datamodule.dev_dataset_batch)
    model = IO_AVSE_DNN((visualfeat_net, audio_unet), args, datamodule.dev_dataset_batch)


    trainer = Trainer.from_argparse_args(args, default_root_dir=log_dir, callbacks=[checkpoint_callback])
    if args.tune:
        trainer.tune(model, datamodule)
    else:
        trainer.fit(model, datamodule)


#python train.py --a_only False --emotion True --stage train --max_epochs 5 --gpu 1 --batch_size 8 --loss l1 --full_face True --model_name unet --fea_type mag


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, default=False)
    parser.add_argument("--tune", type=str2bool, default=False)
    parser.add_argument("--emotion", type=str2bool, required=True)  ## unet, unet_lstm, unet_transformer
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00158)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--mask", type=str, default="mag")
    parser.add_argument("--stage", type=str, required=True, default="train")
    parser.add_argument("--model_name", type=str, required=True, default="unet")  ## unet, unet_lstm, unet_transformer
    parser.add_argument("--full_face", type=str2bool, required=True)
    parser.add_argument("--dat", type=str, default="train")
    parser.add_argument("--fea_type", type=str, default="mag")
    parser.add_argument("--forward_mode", type=str, default='4sum_pre_logits',
                        choices=['average_pre_logits', 'average_logits', '4sum_pre_logits'])

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
