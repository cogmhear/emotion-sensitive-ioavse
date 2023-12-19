


dataset="/Data/user_hussain/Beamforming_using_NN/data/MHINT_multispeaker_noisy_dataset/Test/"
#noisy_wav_list=$dataset/"Server_MHINT_1chann_3speak_30uttEach_3noise_6snr_test_noisy-only_wav_list.txt"
#noisy_wav_list=$dataset/"Server-123_MHINT_1chann_3speak_test_noisy-only_wav_list.txt"
file_name="Server_MHINT_Test_Noisy_multispeak_1ch_list.txt"
noisy_wav_list="${dataset}/Server_SimData-RealData_Noise-only_1ch_list.txt"
#python read_wav.py -i $noisy_wav_list
#Train_Noisy_paths = get_filenames(noisy_wav_list)
#python beamform.py $noisy_wav_list "$@"
python beamform.py $noisy_wav_list

loss = "stoi"
a_only = True
log_dir = "./log/speech_noises/audio_only/${loss}"
gpu=1
max_epoch=10

python train.py --log_dir ./logs --a_only False --gpu 1 --max_epochs 15 --loss stoi


python main.py --mode train --path 3/ --data 3 --gpus 3
python main.py --mode test --path 3/ --data 3 --gpus 3
python main.py --mode test --path 3/ --data 6 --gpus 3
python main.py --mode test --path 3/ --data 9 --gpus 3
python main.py --mode test --path 3/ --data 4 --gpus 3
python main.py --mode test --path 3/ --data 7 --gpus 3
python main.py --mode test --path 3/ --data 10 --gpus 3
python main.py --mode train --path 6/ --data 6 --gpus 3
python main.py --mode test --path 6/ --data 3 --gpus 3
python main.py --mode test --path 6/ --data 6 --gpus 3
python main.py --mode test --path 6/ --data 9 --gpus 3
python main.py --mode test --path 6/ --data 4 --gpus 3
python main.py --mode test --path 6/ --data 7 --gpus 3
python main.py --mode test --path 6/ --data 10 --gpus 3
python main.py --mode train --path 9/ --data 9 --gpus 3
python main.py --mode test --path 9/ --data 3 --gpus 3
python main.py --mode test --path 9/ --data 6 --gpus 3
python main.py --mode test --path 9/ --data 9 --gpus 3
python main.py --mode test --path 9/ --data 4 --gpus 3
python main.py --mode test --path 9/ --data 7 --gpus 3
python main.py --mode test --path 9/ --data 10 --gpus 3
python main.py --mode s_train --path soft/ --gpus 3
python main.py --mode s_test --path soft/ --data 3 --gpus 3
python main.py --mode s_test --path soft/ --data 6 --gpus 3
python main.py --mode s_test --path soft/ --data 9 --gpus 3
python main.py --mode s_test --path soft/ --data 4 --gpus 3
python main.py --mode s_test --path soft/ --data 7 --gpus 3
python main.py --mode s_test --path soft/ --data 10 --gpus 3
python main.py --mode base --path base/ --gpus 3
python main.py --mode test --path base/ --data 4 --gpus 3
python main.py --mode test --path base/ --data 7 --gpus 3
python main.py --mode test --path base/ --data 10 --gpus 3
python main.py --mode test --path base/ --data 3 --gpus 3
python main.py --mode test --path base/ --data 6 --gpus 3




from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import GridDataModule
from model import *
from utils.generic import str2bool
import pdb

def main():

    audio_unet = build_audio_unet(filters=64, a_only=args.a_only, visual_feat_dim=1024)
    visualfeat_net = build_visualfeat_net(extract_feats=True) if not args.a_only else None
    model = IO_AVSE_DNN((visualfeat_net, audio_unet), args, datamodule.test_dataset)
    
    trainer = Trainer.from_argparse_args(args, default_root_dir=args.log_dir, callbacks=[checkpoint_callback])

    if args.tune:
        trainer.tune(model, datamodule)
    else:
        trainer.fit(model, datamodule)


if __name__ == '__main__'
    parser = ArgumentParser()
    parser.add_argument("--a_only", type=str2bool, default=True)
    parser.add_argument("--tune", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00158)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--loss", type=str, default="l1", required=True)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch")


    main()
