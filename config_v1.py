from scipy.signal import windows as w
import platform

SEED = 999999
dB_levels = [0, 3, 6, 9]
sampling_rate = 16000
img_rows, img_cols = 224, 224
windows = w.hann
# windows = w.hamming

max_frames = 75
### Baseline settings
#stft_size = 512
#window_size = 512
#window_shift = 128
#window_length = None
#fading = False

## IO_AVSE settings
stft_size = 511
window_size = 400
window_shift = 160
window_length = None
fading = False

### DCCA settings
inpdim_size, outdim_size = 256, 256
#layer1 = [512, outdim_size]
#layer2 = [512, outdim_size]
#layer1 = [256, 256, 256, outdim_size]
#layer2 = [256, 256, 256, outdim_size]
layer1 = [1024, 512, outdim_size]
layer2 = [1024, 512, outdim_size]
apply_linear_cca = True
learning_rate = 1e-3
epoch_num = 1
batch_size = 128
reg_par = 1e-5
use_all_singular_values = False
###

max_utterance_length = 48000
num_frames = int(25 * (max_utterance_length / 16000))
num_stft_frames = 376#int((max_utterance_length - window_size + window_shift) / window_shift)

#nb_channels, img_height, img_width = 1, img_rows, img_cols
nb_channels, img_height, img_width = 3, img_rows, img_cols

## local machine
#DATA_ROOT = "C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset/"
#LRS3_ROOT = "/media/mgo/Seagate/datasets/LRS3TED/"
#METADATA_ROOT = "/home/mgo/data/TED/metadata/"

if platform.system() == 'Windows':
	#DATA_ROOT = "C:/Hussain/Research/Datasets/AVSE_challenge2_2023/avsec2_data"
	DATA_ROOT = "C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset"
else:
	#DATA_ROOT = "/scratch/t_hussain/datasets/challenge_data/avsec2"
	DATA_ROOT = "/scratch/prj/aispehea/thussain/datasets/avsec2/"

##avsec2 log dir
##test_root = "C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset/avse1_evalset"
#test_root = "C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset/avsec2_2023/avse2_evalset"
test_root = "/scratch/prj/aispehea/thussain/datasets/avsec2/dev/scenes"

#log_dir = "C:/Experiments/Codes/COGMhear_AV_Challenge/avsec2/logs"
##model_uid = log_dir+"/model_unet_transformer/full_face/fea_mag/loss_stoi/EP20/lightning_logs/version_7757067"
##ckpt_path = model_uid+"/checkpoints/epoch=15-step=69039.ckpt"

## only for unet-transformer
#test_data = "evalset"
test_data = "devset"

#save_path = "C:/Experiments/Codes/COGMhear_AV_Challenge/avsec2/avsec2_enhanced_data/model_unet_transformer/logs_recent/Fea-Mag/loss_mod_loss1"
save_path = "/scratch/prj/aispehea/thussain/codes/AVSEC2/avsec2_enhanced_data/model_unet/logs_recent/Fea-Mag/loss_stoi"
save_root = save_path+'/'+test_data
fusion = "concat"
##model_name="model_unet_lstm"
fea_type="mag"

##ENU server
#DATA_ROOT = "/media/a_hussain_disk/t_hussain/datasets/challenge_data/"
#DATA_ROOT = "/scratch/t_hussain/datasets/challenge_data/avsec2"

#DATA_ROOT = "/scratch/t_hussain/datasets/challenge_data/"
#LRS3_ROOT = "/scratch/t_hussain/datasets/LRS3-TED/"
#METADATA_ROOT = "/scratch/t_hussain/datasets/challenge_data/metadata/"
