from scipy.signal import windows as w
import platform

SEED = 999999
dB_levels = [0, 3, 6, 9]
sampling_rate = 16000
img_rows, img_cols = 224, 224
windows = w.hann
# windows = w.hamming

max_frames = 75

## IO_AVSE settings
stft_size = 511
window_size = 400
window_shift = 160
window_length = None
fading = False

max_utterance_length = 48000
num_frames = int(30 * (max_utterance_length / 16000))
num_stft_frames = 376#int((max_utterance_length - window_size + window_shift) / window_shift)

nb_channels, img_height, img_width = 3, img_rows, img_cols

## local machine
#DATA_ROOT = "C:/Experiments/Datasets/AV_dataset/AV_challenge_dataset/"
#LRS3_ROOT = "/media/mgo/Seagate/datasets/LRS3TED/"
#METADATA_ROOT = "/home/mgo/data/TED/metadata/"

if platform.system() == 'Windows':
	DATA_ROOT = "C:/Hussain/Research/Datasets/AVSE_challenge2_2023/avsec2_data"
else:
	#DATA_ROOT = "/scratch/t_hussain/datasets/challenge_data/avsec2"
	#DATA_ROOT = "/scratch/prj/aispehea/thussain/datasets/avsec2/"
	DATA_ROOT = "../datasets/mosei/Raw/Videos/Full/segmented_4s_30fps_noisy_audio/"
	visual_root = "../datasets/mosei/Raw/Videos/Full/segmented_4s_video_audio_30fps/"
	TEST_ROOT = "../datasets/mosei/Raw/Videos/Full/segmented_4s_30fps_noisy_audio/test_noisy_data"

save_path = "../mosei_enhanced_data/"
facial_feat_dir = "../datasets/mosei/Raw/Videos/Full/face_images_facial_features_30fps"


