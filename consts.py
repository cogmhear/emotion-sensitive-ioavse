import os.path

import torch

##DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEGMENT_STRIDE = 15.0  # use 7.5 for sliding window, else use 15.0 for non overlap
SEGMENT_DURATION = 15.0
REQUIRED_SAMPLE_RATE = 16000
FACE_DETECT_THRESH = 0.9
MAX_BATCH = 512  # maximum number of face images to pass to the visual model at once when using batch mode.
IMG_SIZE = 224

VISUAL_MODEL_PATH = os.path.join("checkpoints", "visual_model.pt")

LINGUISTIC_MODEL_EN_PATH = os.path.join("checkpoints", "linguistic_head_en.ckpt")
LINGUISTIC_MODEL_ZH_PATH = os.path.join("checkpoints", "linguistic_head_zh_ldc.ckpt")
AUDIO_MODEL_PATH = os.path.join("checkpoints", "audio_head.pt")

LINGUISTIC_MODEL_ZH_URL = "https://github.com/islam-nassar/ccu_mini_eval/releases/download/misc/linguistic_head_zh.ckpt"
LINGUISTIC_MODEL_EN_URL = "https://github.com/islam-nassar/ccu_mini_eval/releases/download/misc/linguistic_head_en.ckpt"
AUDIO_MODEL_URL = "https://github.com/islam-nassar/ccu_mini_eval/releases/download/misc/audio_model_trill.pt"
