import torch
import torchaudio
from src.model import KWSNet
from config import params
from src.dataset import transforms
import wandb
import numpy as np
device = torch.device('cpu')
model = KWSNet(params)
model.load_state_dict(torch.load("kws_model_2.pth", map_location=device))
model = model.eval()
wav_file = "ptk_2.wav"
wav, sr = torchaudio.load(wav_file)
transformed_wav = transforms['test'](wav)
probs = model.inference(transformed_wav, params["window_size"])

probs = np.array(probs)
print(probs)
print(np.sum(probs>0.4))
