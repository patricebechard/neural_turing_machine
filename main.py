import torch
import numpy as np

from model import Vanilla_LSTM, NTM
from train import train
from utils import *

if __name__ == "__main__":

	model = NTM('lstm').to(device)

	loss_tracker = train(model, n_updates=50000, print_every=1000, save_model=True)

	np.savetxt('loss_ntm_lstm.csv', loss_tracker)
