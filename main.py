import torch
import numpy as np
import json

from model import Vanilla_LSTM, NTM
from train import train
from utils import *

if __name__ == "__main__":

	with open('params.json') as json_file:
		params = json.load(json_file)


	model = NTM(controller_type=params['controller_type'],
				hidden_size=params['hidden_size'],
				num_layers=params['num_layers'],
				num_memory_loc=params['num_memory_loc'],
				memory_loc_size=params['memory_loc_size'],
				shift_range=params['shift_range'],
				batch_size=params['batch_size']).to(device)

	loss_tracker = train(model, 
						 num_updates=params['num_updates'], 
						 learning_rate=params['learning_rate'],
						 momentum=params['momentum'],
						 print_every=params['print_every'], 
						 show_plot=params['show_plot'],
						 save_model=params['save_model'])

	np.savetxt('loss_ntm_%s.csv' % (params['controller_type']), loss_tracker)
