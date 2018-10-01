import torch
from utils import *

def eval(model, example, show_plot=True):

    inputs = example.to(device)
    output = model(inputs)

    if show_plot:
        show_last_example(inputs, output, output)