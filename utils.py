import torch
from torch import

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_input_example(sequence_length=None, batch_size=1):
    
    #length of binary vectors fed to models
    vector_length = 8
    
    #length of sequence of binary vectors
    if sequence_length is None:
        # generate random sequence length between 1 and 20
        sequence_length = np.random.randint(1, 21)            
        
    data = np.random.randint(2, size=(sequence_length, batch_size, vector_length+1))

    # making sure all data has no EOS (no 1 at 9th position)
    data[:, :, -1] = 0.0
    
    padding = np.zeros((sequence_length, batch_size, vector_length+1))
    
    delimiter = np.zeros((1, batch_size, vector_length+1))
    delimiter[:, :, -1] = 1.0    

    inputs = np.concatenate((data, delimiter, padding))
    
    delimiter = np.zeros((1, batch_size, vector_length+1))
    targets = np.concatenate((padding, delimiter, data)) 
    
    #convert to torch tensors
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs, targets

def show_last_example(inputs, outputs, targets):
    
# NV - Adjust because BCEWLL
    outputs = F.sigmoid(outputs)
    
    inputs = inputs[:,0].data.cpu().numpy()
    outputs = outputs[:,0].data.cpu().numpy()
    targets = targets[:,0].data.cpu().numpy()
    
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    ax1.matshow(inputs.T, aspect='auto')
    ax2.matshow(targets.T, aspect='auto')
    ax3.matshow(outputs.T, aspect='auto')
    
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    
    plt.show()
    plt.clf()

def visualise_read_write(model):
    plt.clf()
  
    inputs, targets = generate_input_example(sequence_length=20, batch_size=1)
    inputs, targets = Variable(inputs), Variable(targets)
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()

    outputs = model(inputs)
# NV - Adjust because BCEWLL
    outputs = F.sigmoid(outputs)
    
    inputs = inputs.data.squeeze().numpy().T
    outputs = outputs.data.squeeze().numpy().T
    read_vecs = model.kept_read_vectors.squeeze().numpy().T
    add_vecs = model.kept_write_vectors.squeeze().numpy().T
    read_weights = model.kept_read_weights.squeeze().numpy().T
    write_weights = model.kept_write_weights.squeeze().numpy().T
    
    n_shown_mem_loc = 40

    fig = plt.figure(figsize=(10, 8.8)) 
    gs = gridspec.GridSpec(3, 2, height_ratios=[9, 20, n_shown_mem_loc]) 
    gs.update(wspace=0.02, hspace=0.02) # set the spacing between axes. 
    ax0 = plt.subplot(gs[0, 0])
    ax0.matshow(inputs)
    ax0.set_title('Inputs', size=14)
    ax0.axis('off')
    ax1 = plt.subplot(gs[0, 1])
    ax1.matshow(outputs)
    ax1.set_title('Outputs', size=14)
    ax1.axis('off')
    ax2 = plt.subplot(gs[1, 0])
    ax2.matshow(add_vecs)
    ax2.text(-3, 8, 'Adds', rotation=90, size=14)
    ax2.axis('off')
    ax3 = plt.subplot(gs[1, 1])
    ax3.matshow(read_vecs)
    ax3.text(40.75, 8, 'Reads', rotation=270, size=14)
    ax3.axis('off')
    ax4 = plt.subplot(gs[2, 0])
    ax4.matshow(write_weights[:n_shown_mem_loc])
    ax4.text(-3, 28, r'Location $\longrightarrow$', rotation=90, size=14)
    ax4.text(0, 42, r'Time $\longrightarrow$', size=14)
    ax4.set_title('Write Weightings', y=-0.2)
    ax4.axis('off')
    ax5 = plt.subplot(gs[2, 1])
    ax5.matshow(read_weights[:n_shown_mem_loc])
    ax5.text(0, 42, r'Time $\longrightarrow$', size=14)
    ax5.set_title('Read Weightings', y=-0.2)
    ax5.axis('off')

    plt.savefig('read_write_memory.png')
    plt.show()