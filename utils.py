import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
    inputs = torch.from_numpy(inputs).float().to(device)
    targets = torch.from_numpy(targets).float().to(device)

    return inputs, targets

def show_last_example(inputs, outputs, targets):
    
# NV - Adjust because BCEWLL
    outputs = torch.sigmoid(outputs)
    
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

    
def plot_training_curves(rng, lstm_ntm_loss, ffnn_ntm_loss, lstm_loss):

    plt.figure(figsize=(12, 8))
    plt.plot(rng, lstm_loss, 'b', label="LSTM")
    plt.plot(rng, lstm_ntm_loss, 'g', label="NTM with LSTM Controller")
    plt.plot(rng, ffnn_ntm_loss, 'r', label="NTM with FeedForward Controller")
    
    plt.xlabel("Number of sequences")
    plt.ylabel("BCE Loss")
  #  plt.axis([0, len(rng)*10, 0, 0.05])
    plt.legend(fancybox=True)
    plt.show()
    
def visualise_read_write(model):
    plt.clf()
  
    inputs, targets = generate_input_example(sequence_length=20, batch_size=1)

    outputs = model(inputs)
# NV - Adjust because BCEWLL
    outputs = torch.sigmoid(outputs)
    
    inputs = inputs.data.squeeze().cpu().numpy().T
    outputs = outputs.data.squeeze().cpu().numpy().T
    read_vecs = model.kept_read_vectors.squeeze().cpu().numpy().T
    add_vecs = model.kept_write_vectors.squeeze().cpu().numpy().T
    read_weights = model.kept_read_weights.squeeze().cpu().numpy().T
    write_weights = model.kept_write_weights.squeeze().cpu().numpy().T
    
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
    
def show_generalization(model):
    
    # showing models results on inputs of size 10, 20, 30, 50, 120
    
    sequence_lengths = [10, 20, 30, 50, 120]
    
    inputs = []
    outputs = []
    
    for i, seq_len in enumerate(sequence_lengths):
        
        input, _ = generate_input_example(sequence_length=seq_len)
        inputs.append(input[:seq_len])
        
        output = torch.sigmoid(model(input.to(device)).to('cpu')[seq_len+1:])
        outputs.append(output)


    # creating plot similar to figure 4 from Graves et. al. 2014
    
    fig = plt.figure(figsize=(20, 4))
    
    gs1 = gridspec.GridSpec(4, 11)
    gs1.update(left=0.05, right=0.48, wspace=0.33)

    # targets for 10, 20, 30, 50
    ax1 = plt.subplot(gs1[0, 0:1])
    ax1.matshow(inputs[0].data.squeeze().cpu().numpy().T)
    ax1.axis('off')
    ax2 = plt.subplot(gs1[0, 1:3])
    ax2.matshow(inputs[1].data.squeeze().cpu().numpy().T)
    ax2.axis('off')
    ax3 = plt.subplot(gs1[0, 3:6])
    ax3.matshow(inputs[2].data.squeeze().cpu().numpy().T)
    ax3.axis('off')
    ax4 = plt.subplot(gs1[0, 6:11])
    ax4.matshow(inputs[3].data.squeeze().cpu().numpy().T)
    ax4.axis('off')
    
    # outputs for 10, 20, 30, 50
    ax5 = plt.subplot(gs1[1, 0:1])
    ax5.matshow(outputs[0].data.squeeze().cpu().numpy().T)
    ax5.axis('off')
    ax6 = plt.subplot(gs1[1, 1:3])
    ax6.matshow(outputs[1].data.squeeze().cpu().numpy().T)
    ax6.axis('off')
    ax7 = plt.subplot(gs1[1, 3:6])
    ax7.matshow(outputs[2].data.squeeze().cpu().numpy().T)
    ax7.axis('off')
    ax8 = plt.subplot(gs1[1, 6:11])
    ax8.matshow(outputs[3].data.squeeze().cpu().numpy().T)
    ax8.axis('off')
    
    # targets and outputs for 120
    ax9 = plt.subplot(gs1[2, 0:11])
    ax9.matshow(inputs[4].data.squeeze().cpu().numpy().T)
    ax9.axis('off')
    ax10 = plt.subplot(gs1[3, 0:11])
    ax10.matshow(outputs[4].data.squeeze().cpu().numpy().T)
    ax10.axis('off')
    
    
    ax1.text(-15, 4.5, 'Targets', size=14)
    ax5.text(-15, 4.5, 'Outputs', size=14)
    ax9.text(-13, 4.5, 'Targets', size=14)
    ax10.text(-13, 4.5, 'Outputs', size=14)
    ax10.text(0, 12, r'Time $\longrightarrow$', size=14)


    plt.show()