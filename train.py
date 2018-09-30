import torch
from torch import nn
from torch import optim
from utils import *

def train(model, num_updates=100000, learning_rate=1e-4, momentum=0.9,
          print_every=100, show_plot=False, save_model=False):
        
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, alpha=0.95)
    
    loss_tracker = []
    cost_per_seq = 0
    
    for update in range(num_updates):
      
        optimizer.zero_grad()
      
        inputs, targets = generate_input_example(batch_size=model.batch_size)

        inputs = inputs.to(device)
        targets = targets.to(device)
                
        outputs = model(inputs)
        
        # Going for all the sequence
        #output_len = outputs.shape[0]//2
        
        loss = criterion(outputs, targets)
        cost_per_seq += loss.item()

        loss.backward()
        parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)
        optimizer.step()

        if update % print_every == 0:
          
            if update != 0:
                cost_per_seq /= print_every
            loss_tracker.append(cost_per_seq)
            print("Number of sequences processed : %d ----- Cost per sequence(bits) : %.3f" % (update*model.batch_size, loss_tracker[-1]))
            
            if show_plot:
                show_last_example(inputs, outputs, targets)

            if save_model and update != 0:
                torch.save(model.state_dict(), 'pretrained.pt')

            cost_per_seq = 0
    return loss_tracker