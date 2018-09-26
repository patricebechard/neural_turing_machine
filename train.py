import torch
from torch import nn
from torch import optim

def train(model, n_updates=100000, learning_rate=1e-4, print_every=100,
          show_plot=False):
    
    if use_cuda:
        model = model.cuda()
        
# NV - Change BCELoss to BCEWithLogitsLoss for stability. (not computed for lstm_ntm)
    criterion = nn.BCEWithLogitsLoss() 
#    criterion = nn.BCELoss()
    # original paper uses RMSProp with momentum 0.9
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, alpha=0.95)
    
    loss_tracker = []
    cost_per_seq = 0
    
    for update in range(n_updates):
      
        optimizer.zero_grad()
      
        inputs, targets = generate_input_example(batch_size=model.batch_size)
        inputs, targets = Variable(inputs), Variable(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
                
        outputs = model(inputs)
        
        # Going for all the sequence
        #output_len = outputs.shape[0]//2
        
        loss = criterion(outputs, targets)
        cost_per_seq += loss.data[0]

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

            cost_per_seq = 0
    return loss_tracker