def test(model, seq_length=range(10, 101, 10), n_inputs=20):
    
    avg_loss_array = []
    avg_loss_pushed_array = []
  #  criterion = nn.BCEWithLogitsLoss() 

    for length in seq_length:
        
        length_loss = 0
        length_loss_pushed = 0
        
        for i in range(n_inputs):
            
            inputs, targets = generate_input_example(sequence_length=length)
            inputs, targets = Variable(inputs), Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
        # Take Sigmoid because model outputs without (BCEWLL)
            outputs_sig = F.sigmoid(outputs)   
            
        # Error without pushing the values
            err = torch.mean( torch.abs(outputs_sig[-length:] - targets[-length:]) )
            length_loss += err
     #       print("err no push", err)   
            
#        # Error pushing the output values to 0 or 1
#            outputs_pushed = torch.round(outputs_sig) 
#            err_pushed = torch.mean( torch.abs(outputs_pushed - targets) )
#            length_loss_pushed += err_pushed
#            print("err pushED", err_pushed)                        
            
       #print outputs pushed to 0 or 1     
           # outputs_pushed = torch.round(outputs_sig)
           # show_last_example(inputs, outputs_pushed, targets)
            
        length_loss /= n_inputs
   #     length_loss_pushed /= n_inputs 
        
        avg_loss_array.append(length_loss.data)
#        avg_loss_pushed_array.append(length_loss_pushed.data)
            
    return avg_loss_array#, avg_loss_pushed_array