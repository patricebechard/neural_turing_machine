import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import device

class NTM(nn.Module):
    def __init__(self, controller_type, input_size=9, hidden_size=100, num_layers=1,
                 num_memory_loc=128, memory_loc_size=20, shift_range=1, batch_size=1):
        super(NTM, self).__init__()

        self.controller_type = controller_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = (num_memory_loc, memory_loc_size)
        self.shift_range = shift_range
        self.batch_size = batch_size

        if controller_type not in ['lstm', 'ffnn']:
            raise Exception("Controller type '%s' not supported. "
                            "Please choose between 'lstm' and 'ffnn'." % controller_type)

        # creating controller, read head and write head
        self.controller = Controller(controller_type, input_size + memory_loc_size, 
                                     hidden_size, batch_size=batch_size)
        self.write_head = Head(hidden_size, num_memory_loc, 
                               memory_loc_size, batch_size=batch_size)
        self.read_head = Head(hidden_size, num_memory_loc, 
                              memory_loc_size, batch_size=batch_size)

        self.fc_erase = nn.Linear(hidden_size, memory_loc_size)
        self.fc_add = nn.Linear(hidden_size, memory_loc_size)
        self.fc_out = nn.Linear(memory_loc_size, input_size)

        self.memory0 = nn.Parameter(torch.randn(1, self.memory_size[0],
                                                self.memory_size[1]) * 0.05)
        self.write_weight0 = nn.Parameter(torch.randn(1, self.memory_size[0]) * 0.05)
        self.read_weight0 = nn.Parameter(torch.randn(1, self.memory_size[0]) * 0.05)

        self.read0 = nn.Parameter(torch.randn(1, self.memory_size[1]) * 0.05)

        self.init_parameters()

    def init_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_erase.weight)
        nn.init.constant_(self.fc_erase.bias, 0)

        nn.init.xavier_uniform_(self.fc_add.weight)
        nn.init.normal_(self.fc_add.bias, 0)

        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x):

        self.ntm_out = None
        
        #used to store read/write vectors/weights to visualize
        self.kept_read_vectors = None
        self.kept_write_vectors = None
        self.kept_read_weights = None
        self.kept_write_weights = None

        self.memory = self._init_memory()
        self.prev_write_weight, self.prev_read_weight = self._init_weight()
        self.read = self._init_read()

        if self.controller_type == "lstm":
            self.controller.hidden = self.controller._init_hidden()

        #Take a slice of length batch_size x input (original size of input + M)
        for input in x:

            input = torch.cat((input, self.read), dim=1)
            ht = self.controller(input)

            self.write_weight = self.write_head(ht, self.memory, self.prev_write_weight)
            self.read_weight = self.read_head(ht, self.memory, self.prev_read_weight)
            
            if self.kept_read_weights is None:
                self.kept_read_weights = self.read_weight.unsqueeze(0).data
            else:
                self.kept_read_weights = torch.cat((self.kept_read_weights, 
                                                   self.read_weight.unsqueeze(0).data))
            if self.kept_write_weights is None:
                self.kept_write_weights = self.write_weight.unsqueeze(0).data
            else:
                self.kept_write_weights = torch.cat((self.kept_write_weights,
                                                    self.write_weight.unsqueeze(0).data))

            self.prev_write_weight = self.write_weight
            self.prev_read_weight = self.read_weight


            self.erase = torch.sigmoid(self.fc_erase(ht))
            self.add = torch.sigmoid(self.fc_add(ht))

            self._write()
            self._read()
            
            if self.kept_read_vectors is None:
                self.kept_read_vectors = self.read.unsqueeze(0).data
            else:
                self.kept_read_vectors = torch.cat((self.kept_read_vectors, 
                                                   self.read.unsqueeze(0).data))
            if self.kept_write_vectors is None:
                self.kept_write_vectors = self.add.unsqueeze(0).data
            else:
                self.kept_write_vectors = torch.cat((self.kept_write_vectors,
                                                        self.add.unsqueeze(0).data))

            out = self.fc_out(self.read).unsqueeze(0)
            
        #NV - Retrait du sigmoid pour stabilité et erreur NAN avec BCEWithLogitLoss
        #    out = torch.sigmoid(out)

            if self.ntm_out is None:
                self.ntm_out = out
            else:
                self.ntm_out = torch.cat((self.ntm_out, out))

        return self.ntm_out

    def _read(self):
        self.read = torch.matmul(self.read_weight.unsqueeze(1), self.memory).view(self.batch_size, -1)

    def _write(self):
        erase_tensor = torch.matmul(self.write_weight.unsqueeze(-1), self.erase.unsqueeze(1))
        add_tensor = torch.matmul(self.write_weight.unsqueeze(-1), self.add.unsqueeze(1))
        self.memory = self.memory * (1 - erase_tensor) + add_tensor

    def _init_memory(self):
        memory = self.memory0.clone().repeat(self.batch_size, 1, 1).to(device)

        return memory

    def _init_weight(self):
        read_weight = self.read_weight0.clone().repeat(self.batch_size, 1).to(device)
        write_weight = self.write_weight0.clone().repeat(self.batch_size, 1).to(device)

        #print torch.sum(read_weight)

        read_weight = F.softmax(read_weight, 1)
        write_weight = F.softmax(write_weight, 1)

        return write_weight, read_weight

    def _init_read(self):
        return self.read0.clone().repeat(self.batch_size, 1).to(device)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

##################################################################################################
#
# Each head has 5 linear layers:
#   For the key parameter
#   For the beta parameter
#   For the blending parameter
#   For the shift parameter
#   For the gamma parameter
# It also has a refernce to the memory and the previous weight since it needs them to
# calculate the addressing
# The output of its forward() method is a normalized new weight
#
##################################################################################################
class Head(nn.Module):
    def __init__(self, hidden_size, weight_size, key_size, shift_range=1, batch_size=1):
        super(Head, self).__init__()

        self.hidden_size = hidden_size
        self.weight_size = weight_size
        self.key_size = key_size
        self.shift_range = shift_range
        self.batch_size = batch_size

        self.fc_key = nn.Linear(hidden_size, key_size)
        self.fc_beta = nn.Linear(hidden_size, 1)
        self.fc_blending = nn.Linear(hidden_size, 1)
        self.fc_shift = nn.Linear(hidden_size, 2 * shift_range + 1)
        self.fc_gamma = nn.Linear(hidden_size, 1)

        self.init_parameters()

    def init_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_normal_(self.fc_key.weight)
        nn.init.constant_(self.fc_key.bias, 0)

        nn.init.xavier_uniform_(self.fc_beta.weight)
        nn.init.constant_(self.fc_beta.bias, 0)

        nn.init.xavier_uniform_(self.fc_blending.weight)
        nn.init.constant_(self.fc_blending.bias, 0)

        nn.init.xavier_uniform_(self.fc_shift.weight)
        nn.init.constant_(self.fc_shift.bias, 0)

        nn.init.xavier_normal_(self.fc_gamma.weight)
        nn.init.constant_(self.fc_gamma.bias, 0)

    def forward(self, x, memory, prev_weight):
        self.memory = memory
        self.prev_weight = prev_weight
        self.key = F.relu(self.fc_key(x))
        self.beta = F.softplus(self.fc_beta(x))
        self.blending = torch.sigmoid(self.fc_blending(x))
        self.shift = F.softmax(self.fc_shift(x), 1)
        self.gamma = F.relu(self.fc_gamma(x)) + 1

        self._addressing()

        return self.weight

    def _addressing(self):
        self._content_addressing()
        self._interpolation()
        self._convolutional_shift()
        self._sharpening()

    def _content_addressing(self):
        self.weight = F.softmax(self.beta * F.cosine_similarity(self.key.unsqueeze(1), self.memory, dim=-1), 1)

    def _interpolation(self):
        self.weight = self.blending * self.weight + (1 - self.blending) * self.prev_weight

    def _convolutional_shift(self):
        tmp = torch.zeros_like(self.weight)
        # expanding weight vector for same convolution
        self.weight = torch.cat((self.weight[:, -1:], self.weight, self.weight[:, :1]), dim=1)
        for b in range(self.batch_size):
            tmp[b] = F.conv1d(self.weight[b].view(1, 1, -1), self.shift[b].view(1, 1, -1))
        self.weight = tmp

    def _sharpening(self):
        self.weight = self.weight ** self.gamma
        self.weight = torch.div(self.weight, torch.sum(self.weight, dim=1).unsqueeze(1))

#################################################################################################################
#
# The controller uses an LSTM or an MLP
#
#################################################################################################################
class Controller(nn.Module):
    def __init__(self, controller_type, input_size, hidden_size, batch_size=1,
                 num_layers=1):
        super(Controller, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.controller_type = controller_type
        self.batch_size = batch_size
        self.num_layers = num_layers

        if self.controller_type == "lstm":
            self.controller = nn.LSTM(input_size, hidden_size, num_layers)
            self.hidden0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
            self.cell0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.05)
            self.init_parameters()
        elif self.controller_type == "ffnn":
            self.controller = nn.Linear(input_size, hidden_size)
            nn.init.xavier_normal_(self.controller.weight)


    def forward(self, x):
        if self.controller_type == "lstm":
            x, self.hidden = self.controller(x.view(1, self.batch_size, -1), self.hidden)
            x = x.view(self.batch_size, -1)
        elif self.controller_type == "ffnn":
            x = self.controller(x)
        return torch.tanh(x)

    # Crucial for learning: the hidden and cell state of the LSTM at the start of each mini-batch must be learned
    def _init_hidden(self):
        hidden_state = self.hidden0.clone().repeat(1, self.batch_size, 1).to(device)
        cell_state = self.cell0.clone().repeat(1, self.batch_size, 1).to(device)

        return hidden_state, cell_state

    def init_parameters(self):
        for param in self.controller.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                stdev = 5 / (np.sqrt(self.input_size + self.hidden_size))
                nn.init.uniform_(param, -stdev, stdev)

class Vanilla_LSTM(nn.Module):
    
    def __init__(self, input_size=9, hidden_size=100, output_size=9, num_layers=1, batch_size=1):
        super(Vanilla_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.hidden0 = self._init_hidden()
        
    def forward(self, x):#, hidden):
      
       # self.hidden = hidden
        
        output, self.hidden = self.lstm(x.squeeze(), self.hidden0)
    #NV - out for BCEloss
        #output = torch.sigmoid(self.fc(output))
        output = self.fc(output)
        return output
        
    
    def _init_hidden(self):
        
        hidden_state = (torch.randn(self.num_layers, 
                                   self.batch_size, 
                                   self.hidden_size) * 0.05).to(device)
        cell_state = (torch.randn(self.num_layers, 
                                 self.batch_size, 
                                 self.hidden_size) * 0.05).to(device)
            
        return (hidden_state, cell_state)
    
    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)