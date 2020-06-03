import torch
import torch.nn as nn
import numpy as np

from network import Flashback

class FlashbackTrainer():
    ''' Instantiates Flashback module with spatial and temporal weight functions.
    Performs loss computation and prediction.
    '''
    
    def __init__(self, lambda_t, lambda_s):
        ''' The hyper parameters to control spatial and temporal decay.
        '''
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
    
    def __str__(self):
        return 'Use flashback training.'        
    
    def parameters(self):   
        return self.model.parameters()
    
    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        f_t = lambda delta_t, user_len: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t)) # hover cosine + exp decay
        f_s = lambda delta_s, user_len: torch.exp(-(delta_s*self.lambda_s)) # exp decay   
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = Flashback(loc_count, user_count, hidden_size, f_t, f_s, gru_factory).to(device)
    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        ''' takes a batch (users x location sequence)
        then does the prediction and returns a list of user x sequence x location
        describing the probabilities for each location at each position in the sequence.
        t, s are temporal and spatial data related to the location sequence x
        y_t, y_s are temporal and spatial data related to the target sequence y.
        Flashback does not access y_t and y_s for prediction!
        '''
        
        self.model.eval()
        out, h = self.model(x, t, s, y_t, y_s, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h # model outputs logits
    
    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        ''' takes a batch (users x location sequence)
        and corresponding targets in order to compute the training loss '''
        
        self.model.train()
        out, h = self.model(x, t, s, y_t, y_s, h, active_users)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h