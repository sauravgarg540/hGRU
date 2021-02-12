import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import initialization as ini

class HgruCell(nn.Module):
    
    def __init__(self, num_filter = 25, kernel_size = 15, timesteps = 8):
        super().__init__()
        self.timesteps = timesteps
        self.padding = kernel_size//2
        self.gain_kernel = nn.Conv2d(num_filter,num_filter,1, padding = 0, bias = False)
        self.mix_kernel = nn.Conv2d(num_filter,num_filter,1, padding = 0, bias = False)
        
        # Chronos initialized for bias
        bias_init = -np.log(torch.FloatTensor(1,num_filter,1,1).uniform_(1, self.timesteps - 1))
        self.gain_bias = nn.Parameter(bias_init)
        self.mix_bias = nn.Parameter(-bias_init)
        self.w_gate = nn.Parameter(torch.empty(num_filter , num_filter , kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.mu= nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.gamma = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.kappa = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.omega = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.n = nn.Parameter(torch.FloatTensor(self.timesteps).uniform_(-0.5, 0.5))

        # making W symmetric across the channel.
        # The HxW filter is not symmetric rather it is symmetric across channel decreasing number of parameters to be learned by half.
        ini.xavier_uniform_(self.w_gate)
        self.w_gate = nn.Parameter(0.5* (self.w_gate + torch.transpose(self.w_gate, 0,1)))
        
        ini.xavier_uniform_(self.gain_kernel.weight)
        ini.xavier_uniform_(self.mix_kernel.weight)
        ini.xavier_uniform_(self.alpha)
        ini.xavier_uniform_(self.mu)
        ini.xavier_uniform_(self.gamma)
        ini.xavier_uniform_(self.kappa)
        ini.xavier_uniform_(self.omega)
        
    def forward(self, x, timesteps):
        h_2 = None
        for i in range(timesteps):
            if (i==0):
                h_2 = ini.xavier_uniform_(torch.empty(25,75,75,32))
                h_2 = h_2.permute(3,0,1,2).cuda()
            g1_intermediate = self.gain_kernel(h_2)
            g_1 = torch.sigmoid(g1_intermediate + self.gain_bias)
            c_1 = F.conv2d(h_2*g_1, self.w_gate, padding =self.padding)
            h_1 = torch.tanh(x - ((self.alpha*h_2 + self.mu)*c_1))
            g2_intermediate = self.mix_kernel(h_1) 
            g_2 = torch.sigmoid(g2_intermediate + self.mix_bias)
            c_2 = F.conv2d(h_1, self.w_gate, padding = self.padding)
            h_2_intermediate = torch.tanh((self.kappa * (h_1 + (self.gamma * c_2))) + (self.omega * (h_1 * (self.gamma * c_2))))
            h_2 = ((( 1-g_2) * h_2_intermediate) + (g_2 * h_2))* self.n[i]
        return h_2