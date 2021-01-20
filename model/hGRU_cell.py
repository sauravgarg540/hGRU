import torch
import torch.nn as nn
import numpy as np

class HgruCell(nn.Module):
    """This is a hGRU cell. Fig 2.  of the paper is implemented here

    Args:
        nn ([type]): [description]
    """
    
    def __init__(self, num_filter = 25, kernel_size = 15, timesteps = 8):
        super().__init__()
        self.timesteps = timesteps
        self.u_1 = nn.Conv2d(num_filter,num_filter,1, padding = 0, bias = False)#gain_kernel
        self.u_2 = nn.Conv2d(num_filter,num_filter,1, padding = 0, bias = False)#mix_kernel
        
        # Chronos initialized for bias
        self.b_1 = nn.Parameter(-np.log(torch.FloatTensor(1,num_filter,1,1).uniform_(1, self.timesteps - 1)))#gain_bias
        self.b_2 = nn.Parameter(np.log(torch.FloatTensor(1,num_filter,1,1).uniform_(1, self.timesteps - 1)))#mix_bias

        self.w_gate_inh = torch.nn.Conv2d(num_filter, num_filter, kernel_size, padding = kernel_size//2, bias = False)#horizontal_kernel
        self.w_gate_exc = torch.nn.Conv2d(num_filter, num_filter, kernel_size, padding = kernel_size//2, bias = False)
        
        self.alpha = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.mu= nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.gamma = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.kappa = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.omega = nn.Parameter(torch.empty((1,num_filter,1,1)))
        # check shapes of alpha, mu, gamma, kappa
        # learnable T-dimensional parameter
        self.n = nn.Parameter(torch.FloatTensor(self.timesteps,1,1).uniform_(-0.5, 0.5))

        # making W symmetric across the channel.
        # The HxW filter is not symmetric rather it is symmetric across channel decreasing number of parameters to be learned by half.
        nn.init.xavier_uniform_(self.w_gate_exc.weight)
        self.w_gate_exc.weight = nn.Parameter(0.5* (self.w_gate_exc.weight + torch.transpose(self.w_gate_exc.weight, 0,1)))
        nn.init.xavier_uniform_(self.w_gate_inh.weight)
        self.w_gate_inh.weight = nn.Parameter(0.5* (self.w_gate_inh.weight + torch.transpose(self.w_gate_inh.weight, 0,1)))

        nn.init.xavier_uniform_(self.u_1.weight)
        # nn.init.constant_(self.u_1.bias, 0)
        nn.init.xavier_uniform_(self.u_2.weight)
        # nn.init.constant_(self.u_2.bias, 0)
        nn.init.xavier_uniform_(self.alpha)
        nn.init.xavier_uniform_(self.gamma)
        nn.init.xavier_uniform_(self.kappa)
        nn.init.xavier_uniform_(self.omega)
        nn.init.xavier_uniform_(self.mu)
        
    def forward(self, x, timesteps):
        h_2 = None
        for i in range(timesteps):
            if (i==0):
                h_2 = nn.init.xavier_uniform_(torch.empty_like(x))
            g1_intermediate = self.u_1(h_2)
            g_1 = torch.sigmoid(g1_intermediate + self.b_1)
            c_1 = self.w_gate_inh(h_2*g_1)
            h_1 = torch.tanh(x - ((self.alpha * h_2 + self.mu)*c_1))
            g2_intermediate = self.u_2(h_1) 
            g_2 = torch.sigmoid(g2_intermediate + self.b_2)
            c_2 = self.w_gate_exc(h_1)
            h_2_intemmediate = torch.tanh((self.kappa * (h_1 + (self.gamma * c_2))) + (self.omega * (h_1 * (self.gamma * c_2))))
            # h_2 = self.n[i] * (( 1-g_2) * h_2 +  g_2 * h_2_intemmediate)
            h_2 = ((( 1-g_2) * h_2_intemmediate) + (g_2 * h_2))* self.n[i]

        return h_2

      
    


