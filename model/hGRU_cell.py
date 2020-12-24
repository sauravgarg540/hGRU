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
        self.u_1 = nn.Conv2d(num_filter,num_filter,1, padding = 0)
        self.u_2 = nn.Conv2d(num_filter,num_filter,1, padding = 0)
        self.b_1 = nn.Parameter(torch.empty(num_filter,1,1))
        self.b_2 = nn.Parameter(torch.empty(num_filter,1,1))
        self.w_gate = torch.nn.Conv2d(num_filter, num_filter, kernel_size, padding = kernel_size//2)
        self.alpha = nn.Parameter(torch.empty((num_filter,1,1)))
        self.gamma = nn.Parameter(torch.empty((num_filter,1,1)))
        self.kappa = nn.Parameter(torch.empty((num_filter,1,1)))
        self.w = nn.Parameter(torch.empty((num_filter,1,1)))
        self.mu= nn.Parameter(torch.empty((num_filter,1,1)))
        self.timesteps = timesteps

        # making W symmetric across the channel.
        # The HxW filter is not symmetric rather it is symmetric across channel decreasing number of parameters to be learned by half.
        nn.init.xavier_uniform_(self.w_gate.weight)
        self.w_gate.weight = nn.Parameter(0.5* (self.w_gate.weight + torch.transpose(self.w_gate.weight, 0,1)))

# W is constrained to have symmetric weights between channels,
# such that the weight Wx0+∆x;y0+∆y;k1;k2 is equal to the weight Wx0+∆x;y0+∆y;k2;k1 where x0 and
# y0 denote the center of the kernel.

        
    def forward(self, x,timesteps = 8):

        h_2 = nn.init.xavier_uniform_(torch.empty_like(x))   
        for i in range(timesteps):

            g_1 = torch.sigmoid((self.u_1(h_2) + self.b_1))
            print(f" g_1 : {g_1.shape}")
            c_1 = self.w_gate(g_1 * h_2)
            print(f" c_1 : {c_1.shape}")
            h_1 = torch.relu(x - c_1 * (self.alpha * h_2 + self.mu))
            print(f" h_1 : {h_1.shape}")
            g_2 = torch.sigmoid((self.u_2(h_1) + self.b_2))
            print(f" g_1 : {g_1.shape}")
            c_2 = self.w_gate(h_1)
            print(f" c_2 : {c_2.shape}")
            h_2_intemmediate = torch.nn.functional.relu(self.kappa * h_1 + self.gamma * c_2 + self.w * h_1 * c_2)
            print(f" h_2_inter : {h_2_intemmediate.shape}")
            # TODO: implement self.n
            h_2 = self.n * ( h_2 * ( 1-g_2) + h_2_intemmediate * g_2)
            print(f" h_2 : {h_2.shape}")
            print("success")

      
    


