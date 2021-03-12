import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import initialization as ini

class HgruCell(nn.Module):
    
    def __init__(self, num_filter = 25, kernel_size = 15, timesteps = 8, writer = None):
        super().__init__()
        self.writer = writer
        self.timesteps = timesteps
        self.padding = kernel_size//2
        self.gain_kernel = nn.Conv2d(num_filter,num_filter,1, padding = 0, bias = False)
        self.mix_kernel = nn.Conv2d(num_filter,num_filter,1, padding = 0, bias = False)
        
        # Chronos initialized for bias
        bias_init = -np.log(torch.distributions.Uniform(1, self.timesteps).sample((1,num_filter,1,1)))
        self.gain_bias = nn.Parameter(bias_init)
        self.mix_bias = nn.Parameter(-bias_init)
        self.w_gate = nn.Parameter(torch.empty(num_filter , num_filter , kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.mu= nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.gamma = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.kappa = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.omega = nn.Parameter(torch.empty((1,num_filter,1,1)))
        self.n = nn.Parameter(torch.distributions.Uniform(-2, 2).sample((1,8)))

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
        
    def forward(self, x, timesteps, h_2):
       
        if h_2 == None:
            h_2 = ini.xavier_uniform_(torch.empty(x.size(1),x.size(2),x.size(3),x.size(0)))
            h_2 = h_2.permute(3,0,1,2).cuda()
        grid_img = torchvision.utils.make_grid(h_2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__2_H_2_inside", grid_img, timesteps)
        g1_intermediate = self.gain_kernel(h_2)
        grid_img = torchvision.utils.make_grid(g1_intermediate[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__3_g1_intermediate", grid_img, timesteps)
        g_1_bias = g1_intermediate + self.gain_bias
        grid_img = torchvision.utils.make_grid(g_1_bias[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__4_g_1_bias", grid_img, timesteps)
        g_1 = torch.sigmoid(g1_intermediate + self.gain_bias)
        grid_img = torchvision.utils.make_grid(g_1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__5_g_1", grid_img, timesteps)
        c_1_intermediate = h_2*g_1
        grid_img = torchvision.utils.make_grid(c_1_intermediate[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__6_c_1intermediate_inside_0", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(c_1_intermediate[1].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__6_c_1intermediate_inside_1", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(c_1_intermediate[2].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__6_c_1intermediate_inside_2", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(c_1_intermediate[3].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__6_c_1intermediate_inside_3", grid_img, timesteps)
        c_1 = F.conv2d(h_2*g_1, self.w_gate, padding =self.padding)
        grid_img = torchvision.utils.make_grid(c_1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__7_c1", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(self.w_gate[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__71_W_gate", grid_img, timesteps)
        alpha_h2 = self.alpha*h_2
        alpha_h2_mu = alpha_h2 + self.mu
        alpha_h2_mu_c1 = alpha_h2_mu * c_1
        x_alpha_h2_mu_c1 = x-alpha_h2_mu_c1
        grid_img = torchvision.utils.make_grid(alpha_h2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__8_alpha_h2", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(alpha_h2_mu[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__9_alpha_h2_mu", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(alpha_h2_mu_c1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__10_alpha_h2_mu_c1", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(x_alpha_h2_mu_c1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__11_x_alpha_h2_mu_c1", grid_img, timesteps)
        h_1 = torch.tanh(x - ((self.alpha*h_2 + self.mu)*c_1))
        grid_img = torchvision.utils.make_grid(h_1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__12_h_1", grid_img, timesteps)
        g2_intermediate = self.mix_kernel(h_1)
        grid_img = torchvision.utils.make_grid(g2_intermediate[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__13_g2_intermediate", grid_img, timesteps)
        g2_intermediate_bias = g2_intermediate + self.mix_bias
        grid_img = torchvision.utils.make_grid(g2_intermediate_bias[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__14_g2_intermediate_bias", grid_img, timesteps)
        g_2 = torch.sigmoid(g2_intermediate + self.mix_bias)
        grid_img = torchvision.utils.make_grid(g_2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__15_g_2", grid_img, timesteps)
        c_2 = F.conv2d(h_1, self.w_gate, padding = self.padding)
        grid_img = torchvision.utils.make_grid(c_2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__16_c_2", grid_img, timesteps)
        h_1_c_2 = h_1 * c_2
        grid_img = torchvision.utils.make_grid(h_1_c_2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__17_h_1_c_2", grid_img, timesteps)
        omega_h1_c1 = self.omega * h_1_c_2
        grid_img = torchvision.utils.make_grid(omega_h1_c1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__18_omega_h1_c2", grid_img, timesteps)
        gamma_c2 = self.gamma * c_2
        grid_img = torchvision.utils.make_grid(gamma_c2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__19_gamma_c2", grid_img, timesteps)
        kappa_h1 = self.kappa * h_1
        grid_img = torchvision.utils.make_grid(kappa_h1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__20_kappa_h1", grid_img, timesteps)
        h_2_intermediate_before = omega_h1_c1 + gamma_c2 + omega_h1_c1
        grid_img = torchvision.utils.make_grid(h_2_intermediate_before[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__21_h_2_intermediate_before", grid_img, timesteps)
        h_2_intermediate = torch.tanh((self.kappa * h_1) + (self.gamma * c_2) + (self.omega * (h_1 * c_2)))
        grid_img = torchvision.utils.make_grid(h_2_intermediate[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__22_h_2_intermediate", grid_img, timesteps)
        g2_h2_intermediate = g_2 * h_2_intermediate
        g_2_1 = 1-g_2
        g_2_1_h2 = g_2_1 * h_2
        grid_img = torchvision.utils.make_grid(g2_h2_intermediate[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__23_g2_h2_intermediate", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(g_2_1[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__24_g_2_1", grid_img, timesteps)
        grid_img = torchvision.utils.make_grid(g_2_1_h2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__25_g_2_1_h2", grid_img, timesteps)
        h_2 = ((( 1-g_2) * h_2) + (g_2 * h_2_intermediate))
        grid_img = torchvision.utils.make_grid(h_2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__26_h_2", grid_img, timesteps)
        h_2 = h_2 * self.n[0][timesteps]
        grid_img = torchvision.utils.make_grid(h_2[0].unsqueeze(1), pad_value = 10)
        self.writer.add_image("__27_h_2", grid_img, timesteps)
        return h_2