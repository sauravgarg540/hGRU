import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import initialization as ini
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
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
        self.writer.add_figure("__2_H_2_inside", gen_plot(h_2.detach().cpu().numpy()), timesteps)
        g1_intermediate = self.gain_kernel(h_2)
        self.writer.add_figure("__3_g1_intermediate", gen_plot(g1_intermediate.detach().cpu().numpy()), timesteps)
        g_1_bias = g1_intermediate + self.gain_bias
        self.writer.add_figure("__4_g_1_bias", gen_plot(g_1_bias.detach().cpu().numpy()), timesteps)
        g_1 = torch.sigmoid(g1_intermediate + self.gain_bias)
        self.writer.add_figure("__5_g_1", gen_plot(g_1.detach().cpu().numpy()), timesteps)
        c_1_intermediate = h_2*g_1
        self.writer.add_figure("__6_c_1intermediate_inside", gen_plot(c_1_intermediate.detach().cpu().numpy()), timesteps)
        c_1 = F.conv2d(h_2*g_1, self.w_gate, padding =self.padding)
        self.writer.add_figure("__7_c1", gen_plot(c_1.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__71_W_gate", gen_plot(self.w_gate.detach().cpu().numpy()), timesteps)
        alpha_h2 = self.alpha*h_2
        alpha_c_1 = c_1 * self.alpha
        mu_c1 = self.mu * c_1
        alpha_h2_mu = alpha_h2 + self.mu
        alpha_h2_mu_c1 = alpha_h2_mu * c_1
        x_alpha_h2_mu_c1 = x-alpha_h2_mu_c1
        self.writer.add_figure("__8_alpha_h2", gen_plot(alpha_h2.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__9_alpha_h2_mu", gen_plot(alpha_h2_mu.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__8_alpha_c1", gen_plot(alpha_c_1.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__9_mu_c1", gen_plot(mu_c1.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__10_alpha_h2_mu_c1", gen_plot(alpha_h2_mu_c1.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__11_x_alpha_h2_mu_c1", gen_plot(x_alpha_h2_mu_c1.detach().cpu().numpy()), timesteps)
        h_1 = torch.tanh(x - ((self.alpha*h_2 + self.mu)*c_1))
        self.writer.add_figure("__12_h_1", gen_plot(h_1.detach().cpu().numpy()), timesteps)
        g2_intermediate = self.mix_kernel(h_1)
        self.writer.add_figure("__13_g2_intermediate", gen_plot(g2_intermediate.detach().cpu().numpy()), timesteps)
        g2_intermediate_bias = g2_intermediate + self.mix_bias
        self.writer.add_figure("__14_g2_intermediate_bias", gen_plot(g2_intermediate_bias.detach().cpu().numpy()), timesteps)
        g_2 = torch.sigmoid(g2_intermediate + self.mix_bias)
        self.writer.add_figure("__15_g_2", gen_plot(g_2.detach().cpu().numpy()), timesteps)
        c_2 = F.conv2d(h_1, self.w_gate, padding = self.padding)
        self.writer.add_figure("__16_c_2", gen_plot(c_2.detach().cpu().numpy()), timesteps)
        h_1_c_2 = h_1 * c_2
        self.writer.add_figure("__17_h_1_c_2", gen_plot(h_1_c_2.detach().cpu().numpy()), timesteps)
        omega_h1_c1 = self.omega * h_1_c_2
        self.writer.add_figure("__18_omega_h1_c2", gen_plot(omega_h1_c1.detach().cpu().numpy()), timesteps)
        gamma_c2 = self.gamma * c_2
        self.writer.add_figure("__19_gamma_c2", gen_plot(gamma_c2.detach().cpu().numpy()), timesteps)
        kappa_h1 = self.kappa * h_1
        self.writer.add_figure("__20_kappa_h1", gen_plot(kappa_h1.detach().cpu().numpy()), timesteps)
        h_2_intermediate_before = omega_h1_c1 + gamma_c2 + omega_h1_c1
        self.writer.add_figure("__21_h_2_intermediate_before", gen_plot(h_2_intermediate_before.detach().cpu().numpy()), timesteps)
        h_2_intermediate = torch.tanh((self.kappa * h_1) + (self.gamma * c_2) + (self.omega * (h_1 * c_2)))
        self.writer.add_figure("__22_h_2_intermediate", gen_plot(h_2_intermediate.detach().cpu().numpy()), timesteps)
        g2_h2_intermediate = g_2 * h_2_intermediate
        g_2_1 = 1-g_2
        g_2_1_h2 = g_2_1 * h_2
        self.writer.add_figure("__23_g2_h2_intermediate", gen_plot(g2_h2_intermediate.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__24_g_2_1", gen_plot(g_2_1.detach().cpu().numpy()), timesteps)
        self.writer.add_figure("__25_g_2_1_h2", gen_plot(g_2_1_h2.detach().cpu().numpy()), timesteps)
        h_2 = ((( 1-g_2) * h_2) + (g_2 * h_2_intermediate))
        self.writer.add_figure("__26_h_2", gen_plot(h_2.detach().cpu().numpy()), timesteps)
        h_2 = h_2 * self.n[0][timesteps]
        self.writer.add_figure("__27_h_2", gen_plot(h_2.detach().cpu().numpy()), timesteps)
        return h_2


def gen_plot(t, sub = True):
    """Create a pyplot plot and save to buffer."""
    if sub:
        dpi = 10
        _, _, height, width = t.shape
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize= figsize)
        grid = AxesGrid(fig, 111,
                    nrows_ncols=(5, 5),
                    label_mode=1,
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1,
                    share_all=True
                    )
        count = 0
        #looping through all the kernels in each channel
        vmin = np.amin(t[0])
        vmax = np.amax(t[0])
        for ax in grid:
            if(count==t.shape[1]):
                break
            pcm = ax.imshow(t[0, count],cmap = 'plasma', vmin = vmin, vmax = vmax)
            ax.axis('off')
            count +=1
        cbar = ax.cax.colorbar(pcm)
        cbar = grid.cbar_axes[0].colorbar(pcm)
        # plt.show()
    else:
        t = t.sum(axis=1)
        fig = plt.figure()
        plt.imshow(t[0])
    return fig

# def plot_heatmap(t, sub):
#     if sub:
#         dpi = 10
#         b, c, height, width = t.shape
#         # What size does the figure need to be in inches to fit the image?  
#         figsize = width / float(dpi), height / float(dpi)
#         fig,ax = plt.subplots(nrows=5, ncols=5, figsize = figsize)
#         count = 0
#         #looping through all the kernels in each channel
#         for i in range(5):
#             for j in range(5):
#                 pcm = ax[i,j].imshow(t[0, count], cmap = 'plasma')
#                 ax[i, j].axis('off')
#                 count +=1
#         plt.subplots_adjust(left=0.12,
#                     bottom=0.412, 
#                     right=0.54, 
#                     top=0.871, 
#                     wspace=0.02, 
#                     hspace=0.02)
#         fig.colorbar(pcm,  ax = ax[:,:], location = 'right')
#         # plt.show()
#     else:
#         t = t.sum(axis=1)
#         fig = plt.figure()
#         plt.imshow(t[0])
#     return fig