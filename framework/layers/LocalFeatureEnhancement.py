import torch
import torch.nn as nn
from .TCN import TemporalConvNet

class LELayer(nn.Module):
    def __init__(self, patch_len, model_dim, tcn_channels):
        super(LELayer, self).__init__()
        
        self.DenseLayer1 = nn.Linear(patch_len, model_dim)
        
        self.DenseLayer2 = nn.Linear(model_dim, model_dim)

        self.tcn = TemporalConvNet(num_inputs=model_dim, num_channels=tcn_channels, kernel_size=2)

    def forward(self, x):

        main_branch = self.DenseLayer1(x)  # (batch_size, num_nodes, num_patches, model_dim)
        

        parallel_branch = self.DenseLayer1(x)  # (batch_size, num_nodes, num_patches, model_dim)
        
        parallel_branch = parallel_branch.permute(0, 3, 1, 2)  # (batch_size, model_dim, num_nodes, num_patches)
        parallel_branch = parallel_branch.reshape(parallel_branch.size(0), parallel_branch.size(1), -1)  # (batch_size, model_dim, seq_len)

        parallel_branch = self.tcn(parallel_branch)             
        parallel_branch = parallel_branch.permute(0, 2, 1)     
        parallel_branch = self.DenseLayer2(parallel_branch)
        parallel_branch = parallel_branch.permute(0, 2, 1)     
        parallel_branch = self.tcn(parallel_branch)          
        parallel_branch = parallel_branch.permute(0, 2, 1)      


        parallel_branch = parallel_branch.reshape(x.size(0), x.size(1), x.size(2), -1) 
        

        output = main_branch + parallel_branch  # (batch_size, num_nodes, num_patches, model_dim)

        return output