"""
SAITS model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}



Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


from layers import *
from modeling_utils import masked_mae_cal
import torch 
import torch.nn as nn


class SAITS(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        **kwargs
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs["input_with_mask"]
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]
        self.MIT = kwargs["MIT"]
        self.device = kwargs["device"]

        if kwargs["param_sharing_strategy"] == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )
        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(8*d_model,2)
        

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]

        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)
        enc_output=enc_output.view(enc_output.size(0),-1)
        X= self.reduce_dim_z(enc_output)
    
        mu,sigma=X[:,0],X[:,1]
        sigma = F.softplus(sigma)
        return mu,sigma

    def forward(self, inputs, stage):
        X, masks,idx = inputs["X_holdout"], inputs["missing_mask"],inputs["idx"]
        reconstruction_loss = 0
        mu,sigma= self.impute(inputs)
        reconstruction_loss += masked_mae_cal(mu,sigma,X,idx)
        return {
            "reconstruction_loss": reconstruction_loss,
        }
