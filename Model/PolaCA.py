from Model.pola_swin_3D_dualinput import PolaSwinTransformer
import torch.nn as nn
import Model.configs_PolaReg as configs

class PolaCA(nn.Module):
    def __init__(self, config, pretrain_img_size, dim_diy, attn_mix):
        super(PolaCA, self).__init__()
        self.transformer = PolaSwinTransformer(pretrain_img_size=pretrain_img_size,
                                            patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           norm_layer=nn.LayerNorm,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           frozen_stages=-1,
                                           use_checkpoint=config.use_checkpoint,    
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           alpha=config.alpha,
                                           kernel_size=config.kernel_size,
                                           attn_type=config.attn_type,
                                           dim_diy=dim_diy,
                                           attn_mix=attn_mix,
                                           )

    def forward(self, x, y):
        moving_fea_cross = self.transformer(x, y)
        return moving_fea_cross
