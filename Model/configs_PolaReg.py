import ml_collections

def get_PolaReg_LPBA40_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96

    config.num_heads = (8, 8, 8, 8)

    config.window_size = (5, 6, 5)   
    
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = [0]
    config.reg_head_chan = 16  
    config.img_size = (160, 192, 160) 
    
    config.alpha = 15
    config.kernel_size=3
    config.depths = [1]
    config.attn_type='LLLL'

    config.attn_mix=['P','P','P','P']

    return config

def get_PolaReg_OASIS_config():
    config = ml_collections.ConfigDict()

    config.if_transskip     = True
    config.if_convskip      = True
    config.patch_size       = 4
    config.in_chans         = 1
    config.embed_dim        = 96
    
    config.num_heads        = (8, 8, 8, 8)
    config.mlp_ratio        = 4
    config.pat_merg_rf      = 4
    config.qkv_bias         = False
    config.drop_rate        = 0
    config.drop_path_rate   = 0.3
    config.ape              = False
    config.spe              = False
    config.patch_norm       = True
    config.use_checkpoint   = False
    config.out_indices      = (0, 1, 2, 3)
    config.reg_head_chan    = 16

    
    config.alpha            = 15
    config.kernel_size      = 3

    config.depths           = [1]
    config.attn_type        = 'LLLL'
    config.attn_mix=['P','P','P','P']


    config.window_size      = (5, 7, 6)
    config.img_size         = (160, 224, 192)

    return config

def get_PolaReg_IXI_config():
    config = ml_collections.ConfigDict()

    config.if_transskip     = True
    config.if_convskip      = True
    config.patch_size       = 4
    config.in_chans         = 1
    config.embed_dim        = 96
    config.depths           = (1, 2, 4, 2)
    config.num_heads        = (4, 4, 8, 8)
    config.mlp_ratio        = 4
    config.pat_merg_rf      = 4
    config.qkv_bias         = False
    config.drop_rate        = 0
    config.drop_path_rate   = 0.3
    config.ape              = False
    config.spe              = False
    config.patch_norm       = True
    config.use_checkpoint   = False
    config.out_indices      = (0, 1, 2, 3)
    config.reg_head_chan    = 16
    config.alpha            = 15
    config.kernel_size      = 5
    config.attn_type        = 'LLLL'
    config.attn_type_C      = 'L'

    config.window_size      = (5, 6, 7)
    config.img_size         = (160, 192, 224)

    return config