config = {
    # -------------------------------------- Generator's setting -------------------------------------- 
    "g_kernel_size": 4,
    "g_stride": 2,
    "g_padding": 1,
    "feature_mapG": 512,
    "g_lr": 0.0002,
    "n_layersG": 4,
    # -------------------------------------- Discriminator's setting -------------------------------------- 
    "d_kernel_size": 4,
    "d_stride": 2,
    "d_padding": 1,
    "feature_mapD": 64,
    "d_lr": 0.0002,
    "n_layersD": 4,
    # -------------------------------------- Model's setting -------------------------------------- 
    "k": 1,
    "batch_size": 128,
    "img_size": 64,
    "n_channels": 3,
    "latent_space_size": 128,
    "epochs": 190,
    "beta_1": 0.5,
    "beta_2": 0.999,
    "device": "cuda",
    "SAVE_PATH": "./saved_models/DCGAN_CelebA.pth",
    "DATA_PATH": "./data"
}