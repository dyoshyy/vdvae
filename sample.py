import os
from data import set_up_data
import numpy as np
from vae import VAE
import argparse
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
import torch
from PIL import Image


def restore_params(model, path, local_rank, mpi_size, map_ddp=True, map_cpu=False):
    state_dict = torch.load(path, map_location='cpu' if map_cpu else None, weights_only=True)
    if map_ddp:
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    model.load_state_dict(state_dict)

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_save_dirs(H)
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    return H

def load_vaes(H):
    vae = VAE(H)
    if H.restore_path:
        restore_params(vae, H.restore_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)

    ema_vae = VAE(H)
    if H.restore_ema_path:
        restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
    else:
        ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(False)

    ema_vae = ema_vae.cuda(H.local_rank)
    return ema_vae


def sampling(H, ema_vae):
    decoder_output = ema_vae.forward_uncond_samples(10, t=1.0)
    save_path = "generated"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(decoder_output)):
        img = decoder_output[i]
        img = Image.fromarray(np.uint8(img))
        img.save(f"{save_path}/sample_{i}.png")
    

def main():
    H = set_up_hyperparams()
    H, _, _, _= set_up_data(H)
    ema_vae = load_vaes(H)
    sampling(H, ema_vae)


if __name__ == "__main__":
    main()
