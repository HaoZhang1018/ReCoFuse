import logging

import torch

from models import modules as M

logger = logging.getLogger("base")

# Generator
def define_G(opt):
    opt_net = opt["network_G"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    print(f"which_model: {which_model}")
    netG = getattr(M, which_model)(**setting)
    return netG

# Latent model
def define_L(opt):
    opt_net = opt["network_L"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netL = getattr(M, which_model)(**setting)
    return netL
#fusion
def define_F(opt):
    opt_net = opt["network_F"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netC = getattr(M, which_model)(**setting)
    return netC

