import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
lpips_fn = lpips.LPIPS(net='alex').to(device)

sde = util.DualPathSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
T=opt["sde"]["T"]

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["X_psnr"] = []
    test_results["X_ssim"] = []
    test_results["X_psnr_y"] = []
    test_results["X_ssim_y"] = []
    test_results["X_lpips"] = []

    test_results["Y_psnr"] = []
    test_results["Y_ssim"] = []
    test_results["Y_psnr_y"] = []
    test_results["Y_ssim_y"] = []
    test_results["Y_lpips"] = []    
    test_results["Fuse_entropy"]=[]
    test_results["Fuse_std"]=[]
    test_times = []

    evaluate_against_LQ = (
        True if test_loader.dataset.opt.get("evaluate_against_LQ") else False
    )

    for i, test_data in enumerate(test_loader):
        X_single_img_psnr = []
        X_single_img_ssim = []
        X_single_img_psnr_y = []
        X_single_img_ssim_y = []
        Y_single_img_psnr = []
        Y_single_img_ssim = []
        Y_single_img_psnr_y = []
        Y_single_img_ssim_y = []

        X_need_GT = False if test_loader.dataset.opt["dataroot_GT_X"] is None  else True
        Y_need_GT = False if test_loader.dataset.opt["dataroot_GT_Y"] is None else True
        X_img_path = test_data["X_GT_path"][0] if X_need_GT else test_data["X_LQ_path"][0]
        Y_img_path = test_data["Y_GT_path"][0] if Y_need_GT else test_data["Y_LQ_path"][0]
        X_img_name = os.path.splitext(os.path.basename(X_img_path))[0]
        Y_img_name = os.path.splitext(os.path.basename(Y_img_path))[0]        

        X_LQ, X_GT,Y_LQ,Y_GT = test_data["X_LQ"], test_data["X_GT"],test_data["Y_LQ"], test_data["Y_GT"]
        ori_h_X, ori_w_X = test_data["original_size_X"][0], test_data["original_size_X"][1]
        ori_h_Y, ori_w_Y = test_data["original_size_Y"][0], test_data["original_size_Y"][1]

        with torch.no_grad():
            latent_LQ_X, hidden_X = model.encode(X_LQ.to(device))
            latent_LQ_Y, hidden_Y = model.encode(Y_LQ.to(device))
            latent_GT_X, _ = model.encode(X_GT.to(device))
            latent_GT_Y, _ = model.encode(Y_GT.to(device))

            noisy_state_X = sde.noise_state(latent_LQ_X)
            noisy_state_Y = sde.noise_state(latent_LQ_Y)

            model.feed_data(noisy_state_X, latent_LQ_X,noisy_state_Y, latent_LQ_Y, latent_GT_X,latent_GT_Y)
            tic = time.time()

            model.test(i,sde, hidden_X,hidden_Y)

            toc = time.time()
            test_times.append(toc - tic)

        visuals = model.get_current_visuals(need_GT=True)
        X_SR_img = visuals["Output_VIS"]
        Y_SR_img = visuals["Output_IR"]
        Fuse = visuals["Output_Fuse"]

        scale = opt["scale"] if opt.get("scale") else 1
        X_SR_img = util.crop_to_original_size(X_SR_img, ori_h_X * scale, ori_w_X * scale)
        X_LQ_=util.crop_to_original_size(X_LQ, ori_h_X * scale, ori_w_X * scale)
        Y_SR_img = util.crop_to_original_size(Y_SR_img, ori_h_Y * scale, ori_w_Y * scale)
        Y_LQ_=util.crop_to_original_size(Y_LQ, ori_h_Y * scale, ori_w_Y * scale)
        Fuse_=util.crop_to_original_size(Fuse, ori_h_Y * scale, ori_w_Y * scale)
        save_HQ = (
            test_loader.dataset.opt.get("save_HQ")
            if test_loader.dataset.opt.get("save_HQ") is not None
            else True
        )
        save_LQ = (
            test_loader.dataset.opt.get("save_LQ")
            if test_loader.dataset.opt.get("save_LQ") is not None
            else True
        )

        X_output = util.tensor2img(X_SR_img.squeeze())  # uint8
        Y_output = util.tensor2img(Y_SR_img.squeeze())  # uint8
        X_LQ_ = util.tensor2img(X_LQ_.squeeze())  # uint8
        Y_LQ_ = util.tensor2img(Y_LQ_.squeeze())  # uint8
        X_GT_ = util.tensor2img(X_GT.squeeze())  # uint8
        Y_GT_ = util.tensor2img(Y_GT.squeeze())  # uint8
        Fuse_XY=util.tensor2img(Fuse_.squeeze()) 

        fuse_entropy, fuse_std = util.calculate_entropy_and_std(Fuse_)
        test_results["Fuse_entropy"].append(fuse_entropy)
        test_results["Fuse_std"].append(fuse_std)
        logger.info(f"Fuse image [{i} - {Y_img_name}] - Entropy: {fuse_entropy:.4f}, Std: {fuse_std:.4f}")
        
        suffix_X = opt["suffix_X"]
        suffix_Y = opt["suffix_Y"]
        suffix_Fuse = opt["suffix_Fuse"]
        if suffix_X:
            X_save_img_path = os.path.join(dataset_dir, X_img_name + suffix_X + ".png")
            
        else:
            X_save_img_path = os.path.join(dataset_dir, X_img_name + "_X.png")
            
        if suffix_Y: 
            Y_save_img_path = os.path.join(dataset_dir, Y_img_name + suffix_Y + ".png")  
        else:
            Y_save_img_path = os.path.join(dataset_dir, Y_img_name + "_Y.png")
        util.save_img(X_output, X_save_img_path)
        util.save_img(Y_output, Y_save_img_path)
        # torch.cuda.empty_cache()
        if save_LQ:
            X_LQ_img_path = os.path.join(dataset_dir, X_img_name + "_LQ_X.png")
            Y_LQ_img_path = os.path.join(dataset_dir, Y_img_name + "_LQ_Y.png")
            util.save_img(X_LQ_, X_LQ_img_path)
            util.save_img(Y_LQ_, Y_LQ_img_path)        
        if save_HQ:
            X_GT_img_path = os.path.join(dataset_dir, X_img_name + "_HQ_X.png")
            Y_GT_img_path = os.path.join(dataset_dir, Y_img_name + "_HQ_Y.png")
            util.save_img(X_GT_, X_GT_img_path)
            util.save_img(Y_GT_, Y_GT_img_path)       
        if suffix_Fuse:
            Fuse_img_path = os.path.join(dataset_dir, Y_img_name + suffix_Fuse + ".png")
        else:
            Fuse_img_path = os.path.join(dataset_dir, Y_img_name + ".png")
        util.save_img(Fuse_XY, Fuse_img_path)

        if X_need_GT:
            X_gt_img = X_GT_ / 255.0
            X_sr_img = X_output / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else 0
            if crop_border == 0:
                X_cropped_sr_img = X_sr_img
                X_cropped_gt_img = X_gt_img
            else:
                X_cropped_sr_img = X_sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                X_cropped_gt_img = X_gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            if evaluate_against_LQ:
                X_psnr = util.calculate_psnr(X_cropped_sr_img * 255, X_cropped_gt_img * 255)
                X_ssim = util.calculate_ssim(X_cropped_sr_img * 255, X_cropped_gt_img * 255)
                X_lp_score = lpips_fn(
                    X_GT.to(device) * 2 - 1, X_SR_img.to(device) * 2 - 1
                ).squeeze().item()

                test_results["X_psnr"].append(X_psnr)
                test_results["X_ssim"].append(X_ssim)
                test_results["X_lpips"].append(X_lp_score)

            if len(X_gt_img.shape) == 3:
                if X_gt_img.shape[2] == 3:  # RGB image
                    X_sr_img_y = bgr2ycbcr(X_sr_img, only_y=True)
                    X_gt_img_y = bgr2ycbcr(X_gt_img, only_y=True)
                    if crop_border == 0:
                        X_cropped_sr_img_y = X_sr_img_y
                        X_cropped_gt_img_y = X_gt_img_y
                    else:
                        X_cropped_sr_img_y = X_sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        X_cropped_gt_img_y = X_gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]

                    if evaluate_against_LQ:
                        X_psnr_y = util.calculate_psnr(
                            X_cropped_sr_img_y * 255, X_cropped_gt_img_y * 255
                        )
                        X_ssim_y = util.calculate_ssim(
                            X_cropped_sr_img_y * 255, X_cropped_gt_img_y * 255
                        )
                        test_results["X_psnr_y"].append(X_psnr_y)
                        test_results["X_ssim_y"].append(X_ssim_y)
                        logger.info(
                            "X_img{:3d}:{:15s} - X_PSNR: {:.6f} dB; X_SSIM: {:.6f}; X_LPIPS: {:.6f}; X_PSNR_Y: {:.6f} dB; X_SSIM_Y: {:.6f}.".format(
                                i, X_img_name, X_psnr, X_ssim, X_lp_score, X_psnr_y, X_ssim_y
                            )
                        )
            else:
                if evaluate_against_LQ:
                    logger.info(
                        "X_img:{:15s} - X_PSNR: {:.6f} dB; X_SSIM: {:.6f}.".format(
                            X_img_name, X_psnr, X_ssim
                        )
                    )

                    test_results["X_psnr_y"].append(X_psnr)
                    test_results["X_ssim_y"].append(X_ssim)
        else:
            logger.info(X_img_name)

        if Y_need_GT:
            Y_gt_img = Y_GT_ / 255.0
            Y_sr_img = Y_output / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else 0
            if crop_border == 0:
                Y_cropped_sr_img = Y_sr_img
                Y_cropped_gt_img = Y_gt_img
            else:
                Y_cropped_sr_img = Y_sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                Y_cropped_gt_img = Y_gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
            if evaluate_against_LQ:
                Y_psnr = util.calculate_psnr(Y_cropped_sr_img * 255, Y_cropped_gt_img * 255)
                Y_ssim = util.calculate_ssim(Y_cropped_sr_img * 255, Y_cropped_gt_img * 255)
                Y_lp_score = lpips_fn(
                    Y_GT.to(device) * 2 - 1, Y_SR_img.to(device) * 2 - 1).squeeze().item()

                test_results["Y_psnr"].append(Y_psnr)
                test_results["Y_ssim"].append(Y_ssim)
                test_results["Y_lpips"].append(Y_lp_score)

            if len(Y_gt_img.shape) == 3:
                if Y_gt_img.shape[2] == 3:  # RGB image
                    Y_sr_img_y = bgr2ycbcr(Y_sr_img, only_y=True)
                    Y_gt_img_y = bgr2ycbcr(Y_gt_img, only_y=True)
                    if crop_border == 0:
                        Y_cropped_sr_img_y = Y_sr_img_y
                        Y_cropped_gt_img_y = Y_gt_img_y
                    else:
                        Y_cropped_sr_img_y = Y_sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        Y_cropped_gt_img_y = Y_gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    if evaluate_against_LQ:
                        Y_psnr_y = util.calculate_psnr(
                            Y_cropped_sr_img_y * 255, Y_cropped_gt_img_y * 255
                        )
                        Y_ssim_y = util.calculate_ssim(
                            Y_cropped_sr_img_y * 255, Y_cropped_gt_img_y * 255
                        )

                        test_results["Y_psnr_y"].append(Y_psnr_y)
                        test_results["Y_ssim_y"].append(Y_ssim_y)

                        logger.info(
                            "Y_img{:3d}:{:15s} - Y_PSNR: {:.6f} dB; Y_SSIM: {:.6f}; Y_LPIPS: {:.6f}; Y_PSNR_Y: {:.6f} dB; Y_SSIM_Y: {:.6f}.".format(
                                i, Y_img_name, Y_psnr, Y_ssim, Y_lp_score, Y_psnr_y, Y_ssim_y
                            )
                        )
            else:
                if evaluate_against_LQ:
                    logger.info(
                        "Y_img:{:15s} - Y_PSNR: {:.6f} dB; Y_SSIM: {:.6f}.".format(
                            Y_img_name, Y_psnr, Y_ssim
                        )
                    )

                    test_results["Y_psnr_y"].append(Y_psnr)
                    test_results["Y_ssim_y"].append(Y_ssim)
        else:
            logger.info(Y_img_name)
          
    if evaluate_against_LQ:
        X_ave_lpips = sum(test_results["X_lpips"]) / len(test_results["X_lpips"])
        X_ave_psnr = sum(test_results["X_psnr"]) / len(test_results["X_psnr"])
        X_ave_ssim = sum(test_results["X_ssim"]) / len(test_results["X_ssim"])

        Y_ave_lpips = sum(test_results["Y_lpips"]) / len(test_results["Y_lpips"])
        Y_ave_psnr = sum(test_results["Y_psnr"]) / len(test_results["Y_psnr"])
        Y_ave_ssim = sum(test_results["Y_ssim"]) / len(test_results["Y_ssim"])

        ave_lpips = (sum(test_results["X_lpips"])+sum(test_results["Y_lpips"])) / (len(test_results["X_lpips"])+len(test_results["Y_lpips"]))
        ave_psnr = (sum(test_results["X_psnr"])+sum(test_results["Y_psnr"])) / (len(test_results["X_psnr"])+len(test_results["Y_psnr"]))
        ave_ssim = (sum(test_results["X_ssim"])+sum(test_results["Y_ssim"])) / (len(test_results["X_ssim"])+len(test_results["Y_ssim"]))
    ave_fuse_entropy = sum(test_results["Fuse_entropy"]) / len(test_results["Fuse_entropy"])
    ave_fuse_std = sum(test_results["Fuse_std"]) / len(test_results["Fuse_std"])

    logger.info("----Average Fused Image Metrics----\n\tEntropy: {:.4f}; Std: {:.4f}".format(ave_fuse_entropy, ave_fuse_std))
    if evaluate_against_LQ:
        logger.info(
            "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
                test_set_name, ave_psnr, ave_ssim
            )
        )
        logger.info(
            "----Average X_PSNR/X_SSIM results for {}----\n\tX_PSNR: {:.6f} dB; X_SSIM: {:.6f}\n".format(
                test_set_name, X_ave_psnr, X_ave_ssim
            )
        )
        logger.info(
            "----Average Y_PSNR/SSIM results for {}----\n\tY_PSNR: {:.6f} dB; Y_SSIM: {:.6f}\n".format(
                test_set_name, Y_ave_psnr, Y_ave_ssim
            )
        )

    if evaluate_against_LQ:
        if test_results["X_psnr_y"] and test_results["X_ssim_y"] and test_results["Y_psnr_y"] and test_results["Y_ssim_y"]:
            ave_psnr_y = (sum(test_results["X_psnr_y"])+sum(test_results["Y_psnr_y"])) /(len(test_results["X_psnr_y"])+len(test_results["Y_psnr_y"]))
            ave_ssim_y = (sum(test_results["X_ssim_y"])+sum(test_results["Y_ssim_y"])) / (len(test_results["X_ssim_y"])+len(test_results["Y_ssim_y"]))
            logger.info(
                "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                    ave_psnr_y, ave_ssim_y
                )
            )

        if test_results["X_psnr_y"] and test_results["X_ssim_y"]:
            X_ave_psnr_y = sum(test_results["X_psnr_y"]) / len(test_results["X_psnr_y"])
            X_ave_ssim_y = sum(test_results["X_ssim_y"]) / len(test_results["X_ssim_y"])
            logger.info(
                "----For X ,  Y channel, average PSNR/SSIM----\n\X_tPSNR_Y: {:.6f} dB; X_SSIM_Y: {:.6f}\n".format(
                    X_ave_psnr_y, X_ave_ssim_y
                )
            )
        if test_results["Y_psnr_y"] and test_results["Y_ssim_y"]:
            Y_ave_psnr_y = sum(test_results["Y_psnr_y"]) / len(test_results["Y_psnr_y"])
            Y_ave_ssim_y = sum(test_results["Y_ssim_y"]) / len(test_results["Y_ssim_y"])
            logger.info(
                "----For Y , Y channel, average PSNR/SSIM----\n\tY_PSNR_Y: {:.6f} dB; Y_SSIM_Y: {:.6f}\n".format(
                    Y_ave_psnr_y, Y_ave_ssim_y
                )
            )

        logger.info(
                "----average LPIPS\t: {:.6f}\n".format(ave_lpips)
            )
        logger.info(
                "----average X_LPIPS\t: {:.6f}\n".format(X_ave_lpips)
            )
        logger.info(
                "----average Y_LPIPS\t: {:.6f}\n".format(Y_ave_lpips)
            )
          
    print(f"average test time: {np.mean(test_times):.4f}")

