import argparse
import logging
import math
import os
import sys
import torch.nn as nn

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import options as option
from models import create_model
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ): 
        mp.set_start_method("spawn", force=True)  
    rank = int(os.environ["RANK"])  
    num_gpus = torch.cuda.device_count()  
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    ) 

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True) 
    
    opt = option.dict_to_nonedict(opt)
    seed = opt["train"]["manual_seed"]
    if args.launcher == "none":  
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist() 
        world_size = (
            torch.distributed.get_world_size()
        )
        rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True
    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=os.path.join(opt["path"]["log"], "tb_logger"))

    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    #### create train and val dataloader
    dataset_ratio = 1000  # enlarge the size of each epoch
    
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt) 
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"]) 
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt) 
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    sde = util.DualPathSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    X_best_psnr = 0.0
    X_best_iter = 0
    Y_best_psnr = 0.0
    Y_best_iter = 0
    error = mp.Value('b', False)

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)

        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            X_LQ, X_GT,Y_LQ,Y_GT = train_data["X_LQ"], train_data["X_GT"],train_data["Y_LQ"], train_data["Y_GT"]

            latent_LQ_X, hidden_LQ_X = model.encode(X_LQ.to(device))
            latent_GT_X, hidden_GT_X = model.encode(X_GT.to(device))
            latent_LQ_Y, hidden_LQ_Y = model.encode(Y_LQ.to(device))
            latent_GT_Y, hidden_GT_Y = model.encode(Y_GT.to(device))

            timesteps, states_X,states_Y = sde.generate_random_states(x0_vis=latent_GT_X,x0_ir=latent_GT_Y,vis_cond=latent_LQ_X,ir_cond=latent_LQ_Y)
            model.feed_data(vis=states_X, vis_cond=latent_LQ_X, ir=states_Y,ir_cond=latent_LQ_Y,gt_vis=latent_GT_X,gt_ir=latent_GT_Y)    
            model.optimize_parameters(current_step, timesteps,hidden_LQ_X, hidden_LQ_Y, sde)
            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])
        
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0:
                if rank == 0:
                    X_avg_psnr = 0.0
                    Y_avg_psnr = 0.0
                    avg_psnr = 0.0
                    idx = 0
                    for idx0, val_data in enumerate(val_loader):
                        
                        X_LQ, X_GT,Y_LQ,Y_GT = val_data["X_LQ"], val_data["X_GT"],val_data["Y_LQ"], val_data["Y_GT"]
                        ori_h_X, ori_w_X = val_data["original_size_X"][0], val_data["original_size_X"][1]
                        ori_h_Y, ori_w_Y = val_data["original_size_Y"][0], val_data["original_size_Y"][1]                        

                        with torch.no_grad():
                            latent_LQ_X, hidden_X = model.encode(X_LQ.to(device))
                            latent_LQ_Y, hidden_Y = model.encode(Y_LQ.to(device))
                            noisy_state_X = sde.noise_state(latent_LQ_X)
                            noisy_state_Y = sde.noise_state(latent_LQ_Y)

                            model.feed_data(noisy_state_X, latent_LQ_X,noisy_state_Y, latent_LQ_Y, X_GT,Y_GT)                            
                            model.val(current_step,sde, hidden_X,hidden_Y,idx=idx0)

                            visuals = model.get_current_visuals()
                            output_X_raw=visuals["Output_VIS"]
                            output_Y_raw=visuals["Output_IR"]
                            output_fuse_raw=visuals["Output_Fuse"]

                            output_X = util.crop_to_original_size(output_X_raw, ori_h_X, ori_w_X)
                            X_LQ_=util.crop_to_original_size(X_LQ, ori_h_X, ori_w_X)
                            output_Y = util.crop_to_original_size(output_Y_raw, ori_h_Y, ori_w_Y)
                            Y_LQ_=util.crop_to_original_size(Y_LQ, ori_h_Y, ori_w_Y)
                            output_fuse=util.crop_to_original_size(output_fuse_raw, ori_h_Y, ori_w_Y)                               
                            output_X_ = util.tensor2img(output_X.squeeze())  # uint8
                            gt_img_X = util.tensor2img(visuals["GT_VIS"].squeeze())  # uint8
                            output_Y_ = util.tensor2img(output_Y.squeeze())  # uint8
                            gt_img_Y = util.tensor2img(visuals["GT_IR"].squeeze())  # uint8
                            output_fuse_ = util.tensor2img(output_fuse.squeeze())
                            LQ_X = util.tensor2img(X_LQ_.squeeze())
                            LQ_Y = util.tensor2img(Y_LQ_.squeeze())

                            vis_filename = "vis_{}_{}.png".format(current_step, idx)
                            out_vis_path =  os.path.join(opt["path"]["val_images"], vis_filename)
                            util.save_img(output_X_, out_vis_path)

                            ir_filename = "ir_{}_{}.png".format(current_step, idx)
                            out_ir_path =  os.path.join(opt["path"]["val_images"], ir_filename)
                            util.save_img(output_Y_, out_ir_path)

                            # vis_gt_filename = "vis_gt_{}.png".format(idx)
                            # vis_gt_path =  os.path.join(opt["path"]["val_images"], vis_gt_filename)
                            # util.save_img(gt_img_X, vis_gt_path)

                            # ir_gt_filename = "ir_gt_{}.png".format(idx)
                            # ir_gt_path =  os.path.join(opt["path"]["val_images"], ir_gt_filename)
                            # util.save_img(gt_img_Y, ir_gt_path)

                            fuse_filename = "fuse_{}_{}.png".format(current_step, idx)
                            out_fuse_path =  os.path.join(opt["path"]["val_images"], fuse_filename)
                            util.save_img(output_fuse_, out_fuse_path)     

                            # x_lq_filename = "vis_lq_{}.png".format(idx)
                            # x_lq_path =  os.path.join(opt["path"]["val_images"], x_lq_filename)
                            # util.save_img(LQ_X, x_lq_path)     

                            # y_lq_filename = "ir_lq_{}.png".format(idx)
                            # y_lq_path =  os.path.join(opt["path"]["val_images"], y_lq_filename)
                            # util.save_img(LQ_Y, y_lq_path)  

                            # calculate PSNR
                            X_avg_psnr += util.calculate_psnr(output_X_, gt_img_X)
                            Y_avg_psnr += util.calculate_psnr(output_Y_, gt_img_Y)
                            idx += 1

                    X_avg_psnr = X_avg_psnr / idx
                    Y_avg_psnr = Y_avg_psnr / idx
                    if X_avg_psnr > X_best_psnr:
                        X_best_psnr = X_avg_psnr
                        X_best_iter = current_step

                    if Y_avg_psnr > Y_best_psnr:
                        Y_best_psnr = Y_avg_psnr
                        Y_best_iter = current_step


                    logger.info("# Validation # X_PSNR: {:.6f}, Best X_PSNR: {:.6f}| Iter: {}".format(X_avg_psnr, X_best_psnr, X_best_iter))
                    logger.info("# Validation # Y_PSNR: {:.6f}, Best Y_PSNR: {:.6f}| Iter: {}".format(Y_avg_psnr, Y_best_psnr, Y_best_iter))

                    logger_val = logging.getLogger("val")  # validation logger
                    logger_val.info(
                        "<epoch:{:3d}, iter:{:8,d}, X_psnr: {:.6f},Y_psnr: {:.6f}".format(
                            epoch, current_step, X_avg_psnr,Y_avg_psnr
                        )
                    )
                    print("<epoch:{:3d}, iter:{:8,d}, X_psnr: {:.6f},Y_psnr: {:.6f}".format(
                            epoch, current_step, X_avg_psnr,Y_avg_psnr
                        ))
                    
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("X_psnr", X_avg_psnr, current_step)
                        tb_logger.add_scalar("Y_psnr", Y_avg_psnr, current_step)

                if dist.is_initialized():
                    dist.barrier()

                torch.cuda.empty_cache()
            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
        tb_logger.close()

if __name__ == "__main__":
    main()
