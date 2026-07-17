import os
import random
import sys
import math
import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

try:
    sys.path.append(".")
    import data.util as util
except ImportError:
    pass

class LQGTXYDataset(data.Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.X_LQ_paths, self.X_GT_paths = None, None
        self.Y_LQ_paths, self.Y_GT_paths = None, None

        self.X_LQ_env, self.X_GT_env = None, None  # environment for lmdb
        self.Y_LQ_env, self.Y_GT_env = None, None  # environment for lmdb
        
        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.X_LQ_paths, self.X_LQ_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ_X"]
            )
            self.Y_LQ_paths, self.Y_LQ_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ_Y"]
            )
            if opt["dataroot_GT_X"]:
                self.X_GT_paths, self.X_GT_sizes = util.get_image_paths(
                    opt["data_type"], opt["dataroot_GT_X"]
                )
            else:
                self.X_GT_paths, self.X_GT_sizes = self.X_LQ_paths, self.X_LQ_sizes
            
            if opt["dataroot_GT_Y"]:
                self.Y_GT_paths, self.Y_GT_sizes = util.get_image_paths(
                    opt["data_type"], opt["dataroot_GT_Y"]
                )
            else:
                self.Y_GT_paths, self.Y_GT_sizes = self.Y_LQ_paths, self.Y_LQ_sizes
        elif opt["data_type"] == "img":
            self.X_LQ_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ_X"]
            )
            self.Y_LQ_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ_Y"]
            )
            if opt["dataroot_GT_X"]:
                self.X_GT_paths = util.get_image_paths(
                    opt["data_type"], opt["dataroot_GT_X"]
                )
            else:
                self.X_GT_paths = self.X_LQ_paths
            if opt["dataroot_GT_Y"]:
                self.Y_GT_paths = util.get_image_paths(
                    opt["data_type"], opt["dataroot_GT_Y"]
                )
            else:
                self.Y_GT_paths = self.Y_LQ_paths
        else:
            print("Error: data_type is not matched in Dataset")

        if self.X_LQ_paths and self.X_GT_paths:
            assert len(self.X_LQ_paths) == len(
                self.X_GT_paths
            ), "XGT and XLQ datasets have different number of images - {}, {}.".format(
                len(self.X_LQ_paths), len(self.X_GT_paths)
            )

        if self.Y_LQ_paths and self.Y_GT_paths:
            assert len(self.Y_LQ_paths) == len(
                self.Y_GT_paths
            ), "YGT and YLQ datasets have different number of images - {}, {}.".format(
                len(self.Y_LQ_paths), len(self.Y_GT_paths)
            )

        if self.Y_LQ_paths and self.X_LQ_paths:
            assert len(self.Y_LQ_paths) == len(
                self.X_LQ_paths
            ), "X and Y datasets have different number of images - {}, {}.".format(
                len(self.X_LQ_paths), len(self.Y_LQ_paths)
            )


    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.X_LQ_env = lmdb.open(
            self.opt["dataroot_LQ_X"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.Y_LQ_env = lmdb.open(
            self.opt["dataroot_LQ_Y"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if self.opt["dataroot_GT_X"]:
            self.X_GT_env = lmdb.open(
                self.opt["dataroot_GT_X"],
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            self.X_GT_env = self.X_LQ_env

        if self.opt["dataroot_GT_Y"]:
            self.Y_GT_env = lmdb.open(
                self.opt["dataroot_GT_Y"],
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        else:
            self.Y_GT_env = self.Y_LQ_env

    def __getitem__(self, index):
        
        if self.opt["data_type"] == "lmdb":
            if (self.X_GT_env is None) or (self.X_LQ_env is None) or (self.Y_GT_env is None) or (self.Y_LQ_env is None):
                self._init_lmdb()

        X_GT_path, X_LQ_path,Y_GT_path, Y_LQ_path= None, None,None,None

        GT_size = self.opt["GT_size"]
        LQ_size = self.opt["LQ_size"]

        X_GT_path = self.X_GT_paths[index]
        Y_GT_path = self.Y_GT_paths[index]

        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.X_GT_sizes[index].split("_")]
        else:
            resolution = None
        X_img_GT = util.read_img(
            self.X_GT_env, X_GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]
        if X_img_GT.shape[2]==1:
            X_img_GT = cv2.merge([X_img_GT, X_img_GT, X_img_GT])

        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.Y_GT_sizes[index].split("_")]
        else:
            resolution = None
        Y_img_GT = util.read_img(
            self.Y_GT_env, Y_GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]
        if Y_img_GT.shape[2]==1:
            Y_img_GT = cv2.merge([Y_img_GT, Y_img_GT, Y_img_GT])       

        X_LQ_path = self.X_LQ_paths[index]
        Y_LQ_path = self.Y_LQ_paths[index]

        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.X_LQ_sizes[index].split("_")]
        else:
            resolution = None
        X_img_LQ = util.read_img(self.X_LQ_env, X_LQ_path, resolution)
        if X_img_LQ.shape[2] == 1:
            X_img_LQ = cv2.merge([X_img_LQ, X_img_LQ, X_img_LQ])

        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.Y_LQ_sizes[index].split("_")]
        else:
            resolution = None
        Y_img_LQ = util.read_img(self.Y_LQ_env, Y_LQ_path, resolution)
        if Y_img_LQ.shape[2] == 1:
            Y_img_LQ = cv2.merge([Y_img_LQ, Y_img_LQ, Y_img_LQ])

        if self.opt["phase"] == "train":
            H, W, C = X_img_LQ.shape
            H_,W_,C_ =Y_img_LQ.shape
            assert H==H_ and W==W_ ,"X size and Y size do not match"
            assert LQ_size == GT_size, "GT size does not match LQ size"

            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            X_img_LQ = X_img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]
            Y_img_LQ = Y_img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]

            rnd_h_GT, rnd_w_GT = rnd_h, rnd_w
            X_img_GT = X_img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]
            Y_img_GT = Y_img_GT[
                rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
            ]

            # augmentation - flip, rotate
            X_img_LQ, X_img_GT ,Y_img_LQ,Y_img_GT= util.augment(
                [X_img_LQ, X_img_GT,Y_img_LQ,Y_img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
                self.opt["use_swap"],
            )

        elif LQ_size is not None:
            H, W, C = X_img_LQ.shape
            H_,W_,C_ =Y_img_LQ.shape

            assert H==H_ and W==W_ ,"X size and Y size do not match"
            assert LQ_size == GT_size, "GT size does not match LQ size"

            if LQ_size < H and LQ_size < W:

                rnd_h = H // 2 - LQ_size//2
                rnd_w = W // 2 - LQ_size//2
                X_img_LQ = X_img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]
                Y_img_LQ = Y_img_LQ[rnd_h : rnd_h + LQ_size, rnd_w : rnd_w + LQ_size, :]

                rnd_h_GT, rnd_w_GT = rnd_h, rnd_w
                X_img_GT = X_img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]
                Y_img_GT = Y_img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]

        original_size_LQ_X = (X_img_LQ.shape[0], X_img_LQ.shape[1])
        original_size_LQ_Y = (Y_img_LQ.shape[0], Y_img_LQ.shape[1])
        pad_h_X, pad_w_X = 0, 0
        pad_h_Y, pad_w_Y = 0, 0
   
        if (self.opt["phase"] == "test"):

            h, w = X_img_LQ.shape[:2]
            pad_h_X = 50
            pad_w_X = 50
            X_img_LQ = np.pad(X_img_LQ,
                            ((0, pad_h_X), (0, pad_w_X), (0, 0)),
                            mode='edge')

            h, w = Y_img_LQ.shape[:2]
            pad_h_Y = 50
            pad_w_Y = 50
            Y_img_LQ = np.pad(Y_img_LQ,
                            ((0, pad_h_Y), (0, pad_w_Y), (0, 0)),
                            mode='edge')

        # change color space if necessary
        if self.opt["color"]:
            H, W, C = X_img_LQ.shape
            X_img_LQ = util.channel_convert(C, self.opt["color"], [X_img_LQ])[
                0
            ]  
            X_img_GT = util.channel_convert(X_img_GT.shape[2], self.opt["color"], [X_img_GT])[
                0
            ]
            H_, W_, C_ = Y_img_LQ.shape
            Y_img_LQ = util.channel_convert(C_, self.opt["color"], [Y_img_LQ])[
                0
            ]
            Y_img_GT = util.channel_convert(Y_img_GT.shape[2], self.opt["color"], [Y_img_GT])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if X_img_GT.shape[2] == 3:
            X_img_GT = X_img_GT[:, :, [2, 1, 0]]
            X_img_LQ = X_img_LQ[:, :, [2, 1, 0]]
        if Y_img_GT.shape[2] == 3:
            Y_img_GT = Y_img_GT[:, :, [2, 1, 0]]
            Y_img_LQ = Y_img_LQ[:, :, [2, 1, 0]]

        X_img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(X_img_GT, (2, 0, 1)))
        ).float()
        X_img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(X_img_LQ, (2, 0, 1)))
        ).float()
        Y_img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(Y_img_GT, (2, 0, 1)))
        ).float()
        Y_img_LQ = torch.from_numpy(
            np.ascontiguousarray(np.transpose(Y_img_LQ, (2, 0, 1)))
        ).float()

        return {"X_LQ": X_img_LQ, "X_GT": X_img_GT, "X_LQ_path": X_LQ_path, "X_GT_path": X_GT_path,
                "Y_LQ": Y_img_LQ, "Y_GT": Y_img_GT, "Y_LQ_path": Y_LQ_path, "Y_GT_path": Y_GT_path,
                "original_size_X": original_size_LQ_X,"original_size_Y": original_size_LQ_Y
                }

    def __len__(self):
        return len(self.X_LQ_paths)
