import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()

##### Gradient Loss by select the most significant pixel gradient 
class L_gradient(nn.Module):
    def __init__(self):
        super(L_gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse,non_seg_mask=None):    
        if  non_seg_mask==None:
            gradient_A_x, gradient_A_y = self.gradient(image_A)
        
            gradient_B_x, gradient_B_y = self.gradient(image_B)
    
            gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)

            loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
            return loss
        else:
            gradient_A_x, gradient_A_y = self.gradient(image_A)
        
            gradient_B_x, gradient_B_y = self.gradient(image_B)
    
            gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)

            loss = F.l1_loss(gradient_fuse_x[non_seg_mask], torch.max(gradient_A_x[non_seg_mask], gradient_B_x[non_seg_mask])) + F.l1_loss(gradient_fuse_y[non_seg_mask], torch.max(gradient_A_y[non_seg_mask], gradient_B_y[non_seg_mask]))
            return loss      

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class fusion_base_loss(nn.Module):
    def __init__(self):
        super(fusion_base_loss, self).__init__()
        self.loss_func_Grad = L_gradient()      

    def forward(self, image_A, image_B, image_fused,int_ratio=1, grad_ratio=6, color_ratio=8):

        image_A_y,image_A_cb,image_A_cr = self.rgb_to_y(image_A)
        image_B_y,image_B_cb,image_B_cr = self.rgb_to_y(image_B)
        image_fuse_y,image_fuse_cb,image_fuse_cr = self.rgb_to_y(image_fused)       
        loss_int = int_ratio * F.l1_loss(image_fuse_y, torch.max(image_A_y,image_B_y))
        loss_grad = grad_ratio * self.loss_func_Grad(image_A_y, image_B_y, image_fuse_y)
        loss_color = color_ratio * (F.l1_loss(image_fuse_cb, image_A_cb)+F.l1_loss(image_fuse_cr, image_A_cr))
        total_loss = (loss_int+loss_grad+loss_color)    
        
        return total_loss, loss_int,loss_grad, loss_color

    def rgb_to_y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)
        return y, cb, cr