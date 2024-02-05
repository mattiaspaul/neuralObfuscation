#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn.functional as F
import imageio.v3 as iio
import torch.nn as nn
import numpy as np

import monai
from model_nerv import Generator
from monai.networks.nets.segresnet import SegResNet
import os

def rand_nerv_img(model):
    rho = 0.08
    img_nerv = torch.zeros(138,1,360,360).float().cuda()
    N_case = 46
    for sub in range(3):
        embed_wb = nn.Embedding(N_case,64).cuda()
        state_dicts = torch.load('nerv_ssim_sub'+str(sub)+'.pth')
        embed_wb.load_state_dict(state_dicts[1])
        model.load_state_dict(state_dicts[0])
        with torch.no_grad():
            idx = (torch.arange(N_case)).cuda()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                code = (torch.eye(N_case)+torch.randn(N_case,N_case)*rho).cuda().mm(embed_wb(idx))
                x = model(code)[-1][:,0]
                img_nerv[idx+sub*N_case,0] = x.view(-1,360,360).float()

    return img_nerv
def main():
    directory = 'output_png/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    args = torch.load('args_nerv.pth')
    args.embed_length = 2*2*8*2
    args.fc_hw_dim = '3_3_16'

    model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion,
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)
    model.cuda()

    img_nerv = rand_nerv_img(model)
    img_label_in = torch.load('img_label_demo.pth').cuda()

    state_dict = torch.load('segresnet_rho_0.08_step_1.pth')
    net = SegResNet(spatial_dims=2,init_filters=24,out_channels=1).cuda()
    net.load_state_dict(state_dict)
    net.eval()

    batch_size = 16
    idx_case = torch.arange(batch_size)+122

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            affine = F.affine_grid(torch.eye(2,3).unsqueeze(0)+torch.randn(batch_size,2,3)*.0,\
                                   (batch_size,1,360,360),align_corners=False).cuda()

            input = (F.grid_sample(img_nerv[idx_case,0].unsqueeze(1).cuda(),affine,align_corners=False))#.flip(flips)
            distmap0 = F.grid_sample(img_label_in[idx_case,1:].cuda(),affine,align_corners=False)#.flip(flips)
            output = .5*net(input)
            input = (F.grid_sample(img_nerv[idx_case,0].unsqueeze(1).cuda(),affine,align_corners=False)).flip(-1)
            output += .5*net(input).flip(-1)

            onehot_pred = (torch.sigmoid(output)>.5).float()
            dice_ = 2*(onehot_pred*distmap0).sum([2,3])/(1e-8+onehot_pred.sum([2,3])+distmap0.sum([2,3]))
    print('Validation Dice %0.2f'%(100*dice_.mean().item())+'%')

    cmap = torch.tensor([[0.1216, 0.4667, 0.7059],[1.0000, 0.4980, 0.0549],[0.1725, 0.6275, 0.1725]])
    for c in range(0,16):
        gray0 = torch.clamp(img_label_in[122+c,0].data.float().cpu()*1.,0,1)
        gray1 = torch.clamp(img_nerv[122+c,0].data.float().cpu()*1.,0,1)
        label0 = img_label_in[122+c,1].data.float().cpu()
        label1 = (onehot_pred[c,0]).data.float().cpu()
        white = torch.ones(360,20,3)
        cdata = cmap[0:1].view(1,1,3); output_color = label0.unsqueeze(-1)*cdata
        alpha = torch.clamp(.5 + 0.5*(1-label0),0.0,1.0)
        overlay0 = (gray0*alpha).unsqueeze(2) + output_color*(1.0-alpha.unsqueeze(2))
        cdata = cmap[2:3].view(1,1,3); output_color = label1.unsqueeze(-1)*cdata
        alpha = torch.clamp(.5 + 0.5*(1-label1),0.0,1.0)
        overlay1 = (gray1*alpha).unsqueeze(2) + output_color*(1.0-alpha.unsqueeze(2))
        cat012 = torch.cat((overlay0,white,overlay1),1)
        iio.imwrite(directory+'example_segout_'+str(c)+'.png',torch.clamp(cat012.mul(255),0,255).byte().numpy())

if __name__ == "__main__":
    main()
