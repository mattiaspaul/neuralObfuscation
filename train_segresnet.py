#!/usr/bin/env python
# coding: utf-8
import argparse
import monai
from model_nerv import Generator
from monai.networks.nets.segresnet import SegResNet
import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm,trange
import time
torch.set_float32_matmul_precision('high')
import sys
import os
from torchvision.transforms import v2

def main(data_file,gpu_num,N_case,rho,N_train,steps):

    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    print(torch.cuda.get_device_name())
    transforms_train = v2.Compose([
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomErasing(p=0.5),
    ])
    transforms_val = nn.Identity()

    
    args = torch.load('args_nerv.pth')
    args.embed_length = 2*2*8*2
    args.fc_hw_dim = '3_3_16'

    model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim,\
                      expansion=args.expansion, num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True,\
                      reduction=args.reduction, conv_type=args.conv_type, stride_list=args.strides,\
                      sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)
    model.cuda()


    img_label = torch.load(data_file).float()[:,1:].cuda() #only extract label
    img_nerv = torch.zeros_like(img_label)
    H,W = img_label.shape[-2:]
    folds = img_label.shape[0]//N_case
    
    if(N_train==0):
        N_train = img_label.shape[0]

    for sub in range(folds):

        embed_wb = nn.Embedding(N_case,64).cuda()
        state_dicts = torch.load('nerv_ssim_sub'+str(sub)+'.pth')
        embed_wb.load_state_dict(state_dicts[1])
        model.load_state_dict(state_dicts[0])
        with torch.no_grad():
            idx = torch.arange(N_case).cuda()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                code = (torch.eye(N_case)+torch.randn(N_case,N_case)*.01*float(rho)).cuda().mm(embed_wb(idx))
                x = model(code)[-1][:,0]
                img_nerv[idx+sub*N_case,0] = x.view(-1,H,W).float()

    
    net_ = SegResNet(spatial_dims=2,init_filters=24,out_channels=1).cuda()
    net = torch.compile(net_)
    
    iters = 4500
    for repeat in range(steps):
        optimizer = torch.optim.Adam(net.parameters(),lr=2e-3)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1500,0.5)
        run_loss = torch.zeros(iters)
        run_dice = torch.zeros(iters)
        torch.cuda.synchronize()
        batch_size = 32
        grid0 = torch.stack(torch.meshgrid(torch.arange(384),torch.arange(384),indexing='ij'),0).cuda().unsqueeze(0)
        t0 = time.time()
        with tqdm(total=iters, file=sys.stdout) as pbar:
            for i in range(iters):

                idx_case = torch.randperm(N_train)[:batch_size]

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        flips = []
                        if(i%2==1):
                            flips = [3]
                        affine = F.affine_grid(torch.eye(2,3).unsqueeze(0)\
                                    +torch.randn(batch_size,2,3)*.07,(batch_size,1,384,384),align_corners=False).cuda()
                        input = transforms_train(F.grid_sample(img_nerv[idx_case,0].unsqueeze(1).cuda(),\
                                                               affine,align_corners=False)).flip(flips)
                        distmap0 = F.grid_sample(img_nerv[idx_case,:1].cuda(),affine,align_corners=False).flip(flips)
                        distmap1 = 20*F.avg_pool2d(F.avg_pool2d(distmap0,15,stride=1,padding=7),15,stride=1,padding=7)
                    output = net(input)
                    onehot_pred = 20*F.avg_pool2d(F.avg_pool2d(F.sigmoid(output),15,stride=1,padding=7),15,stride=1,padding=7)
                    dice_ = 2*(torch.sigmoid(output)*distmap0).sum([2,3])\
                                /(1e-8+torch.sigmoid(output).sum([2,3])+distmap0.sum([2,3]))
                    if(i<250):
                        loss = nn.MSELoss()(onehot_pred,distmap1)
                    else:
                        loss = 1-dice_.mean()
                        loss += nn.BCEWithLogitsLoss()(output,distmap0)
                run_loss[i] = loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    onehot_pred = (torch.sigmoid(output)>.5).float()
                    dice_ = 2*(onehot_pred*distmap0).sum([2,3])/(1e-8+onehot_pred.sum([2,3])+distmap0.sum([2,3]))
                    run_dice[i] = dice_.mean()

                gpu_used = float(torch.cuda.max_memory_allocated()*1e-9)
                str1 = str('%d'%(100*i/iters))+'%'+' - Dice = '+str('%0.2f'%run_dice[i-10:i-1].mean().mul(100).item())\
                        +' - Loss = '+str('%0.2f'%run_loss[i-10:i-1].mean().mul(100).item())+' - vram='+str('%0.2f'%(gpu_used))
                pbar.set_description(str1)    
                pbar.update(1)

        torch.save(net_.state_dict(),'segresnet_rho_0.0'+str(rho)+'_step_'+str(repeat+1)+'.pth')


        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train_segresnet args')
    parser.add_argument('data_file', help='see create_demo_xray')
    parser.add_argument('gpu_num', help='usually 0-3', default='0')
    parser.add_argument('N_case', help='images per NeRV 32,40,64 (has to be divisor of N)', default='64')
    parser.add_argument('rho', help='rho *0.01 (k-anonymity mixing)', default='4')
    parser.add_argument('N_train', help='how many cases in train (0=all)', default='0')
    parser.add_argument('steps', help='how many training loops of 2500', default='4')
    args = parser.parse_args()

    main(args.data_file,int(args.gpu_num),int(args.N_case),int(args.rho),int(args.N_train),int(args.steps))


