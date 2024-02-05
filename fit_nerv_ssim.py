import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm,trange
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
torch.set_float32_matmul_precision('high')
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from model_nerv import Generator

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def main(data_file,gpu_num,N_case):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    print(torch.cuda.get_device_name(),gpu_num)
    args = torch.load('args_nerv.pth')
    args.embed_length = 2*2*8*2
    args.fc_hw_dim = '3_3_16'
    
    imgs_in = torch.load(data_file).cuda().float()
    #N_case = 64
    folds = imgs_in.shape[0]//N_case

    for subfold in range(folds):

        model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
            num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
            stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)
        model.cuda()

        #model = torch.compile(model_)

        embed_wb = nn.Embedding(N_case,64).cuda()
        optimizer = torch.optim.Adam(list(model.parameters())+list(embed_wb.parameters()),lr=0.001)

        num_iterations = 2500
        batch_size = 16
        run_loss = torch.zeros(num_iterations)
        run_psnr = torch.zeros(num_iterations)
        with tqdm(total=num_iterations, file=sys.stdout) as pbar:
            for i in range(num_iterations):
                optimizer.zero_grad();
                idx = torch.randperm(N_case)[:batch_size].cuda()
                target = imgs_in[idx.cpu()+subfold*N_case,0].cuda()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    code = embed_wb(idx)
                    x = model(code)[-1][:,0]
                loss = 1-ssim(target.unsqueeze(1).float(),x.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                
                run_psnr[i] = psnr(target.float().cpu().data.view(-1).numpy(),x.float().cpu().data.reshape(-1).numpy(),)
                run_loss[i] = loss.item()
                str1 = f"iter: {i}, loss: {'%0.3f'%(run_loss[i-28:i-1].mean().mul(100))}, psnr: {'%0.3f'%(run_psnr[i-28:i-1].mean().mul(1))}, GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
        model.half()
        torch.save([model.state_dict(),embed_wb.state_dict(),run_psnr],\
                   'nerv_ssim_sub'+str(subfold)+'.pth')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'fit_nerv_ssim args')
    #parser.add_argument('subfoldstart', help='usually 0,8,16,24')
    parser.add_argument('data_file', help='see create_demo_xray')
    parser.add_argument('gpu_num', help='usually 0-3', default='0')
    parser.add_argument('N_case', help='images per NeRV 32,40,64 (has to be divisor of N)', default='64')
    args = parser.parse_args()

    main(args.data_file,int(args.gpu_num),int(args.N_case))
