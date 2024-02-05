import imageio.v3 as iio
from tqdm import trange,tqdm
import torch
import torch.nn.functional as F
import torchvision
import zipfile
import wget
import os

def get_data(data_url,filename):
    if not os.path.exists(filename):
        wget.download(data_url)

        
def main():
    filename1 = './masksclavicles.zip'
    get_data('https://zenodo.org/records/8108809/files/masksclavicles.zip',filename1)
    filename2 = './masksheart.zip'
    get_data('https://zenodo.org/records/8108809/files/masksheart.zip',filename2)

    clavicles = zipfile.ZipFile(filename1, 'r')
    heart = zipfile.ZipFile(filename2, 'r')

    case_list0 = []; case_list1 = [];
    for i in range(1,len(clavicles.filelist)):
        file = clavicles.filelist[i].filename.split('_')
        if(file[3]=='0.png'):
            case_list0.append(int(file[2]))
        else:
            case_list1.append(int(file[2]))


    img = []
    for i in tqdm(case_list0):
        url = 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_0'+str(i).zfill(3)+'_0.png'
        img.append(F.interpolate(torch.from_numpy(iio.imread(url)).float().unsqueeze(0).unsqueeze(0)\
                                 ,scale_factor=.1,mode='bilinear').squeeze())
    for i in tqdm(case_list1):
        url = 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_0'+str(i).zfill(3)+'_1.png'
        img.append(F.interpolate(torch.from_numpy(iio.imread(url)).float().unsqueeze(0).unsqueeze(0)\
                                 ,scale_factor=.1,mode='bilinear').squeeze())

    mask_c = []
    mask_h = []
    for i in tqdm(case_list0):
        mask_c1 = torch.from_numpy(iio.imread(clavicles.open('masks_clavicles/MCUCXR_0'+str(i).zfill(3)+'_0.png')))
        mask_c.append(F.interpolate(mask_c1.float().unsqueeze(0).unsqueeze(0),scale_factor=.1,mode='nearest').squeeze())
        mask_h1 = torch.from_numpy(iio.imread(heart.open('masks_heart/MCUCXR_0'+str(i).zfill(3)+'_0.png')))
        mask_h.append(F.interpolate(mask_h1.float().unsqueeze(0).unsqueeze(0),scale_factor=.1,mode='nearest').squeeze())
    for i in tqdm(case_list1):
        mask_c1 = torch.from_numpy(iio.imread(clavicles.open('masks_clavicles/MCUCXR_0'+str(i).zfill(3)+'_1.png')))
        mask_c.append(F.interpolate(mask_c1.float().unsqueeze(0).unsqueeze(0),scale_factor=.1,mode='nearest').squeeze())
        mask_h1 = torch.from_numpy(iio.imread(heart.open('masks_heart/MCUCXR_0'+str(i).zfill(3)+'_1.png')))
        mask_h.append(F.interpolate(mask_h1.float().unsqueeze(0).unsqueeze(0),scale_factor=.1,mode='nearest').squeeze())


    bW = bH = 360;
    mask_crop = torch.zeros(len(mask_h),360,360)
    img_crop = torch.zeros(len(mask_h),360,360)
    for i in range(len(mask_h)):
        mask_ch = ((mask_h[i].long()|mask_c[i].long()))
        H,W = mask_ch.shape
        bbox = torchvision.ops.masks_to_boxes(mask_ch.unsqueeze(0)).squeeze()
        #bH,bW = int(bbox[3]-bbox[1]),int(bbox[2]-bbox[0])
        cH = int((bbox[1]+bbox[3])//2)
        cW = int((bbox[0]+bbox[2])//2)
        pad = (-(cW-bW//2),-(W-(cW+bW//2)),-(cH-bH//2)-40,-(H-(cH+bH//2))+40)
        img_crop[i] = F.pad(img[i],pad).div(255)
        mask_crop[i] = F.pad(mask_c[i],pad).div(255)

    torch.save(torch.stack((img_crop,mask_crop),1),'img_label_demo.pth')

         
if __name__ == '__main__':
    main()