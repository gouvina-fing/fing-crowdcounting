from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import tqdm as tqdm

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
from misc.qualitycc import calc_psnr, calc_ssim

#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True

LOG_PARA = 100.0

#------------prepare data loader------------
data_mode = cfg.DATASET
if data_mode == 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode == 'SHHB':
    from datasets.SHHB.loading_data import loading_data 
    from datasets.SHHB.setting import cfg_data 
elif data_mode == 'QNRF':
    from datasets.QNRF.loading_data import loading_data 
    from datasets.QNRF.setting import cfg_data 
elif data_mode == 'UCF50':
    from datasets.UCF50.loading_data import loading_data 
    from datasets.UCF50.setting import cfg_data 
elif data_mode == 'WE':
    from datasets.WE.loading_data import loading_data 
    from datasets.WE.setting import cfg_data 
elif data_mode == 'GCC':
    from datasets.GCC.loading_data import loading_data
    from datasets.GCC.setting import cfg_data
elif data_mode == 'Mall':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.setting import cfg_data
elif data_mode == 'UCSD':
    from datasets.UCSD.loading_data import loading_data
    from datasets.UCSD.setting import cfg_data 
elif data_mode == "NWPU-Crowd":
    from datasets.NWPU.loading_data import loading_data
    from datasets.NWPU.setting import cfg_data


#------------Prepare Tester------------
net = cfg.NET

# TODO:
model_path = "../checkpoints/MCNN-all_ep_907_mae_218.5_mse_700.6_nae_2.005.pth"


exp_name = f'../{data_mode}_{net}_results'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name+'/pred'):
    os.mkdir(exp_name+'/pred')

if not os.path.exists(exp_name+'/gt'):
    os.mkdir(exp_name+'/gt')

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*cfg_data.MEAN_STD),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = cfg_data.DATA_PATH


def main():
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img')]                                           
    test(file_list[0], model_path)
   

def test(file_list, model_path):
    maes = AverageMeter()
    mses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    f1 = plt.figure(1)

    gts = []
    preds = []

    for filename in file_list:
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)

        gt = np.sum(den)
        with torch.no_grad():
            # NWPU does: img = Variable(img).cuda() and crop masks
            img = Variable(img[None,:,:,:]).cuda()
            pred_map = net.test_forward(img)

        sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]

        pred = np.sum(pred_map)/LOG_PARA
        pred_map = pred_map/np.max(pred_map+1e-20)

        den = den/np.max(den+1e-20)

        print(filename, pred, gt)
        s_psnr = calc_psnr(den, pred_map)
        s_ssim = calc_ssim(den, pred_map)

        maes.update(abs(gt-pred))
        mses.update((gt-pred)*(gt-pred))
        psnrs.update(s_psnr)
        ssims.update(s_ssim)

        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False) 
        den_frame.spines['bottom'].set_visible(False) 
        den_frame.spines['left'].set_visible(False) 
        den_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})

        diff = den-pred_map

        diff_frame = plt.gca()
        plt.imshow(diff, 'jet')
        plt.colorbar()
        diff_frame.axes.get_yaxis().set_visible(False)
        diff_frame.axes.get_xaxis().set_visible(False)
        diff_frame.spines['top'].set_visible(False) 
        diff_frame.spines['bottom'].set_visible(False) 
        diff_frame.spines['left'].set_visible(False) 
        diff_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})

    mae = maes.avg
    mse = np.sqrt(mses.avg)
    ssim = ssims.avg
    psnr = psnrs.avg
    print(f"MAE: {mae}; MSE: {mse}, SSIM: {ssim}, PSNR: {psnr}")



if __name__ == '__main__':
    main()




