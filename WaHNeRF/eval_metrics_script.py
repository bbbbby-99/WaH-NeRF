import os
import glob
from absl import app
from absl import flags
import torch
import imageio
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import lpips
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '', 'Dataset name.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  mse_fn = lambda x, y: np.mean((x - y)**2)
  psnr_fn = lambda x, y: -10 * np.log10(mse_fn(x, y))
  def ComputeMetrics(generated, gt):
    psnr_score = psnr_fn(generated, gt)
    return  psnr_score
  generated_views = './lego/'
  ground_truth_views = './data/nerf_synthetic/lego/test/'
  images_to_eval = glob.glob(os.path.join(generated_views, "*.png"))
  depth_to_eval = glob.glob(os.path.join(generated_views, "*depth.png"))
  images_to_eval = list(set(images_to_eval)-set(depth_to_eval))
    
  files = [os.path.basename(s) for s in images_to_eval]
  ssim = []
  psnr = []
  lpips = []

  lpips_mod = of_lpips(net='vgg')

  for k in files:
    try:
      gv_im = imageio.imread(os.path.join(generated_views, k))
      gv_im = (gv_im / 255.)

      num = (int(k.split('.')[0])) * 8
      gt_im = imageio.imread(os.path.join(ground_truth_views,  ("r_"+str(num)+'.png')))

      gt_im = (gt_im / 255.)
      gt_im = gt_im[..., :3] * gt_im[..., -1:] + (1. - gt_im[..., -1:])

    except Exception as e:
      print("I/O Error opening filename: %s" % k)
    lpip = lpips_mod.calc_lpips(gv_im, gt_im)

    psnr_score = ComputeMetrics(gv_im,gt_im)
    ssim_score = img2ssim(gv_im, gt_im)
    lpips.append(lpip.detach().numpy())
    psnr.append(psnr_score)
    ssim.append(ssim_score)

  print("PSNR:")
  print("Mean: %04f" % np.mean(psnr))
  print("Stddev: %04f" % np.std(psnr))
  print()

  print("SSIM:")
  print("Mean: %04f" % np.mean(ssim))
  print("Stddev: %04f" % np.std(ssim))


  print("LPIPS:")
  print("Mean: %04f" % np.mean(lpips))
  print("Stddev: %04f" % np.std(lpips))

def img2ssim(x, y, mask=None):
  if mask is not None:
    x = mask.unsqueeze(-1) * x
    y = mask.unsqueeze(-1) * y
  x = torch.Tensor(x)
  y = torch.Tensor(y)

  x = x.unsqueeze(0)
  y = y.unsqueeze(0)
  x = x.permute(0, 3, 1, 2)
  y = y.permute(0, 3, 1, 2)
  x = torch.clip(x.float(), 0, 1)
  y = torch.clip(y.float(), 0, 1)
  ssim_ = ssim(x, y, data_range=1)
  ms_ssim_ = ms_ssim(x, y, data_range=1)
  return ssim_, ms_ssim_

class of_lpips():
  def __init__(self, net, use_gpu=False):
    ## Initializing the model
    self.loss_fn = lpips.LPIPS(net=net)
    self.use_gpu = use_gpu
    if use_gpu:
      self.loss_fn.cuda()

  def calc_lpips(self, img0, img1):
    img0 = torch.tensor(img0)  # RGB image from [-1,1]
    img1 = torch.tensor(img1)
    img0 = torch.clip(img0.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
    img1 = torch.clip(img1.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
    if self.use_gpu:
      img0 = img0.cuda()
      img1 = img1.cuda()
    dist01 = self.loss_fn.forward(img0, img1)
    return dist01


if __name__ == '__main__':
  app.run(main)
