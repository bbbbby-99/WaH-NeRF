import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))

def train_model(config):
    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    data_AUX = iter(cycle(
        get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="render", factor=config.factor,
                       batch_size= config.batch_size, shuffle=True, device=config.device, pertube = True)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()

    os.makedirs(config.log_dir, exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, 'train'), flush_secs=1)
    SmoothnessLoss = EdgePreservingSmoothnessLoss()
    for step in tqdm(range(0, config.max_steps)):
        rays, pixels = next(data)
        rays_aux, rays_aux_pertube = next(data_AUX)
        rays_aux = namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]), rays_aux)
        rays_aux_pertube = namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]), rays_aux_pertube)
        rays = namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]), rays)
        pixels = pixels.reshape([-1, pixels.shape[-1]])
        comp_rgb, distances, _= model(rays)
############################################################################
        with torch.no_grad():
            comp_rgb_aux, distances_aux, acc, loss = model(rays_aux, Flag= False)
            comp_rgb_aux2, distances_aux2, acc2, _ = model(rays_aux_pertube, Flag= False)
        loss.require_grad = True
        pixels = pixels.to(config.device)
        reshape_to_patch = lambda x, dim: x.reshape(-1, 8, 8, dim)
        EdgePreservingSmoothnessLosses = []
        for comp_rgb_x, distances_x, acc_x in zip(comp_rgb_aux, distances_aux, acc):
            depth = reshape_to_patch(distances_x, 1)
            acc_x = reshape_to_patch(acc_x, 1)
            EdgePreservingSmoothnessLosses.append(
                SmoothnessLoss(depth, acc_x).mean())
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        loss_val = loss_val + (
                    EdgePreservingSmoothnessLosses[0] * 0.1 + EdgePreservingSmoothnessLosses[1]) * 0.1 + (loss_patch(
            comp_rgb_aux[1], comp_rgb_aux2[1]) + loss_patch(comp_rgb_aux[0], comp_rgb_aux2[0])*0.5 + loss_patch(
            distances_aux[1], distances_aux2[1]) + loss_patch(distances_aux[0], distances_aux2[0])*0.5+ loss_patch(
            acc[1], acc2[1]) + loss_patch(acc[0], acc2[0])*0.5) * 0.5 + (MSEloss(
            comp_rgb_aux[1], comp_rgb_aux2[1]) + MSEloss(comp_rgb_aux[0], comp_rgb_aux2[0])*0.5 + MSEloss(
            distances_aux[1], distances_aux2[1]) + MSEloss(distances_aux[0], distances_aux2[0])*0.5+ MSEloss(
            acc[1], acc2[1]) + MSEloss(acc[0], acc2[0])*0.5) * 0.5 + (loss[0]*0.1 + loss[1])*0.0001
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()
        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)
        print('train/loss', float(loss_val.detach().cpu().numpy()), str(step))
        print('train/coarse_psnr', float(np.mean(psnr[:-1])),  str(step))
        print('train/fine_psnr', float(psnr[-1]),  str(step))
        print('train/avg_psnr', float(np.mean(psnr)),  str(step))
        print('train/lr', float(scheduler.get_last_lr()[-1]), str(step))

        if step % config.save_every == 0:
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().cpu().numpy()
                logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
                logger.add_scalar('eval/fine_psnr', float(psnr[-1]), global_step=step)
                logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)), global_step=step)
            model_save_path = path.join(config.log_dir,str(step)+"model.pt")
            optimizer_save_path = path.join(config.log_dir, str(step)+"optim.pt")
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_save_path)

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)

def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])


class EdgePreservingSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 10.0
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / self.gamma)

    def forward(self, inputs, weights):
        inputs = inputs.squeeze(3)
        w1 = self.bilateral_filter(weights[:, :, :-1] - weights[:, :, 1:])
        w2 = self.bilateral_filter(weights[:, :-1, :] - weights[:, 1:, :])
        w3 = self.bilateral_filter(weights[:, :-1, :-1] - weights[:, 1:, 1:])
        w4 = self.bilateral_filter(weights[:, 1:, :-1] - weights[:, :-1, 1:])

        L1 = self.loss(w1 * (inputs[:, :, :-1] - inputs[:, :, 1:]))
        L2 = self.loss(w2 * (inputs[:, :-1, :] - inputs[:, 1:, :]))
        L3 = self.loss(w3 * (inputs[:, :-1, :-1] - inputs[:, 1:, 1:]))
        L4 = self.loss(w4 * (inputs[:, 1:, :-1] - inputs[:, :-1, 1:]))
        return (L1 + L2 + L3 + L4) / 4

def MSEloss(pred, pred2):
    loss = ((pred - pred2) ** 2).sum()
    return loss

def loss_patch(pred, pred2):
    pred = pred.reshape(16, 64, -1).permute(0, 2, 1) # 16,3,64
    pred2 = pred2.reshape(16, 64, -1).permute(0, 2, 1)  # 16,3,64
    for i in range(8*8):
        pred_1 = pred[:, : ,i].unsqueeze(2).expand_as(pred2)
        loss_1 = ((pred_1 - pred2) ** 2).sum(1)
        loss_min = loss_1.mean(1)
        if i == 0:
            patch_loss = loss_min
        else:
            patch_loss += loss_min
    return patch_loss.sum()

if __name__ == "__main__":
    config = get_config()
    train_model(config)
