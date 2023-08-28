import torch
import torch.nn as nn
from ray_utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map
from pose_utils import to8b
from torch.nn.parameter import Parameter

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret


class MipNeRF(nn.Module):
    def __init__(self,
                 use_viewdirs=True,
                 randomized=False,
                 ray_shape="cone",
                 white_bkgd=True,
                 num_levels=2,
                 num_samples=128,
                 hidden=256,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 device=torch.device("cpu"),
                 return_raw=False
                 ):
        super(MipNeRF, self).__init__()
        self.use_viewdirs = use_viewdirs
        self.init_randomized = randomized
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.white_bkgd = white_bkgd
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_input = (max_deg - min_deg) * 3 * 2
        self.rgb_input = 3 + ((viewdirs_max_deg - viewdirs_min_deg) * 3 * 2)
        self.density_noise = density_noise
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.density_bias = density_bias
        self.hidden = hidden
        self.device = device
        self.return_raw = return_raw
        self.density_activation = nn.Softplus()
        self.w1 = Parameter(torch.fmod(torch.zeros((256, 1)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.zeros((1)).cuda(),2))

        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(hidden, 1),
        )
        input_shape = hidden
        if self.use_viewdirs:
            input_shape = num_samples

            self.rgb_net0 = nn.Sequential(
                nn.Linear(hidden, hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(viewdirs_min_deg, viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(hidden + self.rgb_input, num_samples),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)
        self.to(device)

    def forward(self, rays, Flag = True):
        comp_rgbs = []
        distances = []
        accs = []
        losses = []
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample
                t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
                                                        rays.near, rays.far, randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
                                                          t_vals.to(rays.origins.device),
                                                          weights.to(rays.origins.device),
                                                          randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])
            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            new_encodings1 = new_encodings
            raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(rays.viewdirs.to(self.device))
                viewdirs = torch.cat((viewdirs, rays.viewdirs.to(self.device)), -1)
                viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                new_encodings = self.rgb_net0(new_encodings)
                new_encodings = torch.cat((new_encodings, viewdirs), -1)
                new_encodings = self.rgb_net1(new_encodings)
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))
            offset = self.offset_setting(new_encodings1)
            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)
            # volumetric rendering
            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            if Flag:
                comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals,
                                                                               rays.directions.to(rgb.device),
                                                                               self.white_bkgd)
            else:
                comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals,
                                                                               rays.directions.to(rgb.device),
                                                                               self.white_bkgd, offset)
                s_vals = t_to_s(t_vals=t_vals, near=rays.near, far=rays.far)
                mask = getMask(weights)
                loss_I = rmi_loss((1 / (weights + 0.0005)), offset.squeeze(),mask)
                loss = loss_dist(s_vals, weights) + loss_I * 1e3 * 5
                losses.append(loss)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)

        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            if Flag:
                return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)
            else:
                return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), torch.stack(losses)

    def offset_setting(self, feature):
        offset = (torch.matmul(feature, (self.w1+1e-7)) + (self.b1+1e-7)).reshape((-1, 128, 1))
        return  offset

    def render_image(self, rays, height, width, chunks=8192):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        length = rays[0].shape[0]
        rgbs = []
        dists = []
        accs = []
        with torch.no_grad():
            for i in range(0, length, chunks):
                # put chunk of rays on device
                chunk_rays = namedtuple_map(lambda r: r[i:i+chunks].to(self.device), rays)
                rgb, distance, acc = self(chunk_rays)
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                accs.append(acc[-1].cpu())

        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        return rgbs, dists, accs

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()

def getMask(weight, boundary = 0.1):
    one = torch.ones_like(weight)
    zero = torch.zeros_like(weight)
    return torch.where(weight > boundary, one, zero)


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

def def_perturb(size, norm, epsilons):
    d = torch.zeros(size).cuda()
    if norm == "l_inf":
        d.uniform_(-epsilons, epsilons)
    elif norm == "l_2":
        d.normal_()
        ndim = d.ndim
        d_flat = d.view(d.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view([d.size(0)] + [1] * (ndim - 1))
        r = torch.zeros_like(n).uniform_(0, 1)
        d *= r / n * epsilons
    else:
        raise ValueError

    d.requires_grad = False

    return d

def loss_dist(s_vals,weights):
    """compute the loss_dist according to the paper

    Arguments:
        s_vals:torch.tensor,[batch_size,num_samples+1],sampled disparity values.
        weights:torch.tensor(float32),[batch_size,num_samples], weights for t_vals

    Returns:
        loss_dist:torch.tensor(float32),loss_dist according to the paper
    """
    loss_dist = 0
    for i in range(weights.shape[-1]):
        for j in range(weights.shape[-1]):
            loss_dist += torch.sum(weights[...,i] * weights[...,j] * torch.abs((s_vals[...,i] + s_vals[...,i+1])/2-(s_vals[...,j] + s_vals[...,j+1])/2))
    loss_dist += 1/3 * torch.sum(weights ** 2 * (s_vals[...,1:]-s_vals[...,:-1]))

    return loss_dist


def t_to_s(t_vals,near,far):
    """transform t to s:using the formula in the paper"""
    s_vals = (g(t_vals) - g(near)) / (g(far) - g(near))
    return s_vals

def g(x):
    """compute the disparity of x:g(x)=1/x"""
    # pad the tensor to avoid dividing zero
    eps = 1e-6
    x += eps
    s = 1/x
    return s

def s_to_t(s_vals, near, far):
    """transform s to t:using the formula in the paper"""
    t_vals = g(s_vals * g(far) + (1 - s_vals) * g(near))
    return t_vals


def rmi_loss(input, target, mask):
    """
    Calculates the RMI loss between the prediction and target.
    :return:
        RMI loss
    """
    assert input.shape == target.shape

    # Convert to doubles for better precision

    y = target.double()
    p = input.double()

    # Small diagonal matrix to fix numerical issues
    eps = torch.eye(p.size(1), dtype=y.dtype, device=y.device) * 0.0005
    eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

    # Subtract mean
    y = y.unsqueeze(2)
    p = p.unsqueeze(2)
    # Covariances
    y_cov = y @ transpose(y)
    p_cov = p @ transpose(p)
    y_p_cov = y @ transpose(p)

    # Approximated posterior covariance matrix of Y given P
    m = y_cov - y_p_cov @ transpose(inverse(p_cov + eps)) @ transpose(y_p_cov)
    mask = mask.unsqueeze(0).unsqueeze(3)
    # Sum over classes, mean over samples.
    return (m * mask).sum()

def transpose(x):
    return x.transpose(-2, -1)

def inverse(x):
    return torch.inverse(x)
