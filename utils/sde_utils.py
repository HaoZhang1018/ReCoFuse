import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate
from collections import OrderedDict

class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass
    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''
    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01,  device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma 
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1 # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps): 
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1 # T from 1 to 100 
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s = 0.008): 
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2 
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma**2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma**2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))
            
        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0] # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)
        
        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t]**2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t]**2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t,control_from_list=None, **kwargs):
        # need to pre-set mu and score_model
        if control_from_list is not None:
            noise = self.model(x, self.mu, t, control = control_from_list)
        else:
            noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def reverse_optimum_std(self, t):
        A = torch.exp(-2*self.thetas[t] * self.dt)
        B = torch.exp(-2*self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-2*self.thetas_cumsum[t-1] * self.dt)

        posterior_var = (1 - A) * (1 - C) / (1 - B)

        min_value = (1e-20 * self.dt).to(self.device)
        log_posterior_var = torch.log(torch.clamp(posterior_var, min=min_value))
        return (0.5 * log_posterior_var).exp() * self.max_sigma

    def reverse_posterior_step(self, xt, noise, t):
        x0 = self.get_init_state_from_noise(xt, noise, t)
        mean = self.reverse_optimum_step(xt, x0, t)
        std = self.reverse_optimum_std(t)
        return mean + std * torch.randn_like(xt)

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t)**2

    def get_init_state_from_noise(self, xt, noise, t):
        A = torch.exp(self.thetas_cumsum[t] * self.dt)
        return (xt - self.mu - self.sigma_bar(t) * noise) * A + self.mu

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', control_list=None,**kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if control_list is not None:
                score = self.score_fn(x, t, control_from_list=control_list.pop(),**kwargs)
            else:
                score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_ode_step(x, score, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_posterior(self, xt, T=-1, save_states=False, save_dir='posterior_state', **kwargs):
        T = self.T if T < 0 else T

        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.noise_fn(x, t, **kwargs)
            x = self.reverse_posterior_step(x, noise, t)

            if save_states: # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x


    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3,):
        shape = xt.shape

        def to_flattened_numpy(x):
          """Flatten a torch tensor `x` and convert it to numpy."""
          return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
          """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
          return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                     rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x
    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma



############################ DualPath SDE ####################################
class DualPathSDE(IRSDE):

    def score_fn(self, x, t,**kwargs):
        B = x.shape[0] // 2
        x_vis, x_ir = x[:B], x[B:]
        mu_vis,mu_ir=self.mu[:B], self.mu[B:]
        noise_vis_pred, noise_ir_pred = self.model(x_vis, mu_vis, x_ir,mu_ir, t)
        score_vis = -noise_vis_pred / self.sigma_bar(t)
        score_ir = -noise_ir_pred / self.sigma_bar(t)
        return torch.cat([score_vis, score_ir], dim=0)

    def noise_fn(self, x, t, **kwargs):
        B = x.shape[0] // 2
        x_vis, x_ir = x[:B], x[B:]
        mu_vis,mu_ir=self.mu[:B], self.mu[B:]
        noise_vis_pred, noise_ir_pred = self.model(x_vis, mu_vis, x_ir, mu_ir, t)
        return torch.cat([noise_vis_pred, noise_ir_pred], dim=0)
    
    def mu_bar(self, x0, mu,t):
        return mu + (x0 - mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)
    
    def generate_random_states(self, x0_vis, x0_ir, vis_cond, ir_cond):
        x0_vis = x0_vis.to(self.device)
        x0_ir = x0_ir.to(self.device)
        vis_cond = vis_cond.to(self.device)
        ir_cond = ir_cond.to(self.device)
        self.set_mu(torch.cat([vis_cond, ir_cond], dim=0))
        batch = x0_vis.shape[0]
        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        B=x0_ir.shape[0]
        vis_mean = self.mu_bar(x0_vis,self.mu[:B], timesteps)
        ir_mean = self.mu_bar(x0_ir,self.mu[B:], timesteps)
        noise_vis = torch.randn_like(vis_mean)
        noise_ir = torch.randn_like(ir_mean)
        sigma = self.sigma_bar(timesteps)
        noisy_vis = noise_vis * sigma + vis_mean
        noisy_ir = noise_ir * sigma + ir_mean
        return timesteps, noisy_vis.to(torch.float32),noisy_ir.to(torch.float32)

    def reverse_optimum_step_(self, xt, x0, mu,t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t-1] * self.dt)

        term1 = A * (1 - C**2) / (1 - B**2)
        term2 = C * (1 - A**2) / (1 - B**2)

        return term1 * (xt - mu) + term2 * (x0 - mu) + mu

    def reverse_optimum_step(self, xt, x0, t):
        B = xt.shape[0] // 2
        xt_vis, xt_ir = xt[:B], xt[B:]
        x0_vis, x0_ir = x0[:B], x0[B:]

        out_vis = self.reverse_optimum_step_(xt_vis, x0_vis,self.mu[:B], t)
        out_ir = self.reverse_optimum_step_(xt_ir, x0_ir,self.mu[B:], t)

        return out_vis, out_ir
    
    def sde_reverse_drift(self, x, score,mu,t):
        return (self.thetas[t] * (mu - x) - self.sigmas[t]**2 * score) * self.dt

    def reverse_sde_step(self, x, score, t):
        B = x.shape[0] // 2
        x_vis, x_ir = x[:B], x[B:]
        score_vis, score_ir = score[:B], score[B:]

        out_vis= x_vis - self.sde_reverse_drift(x_vis, score_vis,self.mu[0:B],t) - self.dispersion(x_vis, t)
        out_ir= x_ir - self.sde_reverse_drift(x_ir, score_ir,self.mu[B:],t) - self.dispersion(x_ir, t)
        return torch.cat([out_vis, out_ir], dim=0)

    def reverse_sde_step_mean(self, x, score, t):
        B = x.shape[0] // 2
        x_vis, x_ir = x[:B], x[B:]
        score_vis, score_ir = score[:B], score[B:]
        out_vis = x_vis - self.sde_reverse_drift(x_vis, score_vis,self.mu[0:B], t)
        out_ir = x_ir - self.sde_reverse_drift(x_ir, score_ir,self.mu[B:],t)
        return out_vis, out_ir

    def reverse_sde_dual(self, xt, T=-1,fusion_model=None, **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        B = x.shape[0] // 2
            
        for t in tqdm(reversed(range(1, T + 1))):
            x_vis, x_ir = x[:B], x[B:]

            with torch.no_grad():
                x_vis_ir_fused = fusion_model(x_vis, x_ir,t)

            x = torch.cat([x_vis_ir_fused, x_vis_ir_fused], dim=0)
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_sde_step(x, score, t)

        return x
