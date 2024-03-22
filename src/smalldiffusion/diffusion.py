import math
from itertools import pairwise

import torch
import numpy as np
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional

class Schedule:
    '''Diffusion noise schedules parameterized by sigma'''
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas

    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        '''Called during sampling to get a decreasing sigma schedule with a
        specified number of sampling steps:
          - Spacing is "trailing" as in Table 2 of https://arxiv.org/abs/2305.08891
          - Includes initial and final sigmas
            i.e. len(schedule.sample_sigmas(steps)) == steps + 1
        '''
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]

    def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
        '''Called during training to get a batch of randomly sampled sigma values
        '''
        batchsize = x0.shape[0]
        return self[torch.randint(len(self), (batchsize,))].to(x0)

def sigmas_from_betas(betas: torch.FloatTensor):
    return (1/torch.cumprod(1.0 - betas, dim=0) - 1).sqrt()

# Simple log-linear schedule works for training many diffusion models
class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))

# Default parameters recover schedule used in most diffusion models
class ScheduleDDPM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start, beta_end, N)))

# Default parameters recover schedule used in most latent diffusion models, e.g. Stable diffusion
class ScheduleLDM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.00085, beta_end: float=0.012):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start**0.5, beta_end**0.5, N)**2))

# Given a batch of data x0, returns:
#   eps  : i.i.d. normal with same shape as x0
#   sigma: uniformly sampled from schedule, with shape Bx1x..x1 for broadcasting
def generate_train_sample(x0: torch.FloatTensor, schedule: Schedule):
    sigma = schedule.sample_batch(x0)
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    eps = torch.randn_like(x0)
    return sigma, eps

# Model objects
# Always called with (x, sigma):
#   If x.shape == [B, D1, ..., Dk], sigma.shape == [] or [B, 1, ..., 1].
#   If sigma.shape == [], model will be called with the same sigma for each x0
#   Otherwise, x[i] will be paired with sigma[i] when calling model
# Have a `rand_input` method for generating random xt during sampling

def training_loop(loader     : DataLoader,
                  model      : nn.Module,
                  schedule   : Schedule,
                  accelerator: Optional[Accelerator] = None,
                  epochs     : int = 10000,
                  lr         : float = 1e-3):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for _ in (pbar := tqdm(range(epochs))):
        for x0 in loader:
            optimizer.zero_grad()
            sigma, eps = generate_train_sample(x0, schedule)
            eps_hat = model(x0 + sigma * eps, sigma)
            loss = nn.MSELoss()(eps_hat, eps)
            yield SimpleNamespace(**locals()) # For extracting training statistics
            accelerator.backward(loss)
            optimizer.step()

def classifier_free_guidance_training_loop(
        loader     : DataLoader,
        model      : nn.Module,
        schedule   : Schedule,
        accelerator: Optional[Accelerator] = None,
        epochs     : int = 10000,
        lr         : float = 1e-3,
        drop_prob  : float = 0.1):
    """
    Train model using classifier free guidance.
    """
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for _ in (pbar := tqdm(range(epochs))):
        for x0, c in loader:
            optimizer.zero_grad()
            # dropout context with some probability
            context_mask = torch.bernoulli(torch.zeros_like(c)+(1-drop_prob))
            sigma, eps = generate_train_sample(x0, schedule)
            eps_hat = model(x0 + sigma * eps, sigma, c, context_mask)
            loss = nn.MSELoss()(eps_hat, eps)
            yield SimpleNamespace(**locals()) # For extracting training statistics
            accelerator.backward(loss)
            optimizer.step()
            pbar.set_description(f"loss: {loss:.4f}")

# Generalizes most commonly-used samplers:
#   DDPM       : gam=1, mu=0.5
#   DDIM       : gam=1, mu=0
#   Accelerated: gam=2, mu=0
@torch.no_grad()
def samples(model      : nn.Module,
            sigmas     : torch.FloatTensor, # Iterable with N+1 values for N sampling steps
            gam        : float = 1.,        # Suggested to use gam >= 1
            mu         : float = 0.,        # Requires mu in [0, 1)
            xt         : Optional[torch.FloatTensor] = None,
            accelerator: Optional[Accelerator] = None,
            batchsize  : int = 1):
    accelerator = accelerator or Accelerator()
    if xt is None:
        xt = model.rand_input(batchsize).to(accelerator.device) * sigmas[0]
    else:
        batchsize = xt.shape[0]
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps, eps_prev = model(xt, sig.to(xt)), eps
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(batchsize).to(xt)
        yield xt

@torch.no_grad()
def conditioned_samples(model      : nn.Module,
                       sigmas     : torch.FloatTensor,
                       gam        : float = 1.,
                       mu         : float = 0.,
                       xt         : Optional[torch.FloatTensor] = None,
                       accelerator: Optional[Accelerator] = None,
                       batchsize  : int = 1,
                       c          : Optional[torch.FloatTensor] = None,
                       guide_w    : float = 2.0):
    accelerator = accelerator or Accelerator()
    model = model.to(accelerator.device)
    if xt is None:
        xt = model.rand_input(batchsize).to(accelerator.device) * sigmas[0]
    else:
        batchsize = xt.shape[0]
        xt = xt.to(accelerator.device)
    if c is None:
        # mask all context
        context_mask = torch.zeros(batchsize, 1).to(accelerator.device)
        context_mask = context_mask.repeat(2, 1)
        c = torch.zeros(batchsize*2, 1).to(accelerator.device)
    else:
        # don't drop context at test time
        c = c.repeat(2, 1).to(accelerator.device)
        context_mask = torch.ones_like(c)
        # context_mask = context_mask.repeat(2, 1)
        context_mask[batchsize:] = 0. # makes second half of batch context free
        
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        # double the batch
        xt = xt.repeat(2, 1)

        eps_prev = eps
        eps_full = model(xt, sig.to(xt), c, context_mask)
        eps1 = eps_full[:batchsize]
        eps2 = eps_full[batchsize:]
        eps = (1+guide_w)*eps1 - guide_w*eps2
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt[:batchsize] # only keep the first half of the batch
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(batchsize).to(xt)
        yield xt