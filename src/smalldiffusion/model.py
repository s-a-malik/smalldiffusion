import torch
from torch import nn
from itertools import pairwise

def get_sigma_embeds(batches, sigma):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    sigma = sigma.unsqueeze(1)
    return torch.cat([torch.sin(torch.log(sigma)/2),
                      torch.cos(torch.log(sigma)/2)], dim=1)

class ModelMixin:
    def rand_input(self, batchsize):
        assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
        return torch.randn((batchsize,) + self.input_dims)

class TimeInputMLP(nn.Module, ModelMixin):
    def __init__(self, dim=2, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + 2,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], dim))

        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x 2
        nn_input = torch.cat([x, sigma_embeds], dim=1)               # shape: b x (dim + 2)
        return self.net(nn_input)

class TimeInputMLPConditional(nn.Module, ModelMixin):
    def __init__(self, dim=2, cond_dim=1, cond_hid_dim=16, hidden_dims=(16,128,256,128,16)):
        super().__init__()

        self.input_dims = (dim,)
        self.cond_dim = cond_dim
        self.cond_hid_dim = cond_hid_dim

        # one hot embedding
        if cond_dim > 1:
            self.c_embed = nn.Embedding(cond_dim, cond_hid_dim)
        else:
            self.c_embed = nn.Linear(cond_dim, cond_hid_dim)
        layers = []
        for in_dim, out_dim in pairwise((dim + 2 + cond_hid_dim,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma, c, context_mask):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        # c     shape: b x 1
        # context_mask shape: b x 1
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x 2
        # embed c
        c_embeds = self.c_embed(c) # shape: b x cond_hid_dim
        # mask context
        c_embeds = c_embeds * context_mask

        nn_input = torch.cat([x, sigma_embeds, c_embeds], dim=1)               # shape: b x (dim + c_embed_dim + 2)
        return self.net(nn_input)


def sq_norm(M, k):
    # M: b x n --(norm)--> b --(repeat)--> b x k
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

class IdealDenoiser(ModelMixin):
    def __init__(self, dataset):
        self.data = torch.stack([dataset[i] for i in range(len(dataset))])
        self.input_dims = self.data.shape[1:]

    def __call__(self, x, sigma):
        assert sigma.shape == tuple(), 'Only singleton sigma supported'
        data = self.data.to(x)
        x_flat = x.flatten(start_dim=1)
        d_flat = data.flatten(start_dim=1)
        xb, xr = x_flat.shape
        db, dr = d_flat.shape
        assert xr == dr, 'Input x must have same dimension as data!'
        # ||x - x0||^2 ,shape xb x db
        sq_diffs = sq_norm(x_flat, db) + sq_norm(d_flat, xb).T - 2 * x_flat @ d_flat.T
        weights = torch.nn.functional.softmax(-sq_diffs/2/sigma**2, dim=1)
        return (x - torch.einsum('ij,j...->i...', weights, data))/sigma
