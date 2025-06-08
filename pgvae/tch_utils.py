# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: p38
# ---

# %% [markdown]
# ## PyTorch Utilities

# %% code_folding=[5, 12, 42, 373]
import numpy as np
import torch
from torch import nn
import gc
import warnings
from collections import defaultdict


def isscalar(v):
    if torch.is_tensor(v):
        return v.numel() == 1
    else:
        return np.isscalar(v)

def torch_qr_eff(a):
    """
    Due to a bug in MAGMA, qr on cuda is super slow for small matrices. 
    Therefore, this step must be performed on the cpu.
    
    See the following:
        https://github.com/pytorch/pytorch/issues/22573
        https://github.com/cornellius-gp/gpytorch/pull/1224
    """
    assert not a.requires_grad
    q, r = torch.qr(a.detach().cpu(), some=False)
    return q.to(device=a.device), r.to(device=a.device)

def profmem():
    """
    profiles the memory usage by alive pytorch tensors.
    
    Outputs
    -------
    stats (dict): a torch.device mapping to the number of 
        bytes used by the pytorch tensors.
    """
    fltr_msg  = "torch.distributed.reduce_op is deprecated, "
    fltr_msg += "please use torch.distributed.ReduceOp instead"
    warnings.filterwarnings("ignore", message=fltr_msg)

    stats = defaultdict(lambda: 0)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') 
                and torch.is_tensor(obj.data)):            
                stats[str(obj.device)] += obj.numel() * obj.element_size()
        except:
            pass
    
    return stats

class EMA:
    def __init__(self, gamma, gamma_sq):
        self.gamma = gamma
        self.gamma_sq = gamma_sq
        self.ema = dict()
        self.ema_sq = dict()
    
    def __call__(self, key, val):
        gamma = self.gamma
        gamma_sq = self.gamma_sq
        ema, ema_sq = self.ema, self.ema_sq
        with torch.no_grad():
            val_ema = ema.get(key, val)
            val_ema = gamma * val_ema + (1-gamma) * val
            n_seeds = val_ema.numel()
            ema[key] = val_ema
            
            val_ema_sq = ema_sq.get(key, val**2)
            val_ema_sq = gamma_sq * val_ema_sq + (1-gamma_sq) * (val**2)
            ema_sq[key] = val_ema_sq

            val_popvar = (val_ema_sq - (val_ema**2)).detach().cpu().numpy()
            val_popvar[val_popvar < 0] = 0
            val_ema_std = np.sqrt(val_popvar) * np.sqrt((1-gamma)/(1+gamma))
            
            val_ema_mean = val_ema.mean()
            val_ema_std_mean = np.sqrt((val_ema_std**2).sum()) / n_seeds
        return val_ema_mean, val_ema_std_mean


class BatchRNG:
    is_batch = True

    def __init__(self, shape, lib, device, dtype,
                 unif_cache_cols=1_000_000,
                 norm_cache_cols=5_000_000):
        assert lib in ('torch', 'numpy')

        self.lib = lib
        self.device = device

        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)
        self.reset_shape_attrs(shape)

        if self.lib == 'torch':
            self.rngs = [torch.Generator(device=self.device)
                         for _ in range(self.shape_prod)]
        else:
            self.rngs = [None for _ in range(self.shape_prod)]
        self.dtype = dtype

        self.unif_cache_cols = unif_cache_cols
        if self.lib == 'torch':
            self.unif_cache = torch.empty((self.shape_prod, self.unif_cache_cols),
                                          device=self.device, dtype=dtype)
        else:
            self.unif_cache = np.empty(
                (self.shape_prod, self.unif_cache_cols), dtype=dtype)
        # So that it would get refilled immediately
        self.unif_cache_col_idx = self.unif_cache_cols
        self.unif_cache_rng_states = None

        self.norm_cache_cols = norm_cache_cols
        if self.lib == 'torch':
            self.norm_cache = torch.empty((self.shape_prod, self.norm_cache_cols),
                                          device=self.device, dtype=dtype)
        else:
            self.norm_cache = np.empty(
                (self.shape_prod, self.norm_cache_cols), dtype=dtype)
        # So that it would get refilled immediately
        self.norm_cache_col_idx = self.norm_cache_cols
        self.norm_cache_rng_states = None

        self.np_qr = np.linalg.qr
        try:
            ver = tuple(int(x) for x in np.__version__.split('.'))
            np_majorver, np_minorver, np_patchver, *_ = ver
            if not ((np_majorver >= 1) and (np_minorver >= 22)):
                def np_qr_fakebatch(a):
                    b = a.reshape(-1, a.shape[-2], a.shape[-1])
                    q, r = list(zip(*[np.linalg.qr(x) for x in b]))
                    qs = np.stack(q, axis=0)
                    rs = np.stack(r, axis=0)
                    qout = qs.reshape(
                        *a.shape[:-2], qs.shape[-2], qs.shape[-1])
                    rout = rs.reshape(
                        *a.shape[:-2], rs.shape[-2], rs.shape[-1])
                    return qout, rout
                self.np_qr = np_qr_fakebatch
        except Exception:
            pass
        
        

    def reset_shape_attrs(self, shape):
        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)

    def seed(self, seed_arr):
        # Collecting the rng_states after seeding
        assert isinstance(seed_arr, np.ndarray)
        assert len(self.rngs) == seed_arr.size
        flat_seed_arr = seed_arr.copy().reshape(-1)
        if self.lib == 'torch':
            np_random = np.random.RandomState(seed=0)
            for seed, rng in zip(flat_seed_arr, self.rngs):
                np_random.seed(seed)
                balanced_32bit_seed = np_random.randint(
                    0, 2**31-1, dtype=np.int32)
                rng.manual_seed(int(balanced_32bit_seed))
        else:
            self.rngs = [np.random.RandomState(
                seed=seed) for seed in flat_seed_arr]

        if self.unif_cache_col_idx < self.unif_cache_cols:
            self.refill_unif_cache()
            # The cache has been used before, so in order to be able to
            # concat this sampler with the non-reseeded sampler, we should not
            # change the self.unif_cache_cols.

            # Note: We should not refill the uniform cache if the model
            # has not been initialized. This is done to keep the backward
            # compatibility and reproducibility properties with the old scripts.
            # Otherwise, the order of random samplings will change. Remember that
            # the old script first uses dirichlet and priors, and then refills
            # the unif/norm cache. In order to be similar, we should avoid
            # refilling the cache upon the first .seed() call
        if self.norm_cache_col_idx < self.norm_cache_cols:
            self.refill_norm_cache()

    def get_state(self):
        state_dict = dict(unif_cache_rng_states=self.unif_cache_rng_states,
                          norm_cache_rng_states=self.norm_cache_rng_states,
                          norm_cache_col_idx=self.norm_cache_col_idx,
                          unif_cache_col_idx=self.unif_cache_col_idx,
                          rng_states=self.get_rng_states(self.rngs))
        return state_dict

    def set_state(self, state_dict):
        unif_cache_rng_states = state_dict['unif_cache_rng_states']
        norm_cache_rng_states = state_dict['norm_cache_rng_states']
        norm_cache_col_idx = state_dict['norm_cache_col_idx']
        unif_cache_col_idx = state_dict['unif_cache_col_idx']
        rng_states = state_dict['rng_states']

        if unif_cache_rng_states is not None:
            self.set_rng_states(unif_cache_rng_states, self.rngs)
            self.refill_unif_cache()
            self.unif_cache_col_idx = unif_cache_col_idx
        else:
            self.unif_cache_col_idx = self.unif_cache_cols
            self.unif_cache_rng_states = None

        if norm_cache_rng_states is not None:
            self.set_rng_states(norm_cache_rng_states, self.rngs)
            self.refill_norm_cache()
            self.norm_cache_col_idx = norm_cache_col_idx
        else:
            self.norm_cache_col_idx = self.norm_cache_cols
            self.norm_cache_rng_states = None

        self.set_rng_states(rng_states, self.rngs)

    def get_rngs(self):
        return self.rngs

    def set_rngs(self, rngs, shape):
        assert isinstance(rngs, list)
        self.reset_shape_attrs(shape)
        self.rngs = rngs
        assert len(
            self.rngs) == self.shape_prod, f'{len(self.rngs)} != {self.shape_prod}'

    def get_rng_states(self, rngs):
        """
        getting state in ByteTensor
        """
        rng_states = []
        for i, rng in enumerate(rngs):
            rng_state = rng.get_state()
            if self.lib == 'torch':
                rng_state = rng_state.detach().clone()
            rng_states.append(rng_state)
        return rng_states

    def set_rng_states(self, rng_states, rngs):
        """
        rng_states should be ByteTensor (RNG state must be a torch.ByteTensor)
        """
        assert isinstance(
            rng_states, list), f'{type(rng_states)}, {rng_states}'
        for i, rng in enumerate(rngs):
            rs = rng_states[i]
            if self.lib == 'torch':
                rs = rs.cpu()
            rng.set_state(rs)

    def __call__(self, gen, sample_shape):
        assert self.lib == 'torch'
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, rng in enumerate(self.rngs):
            rng_state = rng.get_state()
            rng_state = rng_state.detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(
                self.device).detach().clone())
        rv = torch.stack(random_vars, dim=0).reshape(*sample_shape)
        return rv

    def dirichlet(self, gen_list, sample_shape):
        assert self.lib == 'torch'
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, (gen_, rng) in enumerate(zip(gen_list, self.rngs)):
            rng_state = rng.get_state().detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen_.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(
                self.device).detach().clone())

        rv = torch.stack(random_vars, dim=0)
        rv = rv.reshape(*self.shape, *rv.shape[1:])
        return rv

    def refill_unif_cache(self):
        self.unif_cache_rng_states = self.get_rng_states(self.rngs)
        if self.lib == 'torch':
            for row, rng in enumerate(self.rngs):
                self.unif_cache[row].uniform_(generator=rng)
        else:
            for row, rng in enumerate(self.rngs):
                self.unif_cache[row] = rng.rand(self.unif_cache_cols)

    def refill_norm_cache(self):
        self.norm_cache_rng_states = self.get_rng_states(self.rngs)
        if self.lib == 'torch':
            for row, rng in enumerate(self.rngs):
                self.norm_cache[row].normal_(generator=rng)
        else:
            for row, rng in enumerate(self.rngs):
                self.norm_cache[row] = rng.randn(self.norm_cache_cols)

    def uniform(self, sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        assert sample_shape_tuple[:self.shape_len] == self.shape

        sample_shape_rightmost = sample_shape[self.shape_len:]
        cols = np.prod(sample_shape_rightmost)
        if self.unif_cache_col_idx + cols >= self.unif_cache_cols:
            self.refill_unif_cache()
            self.unif_cache_col_idx = 0

        samples = self.unif_cache[:, self.unif_cache_col_idx: (
            self.unif_cache_col_idx + cols)]
        samples = samples.reshape(*sample_shape)
        self.unif_cache_col_idx += cols

        return samples

    def normal(self, sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        cols = np.prod(sample_shape_tuple) // self.shape_prod
        assert cols * self.shape_prod == np.prod(sample_shape_tuple)
        if self.norm_cache_col_idx + cols >= self.norm_cache_cols:
            self.refill_norm_cache()
            self.norm_cache_col_idx = 0

        samples = self.norm_cache[:, self.norm_cache_col_idx: (
            self.norm_cache_col_idx + cols)]
        samples = samples.reshape(*sample_shape)
        self.norm_cache_col_idx += cols

        return samples

    def so_n(self, sample_shape):        
        sample_shape_tuple = tuple(sample_shape)

        assert sample_shape_tuple[-2] == sample_shape_tuple[-1]
        n_bch, d = self.shape_prod, sample_shape_tuple[-1]
        sample_numel = np.prod(sample_shape_tuple)
        n_v = sample_numel // (self.shape_prod * d * d)
        assert sample_numel == (n_bch * n_v * d * d)
        qr_factorizer = torch_qr_eff if self.lib == 'torch' else self.np_qr
        diagnalizer = torch.diagonal if self.lib == 'torch' else np.diagonal
        signer = torch.sign if self.lib == 'torch' else np.sign

        norms = self.normal((n_bch, n_v, d, d))
        assert norms.shape == (n_bch, n_v, d, d)
        q, r = qr_factorizer(norms)
        assert q.shape == (n_bch, n_v, d, d)
        assert r.shape == (n_bch, n_v, d, d)
        r_diag = diagnalizer(r, 0, -2, -1)
        assert r_diag.shape == (n_bch, n_v, d)
        r_diag_sign = signer(r_diag)
        assert r_diag_sign.shape == (n_bch, n_v, d)
        q_signed = q * r_diag_sign.reshape(n_bch, n_v, 1, d)
        assert q_signed.shape == (n_bch, n_v, d, d)
        so_n = q_signed.reshape(*sample_shape_tuple)
        assert so_n.shape == sample_shape_tuple
        
        return so_n

    @classmethod
    def Merge(cls, sampler1, sampler2):
        assert sampler1.shape_len == sampler2.shape_len == 1

        device = sampler1.device
        dtype = sampler1.dtype
        chain_size = (sampler1.shape[0]+sampler2.shape[0],)

        state_dict1, state_dict2 = sampler1.get_state(), sampler2.get_state()

        merged_state_dict = dict()
        for key in state_dict1:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states', 'rng_states'):
                # saba modified
                if (state_dict1[key] is None) and (state_dict2[key] is None):
                    merged_state_dict[key] = None
                elif (state_dict1[key] is None) or (state_dict2[key] is None):
                    raise ValueError(f"{key} with None occurance")
                else:
                    merged_state_dict[key] = state_dict1[key] + \
                        state_dict2[key]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                assert state_dict1[key] == state_dict2[key]
                merged_state_dict[key] = state_dict1[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size, dtype)
        sampler.set_state(merged_state_dict)
        return sampler

    @classmethod
    def Subset(cls, sampler, inds):
        assert sampler.shape_len == 1

        device = sampler.device
        dtype = sampler.dtype
        chain_size_sub = (len(inds),)

        state_dict = sampler.get_state()

        sub_state_dict = dict()
        for key in state_dict:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states',
                       'rng_states'):
                sub_state_dict[key] = [state_dict[key][ind] for ind in inds]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                sub_state_dict[key] = state_dict[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size_sub, dtype)
        sampler.set_state(sub_state_dict)
        return sampler


class BMLP(nn.Module):
    """batched FF network for approximating functions"""

    def __init__(self, indim, hidden_dims, outdim, activation, shape, batch_rng, use_sigmoid=False):
        super().__init__()
        act_dict = dict(silu=nn.SiLU(), tanh=nn.Tanh(), relu=nn.ReLU(), lrelu=nn.LeakyReLU())
        self.layer_first = nn.ParameterList(
            self.make_linear(shape, indim, hidden_dims[0], batch_rng))
        layers_w, layers_b = [], []
        for idim, odim in zip(hidden_dims[:-1], hidden_dims[1:]):
            w, b = self.make_linear(shape, idim, odim, batch_rng)
            layers_w.append(w)
            layers_b.append(b)
        self.layer_hidden_w = nn.ParameterList(layers_w)
        self.layer_hidden_b = nn.ParameterList(layers_b)
        self.layer_last = nn.ParameterList(
            self.make_linear(shape, hidden_dims[-1], outdim, batch_rng))
        self.indim = indim
        self.outdim = outdim

        self.shape = shape
        self.ndim = len(shape)
        self.size = int(np.prod(shape))
        self.activation = act_dict[activation.lower()]
        self.use_sigmoid = use_sigmoid # added for MLP used for MNIST

    def forward(self, x):
        activation = self.activation
        assert x.shape[:self.ndim] == self.shape
        assert x.shape[-1] == self.indim
        bdims = self.shape
        hidden_w, hidden_b = self.layer_hidden_w, self.layer_hidden_b
        indim = self.indim
        x_middims = x.shape[self.ndim:-1]
        x_pts = int(np.prod(x_middims))
        # x.shape --> (n_bch, n_srf,  n_points,         d)

        u = x.reshape(*bdims,                x_pts, indim)
        # u.shape --> (n_bch, n_srf * n_points,         d)

        w, b = self.layer_first
        # w.shape --> (n_bch,                    d,   nn_width)
        # b.shape --> (n_bch,                    1,   nn_width)

        u = activation(torch.matmul(u, w) + b)
        # u.shape --> (n_bch, n_srf * n_points,   nn_width)

        for _, (w, b) in enumerate(zip(hidden_w, hidden_b)):
            u = activation(torch.matmul(u, w) + b)
        # u.shape --> (n_bch, n_srf * n_points,   nn_width)

        w, b = self.layer_last
        u = torch.matmul(u, w) + b
        # u.shape --> (n_bch, n_srf * n_points,     outdim)

        u = u.reshape(*x.shape[:-1], self.outdim)
        # u.shape --> (n_bch, n_srf,  n_points,     outdim)
        
        if self.use_sigmoid:
            u = torch.sigmoid(u)
        
        return u

    def make_linear(self, shape, indim, out_width, batch_rng):
        k = 1. / np.sqrt(indim).item()
        with torch.no_grad():
            w_unit = batch_rng.uniform((*shape, indim, out_width))
            b_unit = batch_rng.uniform((*shape,         1, out_width))
            w_tensor = w_unit * (2 * k) - k
            b_tensor = b_unit * (2 * k) - k
        w = torch.nn.Parameter(w_tensor)
        b = torch.nn.Parameter(b_tensor)
        return w, b