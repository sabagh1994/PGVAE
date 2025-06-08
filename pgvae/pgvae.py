import torch
from tch_utils import BMLP

# for sampling
from scipy.stats import chi2

# vae utilities (separate python script)
def copy_module(vae_type, encoder, decoder, n_seeds, 
                batch_rng, device, **kwargs):
    if vae_type == "mlp":
        indim, outdim = kwargs["indim"], kwargs["outdim"]
        mlp_hdim = kwargs["mlp_hdim"] #[64, 20]
        activation = kwargs["activation"]
        
        encoder_copy = BMLP(indim=indim, hidden_dims=mlp_hdim, outdim=outdim, 
                            activation=activation, shape=(n_seeds, ), batch_rng=batch_rng)
        decoder_copy = BMLP(indim=ldim, hidden_dims=mlp_hdim[::-1], outdim=indim, 
                            activation=activation, shape=(n_seeds, ), batch_rng=batch_rng)
        encoder_copy.to(device)
        decoder_copy.to(device)
        with torch.no_grad():
            enc_sd, dec_sd = encoder.state_dict(), decoder.state_dict()
            encoder_copy.load_state_dict(enc_sd)
            decoder_copy.load_state_dict(dec_sd)
    else:
        raise NotImplementedError
        
    return encoder_copy, decoder_copy

def encode(x, encoder):
    l = encoder(x)
    (n_seeds, xs, *xd) = x.shape
    
    ldim = int(l.shape[-1]/2)
    mu, logvar = l[...,:ldim], l[...,ldim:]
    assert mu.shape == (n_seeds, xs, ldim)
    assert logvar.shape == (n_seeds, xs, ldim)    

    return (mu, logvar)

def reparameterize(mu, logvar, batch_rng):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    (n_seeds, mus, *ldim) = mu.shape
    std = torch.exp(0.5 * logvar)
    eps = batch_rng.normal(sample_shape=std.shape)
    assert eps.shape == (n_seeds, mus, *ldim)
    sample = mu + eps * std
    assert sample.shape == (n_seeds, mus, *ldim)
    return sample

def forward(x, encoder, decoder, batch_rng):
    (n_seeds, xs, *xd) = x.shape
    mu, log_var = encode(x, encoder)
    z = reparameterize(mu, log_var, batch_rng=batch_rng)
    out = decoder(z)
    assert out.shape == (n_seeds, xs, *xd)
    return  (out, mu, log_var)

def compute_logpz(x, encoder):
    """
        compute the logp of encoded x
        under the standard normal
    """
    (n_seeds, xs, *xd) = x.shape
    mu, logvar = encode(x, encoder)
    ldim = mu.shape[-1]
    assert mu.shape == (n_seeds, xs, ldim)
    assert logvar.shape == (n_seeds, xs, ldim)
    # compute the log prob of mu under N(0,1)
    mu_logprob = -0.5*torch.sum(torch.square(mu), dim=-1)
    assert mu_logprob.shape == (n_seeds, xs)

    return mu_logprob

def compute_logpx(x, encoder, decoder, n_samples, batch_rng, **kwargs):
    """
    Computes log(P(x))
    num_samples: (int) number of z samples to compute log(p(x))

    """

    (n_seeds, xs, *xd) = x.shape
    z, mu, std = sample_condx(x, encoder, n_samples, batch_rng)
    ldim = std.shape[-1]
    
    assert (z.shape == (n_seeds, n_samples, xs, ldim))
    out = decoder(z)
    assert (out.shape == (n_seeds, n_samples, xs, *xd))
    assert std.shape == (n_seeds, xs, ldim)

    x_re = x.reshape((n_seeds, 1, xs, *xd))
    diff = x_re - out
    assert diff.shape == (n_seeds, n_samples, xs, *xd)
    logpx_condz = -0.5*torch.sum(diff**2, dim = -1) # shape (num_samples, N)
    assert logpx_condz.shape == (n_seeds, n_samples, xs)
    logpz = -0.5*torch.sum(z**2, dim = -1) # shape (num_samples, N)
    assert logpz.shape == (n_seeds, n_samples, xs)

    assert mu.shape == (n_seeds, xs, ldim)
    mu_re = mu.reshape((n_seeds, 1, xs, ldim))
    zdiffmu_sq = (z - mu_re)**2
    assert zdiffmu_sq.shape == (n_seeds, n_samples, xs, ldim)
    
    std_inverse = 1./std
    std_inverse = std_inverse.reshape((n_seeds, 1, xs, ldim))
    std_inverse.shape == (n_seeds, 1, xs, ldim)
    
    logqz_condx = -0.5*torch.sum(zdiffmu_sq*std_inverse.square(), dim=-1)
    assert (logqz_condx.shape == (n_seeds, n_samples, xs))

    std_expand = std.reshape((n_seeds, 1, xs, ldim))
    logqz_condx = logqz_condx - torch.sum(torch.log(std_expand), dim=-1) # shape (num_samples, N)
    assert (logqz_condx.shape == (n_seeds, n_samples, xs))

    log_total = logpx_condz + logpz - logqz_condx
    log_prob_x = torch.logsumexp(log_total, dim = 1) # shape (N)
    assert log_prob_x.shape == (n_seeds, xs)
    d_out = dict(log_prob_x=log_prob_x, logpx_condz=logpx_condz, 
                 logpz=logpz, logqz_condx=logqz_condx)

    return d_out


# sampling functions
def sample_condx(x, encoder, n_samples, batch_rng, **kwargs):
    """
    Generates samples close to x by taking z samples from q(z|x)

    x: (torch.tensor) the input to condition on q(z|x), (N,d)
    num_samples: (int) number of samples generated from q(z|x)
    out: (torch.tensor) the z samples generated (num_samples, N, d_latent)
         as well as the mean and std used for sampling z 

    """
    std_scale = kwargs.get("std_scale", 1.) # to sample with higher variance
    (n_seeds, xs, *xd) = x.shape
    
    mu, logvar = encode(x, encoder) # shape mu: N*d, shape logvar: (N, d_latent)
    ldim = mu.shape[-1]
    assert mu.shape == (n_seeds, xs, ldim)
    assert logvar.shape == (n_seeds, xs, ldim)
    std = torch.exp(0.5 * logvar)

    std = std * std_scale
    assert std.shape == (n_seeds, xs, ldim)
    eps = batch_rng.normal(sample_shape=(n_seeds, n_samples, xs, ldim))
    assert eps.shape == (n_seeds, n_samples, xs, ldim)
    std_re = std.reshape((n_seeds, 1, xs, ldim))
    mu_re = mu.reshape((n_seeds, 1, xs, ldim))
    z = mu_re + eps * std_re # shape (num_samples, N, d_latent)
    assert z.shape == (n_seeds, n_samples, xs, ldim)

    return z, mu, std

def sample_comprehensive(n_samples, encoder, decoder, 
                         batch_rng, sample_mode, ldim, 
                         device=None, dtype=None, **kwargs):
    assert sample_mode in ["random", "condx_normal", "condx_prior_scored"]
    
    # first: generate Zs in the latent space
    if sample_mode == "random":
        # take random samples from N(0, 1)
        
        method_name = kwargs["method_name"]
        n_seeds = len(batch_rng.rngs)
        z = batch_rng.normal(sample_shape=(n_seeds, n_samples, ldim))
        assert z.shape == (n_seeds, n_samples, ldim)

        # build prior logp score
        if method_name == "pgvae":
            y_max, y_target = kwargs["y_max"], kwargs["y_target"]
            temperature = kwargs["temperature"]
            
            assert y_max.shape == (n_seeds,)
            assert y_target.shape == (n_seeds,)
            
            rv = chi2(df=ldim)
            chisq = rv.isf(q=0.01) # 99% confidence interval
            diffy = y_max - y_target
            assert diffy.shape == (n_seeds,)

            temperature_d = 0.5*chisq/(diffy + 0.1)
            scale = (temperature/temperature_d)**0.5
            assert scale.shape == (n_seeds,)
            dim_exp = [1]*len(z.shape[1:])
            scale = scale.reshape((n_seeds, *dim_exp))
            z = z*scale
            assert z.shape == (n_seeds, n_samples, ldim)

    elif sample_mode == "condx_normal":
        # Generates samples close to x by taking z samples from q(z|x)
        x = kwargs["X"]
        (n_seeds, xs, *xd) = x.shape
        std_scale = kwargs.get("std_scale", 1.) # variance scale used for sampling

        if n_samples < xs:
            print(f"number of samples {n_samples} less than data size {xs}" \
                  "\nsetting n_samples = data size")
            n_samples = xs
            
        x_mask = torch.ones((n_seeds, xs), dtype=dtype, device=device)
        # construct the cdf
        assert x_mask.shape == (n_seeds, xs)
        x_mask_cdf = torch.cumsum(x_mask, dim=-1)
        assert x_mask_cdf.shape == (n_seeds, xs)
        x_max, _ = torch.max(x_mask_cdf, dim=-1, keepdim=True)
        assert x_max.shape == (n_seeds, 1)
        x_mask_cdf = x_mask_cdf/x_max # cap at 1
        assert x_mask_cdf.shape == (n_seeds, xs)
        q = torch.arange(1, n_samples+1, 1, dtype=dtype, device=device)/n_samples
        assert q.shape == (n_samples,)
        q = q.reshape(1, n_samples)
        q = q.expand(n_seeds, n_samples)
        xids = torch.searchsorted(x_mask_cdf, q)
        assert xids.shape == (n_seeds, n_samples)
        # getting the x around which sampling is performed
        dim_exp = [1]*len(xd)
        d_take = len(xd) + 1
        xids = xids.reshape((n_seeds, n_samples, *dim_exp))
        x = torch.take_along_dim(x, xids, dim=-d_take)
        assert x.shape == (n_seeds, n_samples, *xd)

        # generate one sample per x
        n_samples_upd = 1
        z, _, _ = sample_condx(x, encoder, n_samples_upd, 
                               batch_rng, std_scale=std_scale)
        assert (z.shape == (n_seeds, n_samples_upd, n_samples, ldim))
        z = z.reshape((n_seeds, n_samples_upd*n_samples, ldim))

    elif sample_mode == "condx_prior_scored":
        # map x to mu or mu_calib
        x = kwargs["X"]
        tr_unq_mask = kwargs["tr_unq_mask"]
        
        temperature = kwargs["temperature"]
        std_scale = kwargs.get("std_scale", 1.)
        
        (n_seeds, xs, *xd) = x.shape
        mu_logp_prior = compute_logpz(x, encoder)
        
        assert mu_logp_prior.shape == (n_seeds, xs)
        # assign very low probability to repetitive samples
        assert tr_unq_mask.dtype == torch.bool
        mu_logp_prior = mu_logp_prior*tr_unq_mask + (-10e3)*torch.abs(mu_logp_prior)*(~tr_unq_mask)
        
        max_logp_prior, _ = torch.max(mu_logp_prior, dim=-1, keepdim=True)
        assert max_logp_prior.shape == (n_seeds, 1)
        mu_logp_prior = mu_logp_prior - max_logp_prior
        assert mu_logp_prior.shape == (n_seeds, xs)

        # build prior logp score 
        y_max = kwargs["y_max"]
        y_target = kwargs["y_target"]
        assert y_max.shape == (n_seeds,)
        assert y_target.shape == (n_seeds,)
        
        rv = chi2(df=ldim)
        chisq = rv.isf(q=0.01) # 99% confidence interval
        diffy = y_max - y_target
        assert diffy.shape == (n_seeds,)

        temperature_d = 0.5*chisq/(diffy + 0.01*y_max)
        prior_logp_thr = -0.5*chisq*temperature/temperature_d
        assert prior_logp_thr.shape == (n_seeds,)
        prior_logp_thr = prior_logp_thr.reshape(n_seeds, 1)


        # only keep the mus or xs where the samples are not outliers
        mu_logp_prior, id_sort = torch.sort(mu_logp_prior, descending=True, dim=-1)
        assert id_sort.shape == (n_seeds, xs)
        assert mu_logp_prior.shape == (n_seeds, xs)
        dim_exp = [1]*len(xd)
        id_sort_x = id_sort.reshape((n_seeds, xs, *dim_exp))
        # sorting x
        d_take = len(xd) + 1
        x = torch.take_along_dim(x, id_sort_x, dim= -d_take)
        assert x.shape == (n_seeds, xs, *xd)
        # make the mask (only x passing the threshold can be used in sampling)
        x_mask = mu_logp_prior > prior_logp_thr
        assert x_mask.shape == (n_seeds, xs)
        # If the number of x passing threshold is greater
        # than the number of samples allowed for sampling
        
        if n_samples < xs:
            x_mask[:, n_samples:] = 0
        
        # construct the cdf
        assert x_mask.shape == (n_seeds, xs)
        x_mask_cdf = torch.cumsum(x_mask, dim=-1)
        assert x_mask_cdf.shape == (n_seeds, xs)
        x_max, _ = torch.max(x_mask_cdf, dim=-1, keepdim=True)
        assert x_max.shape == (n_seeds, 1)
        x_mask_cdf = x_mask_cdf/x_max # cap at 1
        assert x_mask_cdf.shape == (n_seeds, xs)
        q = torch.arange(1, n_samples+1, 1, dtype=dtype, device=device)/n_samples
        assert q.shape == (n_samples,)
        q = q.reshape(1, n_samples)
        q = q.expand(n_seeds, n_samples)
        xids = torch.searchsorted(x_mask_cdf, q)
        assert xids.shape == (n_seeds, n_samples)
        # getting the x around which sampling is performed
        xids = xids.reshape((n_seeds, n_samples, *dim_exp))
        x = torch.take_along_dim(x, xids, dim=-d_take)
        assert x.shape == (n_seeds, n_samples, *xd)
        
        n_sample_upd = 1
        z, _, _ = sample_condx(x, encoder, n_sample_upd, 
                               batch_rng, std_scale=std_scale)
        assert (z.shape == (n_seeds, n_sample_upd, n_samples, ldim))
        z = z.reshape((n_seeds, n_sample_upd*n_samples, ldim)) # shape (num_samples//N *N, latent_dim)
    else:
        raise NotImplementedError(f"undefined sample mode {sample_mode}")

    samples = decoder(z)
    return samples, z


# loss related functions
def relationship_constraint_latent(x, y, encoder, temperature):
    n_seeds, xs, *xd = x.shape
    assert y.shape == (n_seeds, xs)
    
    # compute the log prob of mu under N(0,1)
    mu_logprob = compute_logpz(x, encoder)
    assert mu_logprob.shape == (n_seeds, xs)
    pairwise_logprobmu_diff = mu_logprob.reshape(n_seeds, xs, 1) - mu_logprob.reshape(n_seeds, 1, xs) # shape (N, N)
    assert pairwise_logprobmu_diff.shape == (n_seeds, xs, xs)
    pairwise_y_diff = temperature*(y.reshape(n_seeds, xs, 1) - y.reshape(n_seeds, 1, xs))
    assert pairwise_y_diff.shape == (n_seeds, xs, xs)

    logpmudiff_ydiff_diff = (pairwise_logprobmu_diff - pairwise_y_diff)

    constraint = torch.sum(logpmudiff_ydiff_diff**2, dim=(-1, -2))
    assert constraint.shape == (n_seeds,)
    norm_term = xs*xs + xs
    constraint = constraint/norm_term

    return constraint

def loss_function(recons, x, y, mu, log_var, encoder, criterion, method_name,
                  temperature, weights = None, **kwargs):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    
    x: (torch.tensor) encoded protein sequences, images, ... (n_seeds, xs, *xd)
    y: (torch.tensor) scores associated with x (n_seeds, xs, 1)
    label: (torch.tensor) used in case of classification
    mu: (torch.tensor) encoded x in latent space
    logvar: (torch.tensor) logvar associated with mu
    criterion: (str) optimization criterion, "mse" (images) or "ce" (protein)
    method_name: (str) MBO method name, "pgvae", ...
    weights: (torch.tensor) weights used for weighted optimization (not used in pgvae)
    label: (torch.tensor) ground truth labels when criterion is "ce" e.g., proteins
    """
    
    (n_seeds, xs, *xd) = x.shape
    ldim = mu.shape[-1]
    assert recons.shape == (n_seeds, xs, *xd) # input and output have the same shape
    assert y.shape == (n_seeds, xs)
    kld_weight = kwargs['kld_weight'] # Account for the minibatch samples from the dataset
    
    
    if criterion == "ce":
        # classification task so label should be provided
        vae_criterion = nn.CrossEntropyLoss(reduction="none")

        # num_cls is the same as the number of amino acids (onehot)
        num_cls, seq_len = kwargs['num_cls'], kwargs['seq_len']
        
        # label has the class indices, e.g., aa indices
        label = x.reshape((n_seeds, xs, seq_len, num_cls))
        _, label = torch.max(label, dim=-1)
        assert label.shape == (n_seeds, xs, seq_len)
        label = label.reshape((n_seeds*xs, seq_len))
        assert label.shape == (n_seeds*xs, seq_len)

        recons = recons.reshape((n_seeds, xs, seq_len, num_cls))
        recons = recons.transpose(dim0=-1, dim1=-2)
        assert recons.shape == (n_seeds, xs, num_cls, seq_len)
        recons = recons.reshape((n_seeds*xs, num_cls, seq_len))
        assert recons.shape == (n_seeds*xs, num_cls, seq_len)
        recons_loss = vae_criterion(recons, label)
        assert recons_loss.shape == (n_seeds*xs, seq_len)
        recons_loss = recons_loss.reshape((n_seeds, xs, seq_len))
        recons_loss = torch.mean(recons_loss, dim=-1)
        assert recons_loss.shape == (n_seeds, xs)
    elif criterion == "mse":
        # images and toy examples
        diff = recons - x
        assert diff.shape == (n_seeds, xs, *xd)
        mean_dims = tuple(range(2, 2+len(xd))) # mean over xds
        recons_loss = torch.mean(diff**2, dim=mean_dims)
        assert recons_loss.shape == (n_seeds, xs)
    
    mu_loss, var_loss = torch.tensor(-1), torch.tensor(-1) # only pgvae has these terms
    if method_name == "pgvae":
        recons_loss = torch.mean(recons_loss, dim=-1)
        assert recons_loss.shape == (n_seeds,)
        mu_weight = kld_weight
        # separating variance and mu in the kl loss
        var_term = 1 + log_var - log_var.exp()
        assert var_term.shape == (n_seeds, xs, ldim)
        var_loss = -0.5 * torch.mean(torch.mean(var_term, dim = -1), dim = -1)
        assert var_loss.shape == (n_seeds,)
        mu_term = mu**2
        assert mu_term.shape == (n_seeds, xs, ldim)
        mu_loss = torch.mean(0.5*torch.mean(mu_term, dim=-1), dim=-1)
        assert mu_loss.shape == (n_seeds,)
        kld_loss = kld_weight * var_loss + mu_weight * mu_loss
        assert kld_loss.shape == (n_seeds,)
        loss = recons_loss + kld_loss
    else:
        assert weights.shape == (n_seeds, xs)
        recons_loss = weights*recons_loss
        assert recons_loss.shape == (n_seeds, xs)
        recons_loss = torch.mean(recons_loss, dim=-1)
        assert recons_loss.shape == (n_seeds,)
        
        assert mu.shape == (n_seeds, xs, ldim)
        kl_term = 1 + log_var - mu ** 2 - log_var.exp()
        assert kl_term.shape == (n_seeds, xs, ldim)
        weights = weights.reshape((n_seeds, xs, 1))
        assert weights.shape == (n_seeds, xs, 1)
        kl_term = weights * kl_term
        assert kl_term.shape == (n_seeds, xs, ldim)
        kld_loss = -0.5 * torch.mean(torch.mean(kl_term, dim = -1), dim = -1)
        kld_loss = kld_weight * kld_loss
        assert kld_loss.shape == (n_seeds,)
        loss = recons_loss + kld_loss
        
    assert loss.shape == (n_seeds,)
    # add the relationship constraint
    rel_weight = kwargs["rel_weight"]
    rel_const = relationship_constraint_latent(x, y, encoder, temperature)
    assert rel_const.shape == (n_seeds,)
    rel_weight_norm = rel_weight/((temperature+1)**2)
    rel_const = rel_weight_norm * rel_const
    loss = loss + rel_const
    assert loss.shape == (n_seeds,)

    out_dict = {'loss': loss, 'RCL': recons_loss, 
                'KLD': kld_loss, 'RELC': rel_const, 
                'mu_loss': mu_loss, 'var_loss': var_loss}

    return out_dict
