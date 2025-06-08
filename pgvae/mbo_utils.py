import torch
# for weight generation
from torch.distributions.normal import Normal

# MBO utilities:
# weight generator function
def weight_generator(y, method_name, criterion, **kwargs):
    """
        y: (torch.tensor) scored predicted by oracle
        method_name: (str) name of the mbo method ("cbas", "dbas", "cem-pi","rwr")
        criterion: (str) optimization criterion ("mse", "mse")
        device: (torch.device) the device to do the operations on
    """
    
    assert method_name in ("dbas", "cem-pi", "rwr", "cbas")
    assert criterion in ("mse", "ce")
    (n_seeds, ys) = y.shape
    
    if method_name == "dbas":
        q_dbas = kwargs["q_dbas"]
        sigma_y_noise = kwargs["sigma_y_noise"]
        y_star = kwargs["y_star"] #init with -np.inf*torch.ones((n_seeds, 1), device=device)
        assert y_star.shape == (n_seeds, 1)
        
        y_star_1 = torch.quantile(y, q_dbas, dim=-1, keepdim=True)
        assert y_star_1.shape == (n_seeds, 1)
        
        #if y_star_1 > y_star: # define y_star = -np.inf
        #    y_star = y_star_1
        mask = y_star_1 > y_star
        y_star = y_star_1*mask + y_star*(~mask)
        assert y_star.shape == (n_seeds, 1)
        
        distr = Normal(loc=y, scale=sigma_y_noise)
        weights_gen = 1. - distr.cdf(y_star)
        assert weights_gen.shape == (n_seeds, ys)
        
        if not(torch.all(torch.isfinite(weights_gen))):
            print(f"nan or inf detected in weights for {method_name}")
            # setting nan/inf weights to one is helping other methods 
            weights_gen = torch.nan_to_num(weights_gen, nan=1., posinf=1., neginf=1.)
        assert weights_gen.shape == (n_seeds, ys)
        
    elif method_name == "rwr":
        rwr_alpha = kwargs["rwr_alpha"]
        weights_gen = torch.exp(rwr_alpha*y)
        assert weights_gen.shape == (n_seeds, ys)
        norm_weight = torch.sum(weights_gen, dim=-1, keepdim=True)
        assert norm_weight.shape == (n_seeds, 1)
        weights_gen = weights_gen/norm_weight
        assert weights_gen.shape == (n_seeds, ys)
        
        if not(torch.all(torch.isfinite(weights_gen))):
            print(f"nan or inf detected in weights for {method_name}")
            # setting nan/inf weights to one is helping other methods 
            weights_gen = torch.nan_to_num(weights_gen, nan=1., posinf=1., neginf=1.)
        assert weights_gen.shape == (n_seeds, ys)
        
    elif method_name == "cem-pi":
        sigma_y_noise = kwargs["sigma_y_noise"]
        max_tr_y = kwargs["max_tr_y"] # this is a scalar
        q_cem_pi = kwargs["q_cem_pi"]
        
        distr = Normal(loc=y, scale=sigma_y_noise)
        max_tr_y_re = max_tr_y.reshape(n_seeds, 1)
        pi = 1. - distr.cdf(max_tr_y_re)
        assert pi.shape == (n_seeds, ys)
        pi_thresh = torch.quantile(pi, q_cem_pi, dim=-1, keepdim=True)
        assert pi_thresh.shape == (n_seeds, 1)
        weights_gen = (pi > pi_thresh).type(torch.int)
        assert weights_gen.shape == (n_seeds, ys)
        
        if not(torch.all(torch.isfinite(weights_gen))):
            print(f"nan or inf detected in weights for {method_name}")
            # setting nan/inf weights to one is helping other methods 
            weights_gen = torch.nan_to_num(weights_gen, nan=1., posinf=1., neginf=1.)
        assert weights_gen.shape == (n_seeds, ys)
    
    elif method_name == "cbas":
        x = kwargs["x"] # in the encoded format (ds_create is called already)
        encoder, encoder_prior = kwargs["encoder"], kwargs["encoder_prior"]
        decoder, decoder_prior = kwargs["decoder"], kwargs["decoder_prior"]
        q_cbas, sigma_y_noise = kwargs["q_cbas"], kwargs["sigma_y_noise"]
        y_star = kwargs["y_star"] #init with -np.inf*torch.ones((n_seeds, 1), device=device)
        brng = kwargs["batch_rng"]
        
        (n_seeds, xs, *xd) = x.shape # e.g., one-hot encoded for protein
        assert xs == ys
        assert y_star.shape == (n_seeds, 1)
        
        if criterion == "ce":
            num_cls, seq_len = kwargs["num_cls"], kwargs["seq_len"]
            
            x_hat, _, _ = forward(x, encoder, decoder, brng)
            x_hat_prior, _, _ = forward(x, encoder_prior, decoder_prior, brng)
            
            x_hat = x_hat.reshape((n_seeds, xs, seq_len, num_cls))
            x_hat_prior = x_hat_prior.reshape((n_seeds, xs, seq_len, num_cls))
            
            # get the sum of probabilities
            logsum_probs = torch.logsumexp(x_hat, dim=-1)
            assert logsum_probs.shape == (n_seeds, xs, seq_len)
            logsum_probs_prior = torch.logsumexp(x_hat_prior, dim=-1)
            assert logsum_probs_prior.shape == (n_seeds, xs, seq_len)
            
            max_x_hat, _ = torch.max(x_hat, dim=-1)
            max_x_hat_prior, _ = torch.max(x_hat_prior, dim=-1)
            assert max_x_hat.shape == (n_seeds, xs, seq_len)
            assert max_x_hat_prior.shape == (n_seeds, xs, seq_len)
            
            logpxt = torch.sum(max_x_hat - logsum_probs, dim=-1)
            logpx0 = torch.sum(max_x_hat_prior - logsum_probs_prior, dim=-1)
            assert logpxt.shape == (n_seeds, xs)
            assert logpx0.shape == (n_seeds, xs)
            
            w1 = torch.exp(logpx0 - logpxt)
            if not(torch.all(torch.isfinite(w1))):
                print(f"nan or inf detected in w1 for {method_name}")
                # setting nan/inf weights to one is helping other methods 
                w1 = torch.nan_to_num(w1, nan=1.0, posinf=1.0, neginf=1.0)
                #w1 = torch.nan_to_num(w1, nan=0.0, posinf=0.0, neginf=0.0)
            assert w1.shape == (n_seeds, xs)

            y_star_1 = torch.quantile(y, q_cbas, dim=-1, keepdim=True)
            assert y_star_1.shape == (n_seeds, 1)

            mask = y_star_1 > y_star
            y_star = y_star_1*mask + y_star*(~mask)
            assert y_star.shape == (n_seeds, 1)
            
            distr = Normal(loc=y, scale=sigma_y_noise) # to be corrected
            w2 = 1. - distr.cdf(y_star)
            if not(torch.all(torch.isfinite(w2))):
                print(f"nan or inf detected in w2 for {method_name}")
                # setting nan/inf weights to one is helping other methods 
                w2 = torch.nan_to_num(w2, nan=1.0, posinf=1.0, neginf=1.0)
                #w2 = torch.nan_to_num(w2, nan=0.0, posinf=0.0, neginf=0.0)
            assert w2.shape == (n_seeds, xs)

            weights_gen = w1*w2
            assert weights_gen.shape == (n_seeds, xs)
        
        elif criterion == "mse":
            x_hat, _, _ = forward(x, encoder, decoder, brng)
            x_hat_prior, _, _ = forward(x, encoder_prior, decoder_prior, brng)
            
            diff = x - x_hat
            dim_reduc = tuple(range(2, 2+len(xd)))
            logpxt = -0.5*torch.sum(diff**2, dim=dim_reduc)
            assert logpxt.shape == (n_seeds, xs)
            
            diff_prior = x - x_hat_prior
            logpx0 = -0.5*torch.sum(diff_prior**2, dim=dim_reduc)
            assert logpx0.shape == (n_seeds, xs)
            
            w1 = torch.exp(logpx0 - logpxt)
            y_star_1 = torch.quantile(y, q_cbas, dim=-1, keepdim=True)
            assert y_star_1.shape == (n_seeds, 1)

            mask = y_star_1 > y_star
            y_star = y_star_1*mask + y_star*(~mask)
            assert y_star.shape == (n_seeds, 1)
            
            distr = Normal(loc=y, scale=sigma_y_noise)
            w2 = 1. - distr.cdf(y_star)
            assert w2.shape == (n_seeds, xs)
            weights_gen = w1*w2

            if not(torch.all(torch.isfinite(weights_gen))):
                print(f"nan or inf detected in w2 for {method_name}")
                # setting nan/inf weights to one is helping other methods 
                weights_gen = torch.nan_to_num(weights_gen, nan=1.0, posinf=1.0, neginf=1.0)
            assert weights_gen.shape == (n_seeds, xs)
            
        else:
            raise NotImplementedError("invalid criterion")
            
    else: 
        raise NotImplementedError(f"{method_name} is not supported")
        
    out = y_star if method_name in ("cbas", "dbas") else None
    return weights_gen, out
