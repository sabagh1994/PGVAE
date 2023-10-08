# Dataset required packages and functions:
import os
import torch
import numpy as np
from encoding_utils import encode_sequence, convert_idx_to_aas


# functions required for duplicated samples in the dataset:
# Sample duplication in the entire dataset may happen 
# as the samples generated in each round of MBO may have overlap
# with preexsiting samples.

def apply_unique(prtn_oh, device, tch_dtype):
    """
    Pre-processes the input, adds bias to differentiate each seed from 
    the others, and then applies the torch.unique function.
    
    Parameters
    ----------
    prtn_oh: (torch.tensor) one-hot encoded protein with the 
        shape `(n_seeds, n_samps, seq_len, n_aa)`.
    
    Outputs
    -------
    prtinv: (torch.tensor) indicates where elements in the original 
        input ended up in the unique list. This has a shape of 
        `(n_seedsamps,)`.
    
    prtcnt: (torch.tensor) The count of each unique protein. This has 
        a shape of `(n_unq,)`.

    """

    # To prevent unique sequences from different seeds getting mixed 
    # together, we will add a bias to each seed, so that the range of 
    # their values would not be the same.
    
    #n_seeds, n_samps, seq_len, n_aa = prtn_oh.shape
    n_seeds, n_samps, *nd = prtn_oh.shape
    dim_ext = tuple([1]*len(nd))
    prtn_oh_re = prtn_oh.reshape((n_seeds, -1))
    prtn_oh_min = torch.min(prtn_oh_re, dim=-1).values
    prtn_oh_min = prtn_oh_min.reshape((n_seeds, 1, *dim_ext))
    prtn_oh = prtn_oh - prtn_oh_min
    assert prtn_oh.shape == (n_seeds, n_samps, *nd)
    
    prtn_max = prtn_oh.max()
    assert prtn_max.shape == tuple()
    sbias1d = torch.arange(n_seeds, device=device, dtype=tch_dtype)
    assert sbias1d.shape == (n_seeds,)
    #sbias = sbias1d.reshape(n_seeds, 1, 1, 1) * (prtn_max + 1)
    #assert sbias.shape == (n_seeds, 1, 1, 1)
    sbias = sbias1d.reshape(n_seeds, 1, *dim_ext) * (prtn_max + 1)
    assert sbias.shape == (n_seeds, 1, *dim_ext)
    prtn_oph = prtn_oh + sbias
    #assert prtn_oph.shape == (n_seeds, n_samps, seq_len, n_aa)
    assert prtn_oph.shape == (n_seeds, n_samps, *nd), f"{prtn_oph.shape}, {nd}, {prtn_oh.shape}"

    # Now, we will reshape the protein seqs into a 2-d tensor
    #prtn_2doph = prtn_oph.reshape(n_seeds * n_samps, seq_len * n_aa)
    #assert prtn_2doph.shape == (n_seeds * n_samps, seq_len * n_aa)
    prtn_2doph = prtn_oph.reshape(n_seeds * n_samps, np.prod(nd))
    assert prtn_2doph.shape == (n_seeds * n_samps, np.prod(nd))

    # The `sorted=True` argument ensure that the unique groups remain together
    prtunq, prtinv, prtcnt = torch.unique(prtn_2doph, dim=0, sorted=True, 
                                          return_inverse=True, return_counts=True)
    n_unq = prtcnt.shape[0]
    #assert prtunq.shape == (n_unq, seq_len * n_aa)
    assert prtunq.shape == (n_unq, np.prod(nd))
    assert prtcnt.shape == (n_unq,)
    assert prtinv.shape == (n_seeds * n_samps,)
    
    return prtinv, prtcnt

def get_unique_mask(prtinv, prtcnt):
    """
        Generates mask for unique protein samples
        
        Parameters
        ----------
        prtinv: (torch.tensor) indicates where elements in the original 
            input ended up in the unique list. This has a shape of `(n_seedsamps,)`.
        
        prtcnt: (torch.tensor) The count of each unique protein. This has 
            a shape of `(n_unq,)`.
    
    
        Outputs
        -------
        mask1d: (torch.tensor) dtype (torch.bool) `(n_seedsamps,)`.
        
    """
    n_seedsamps, = prtinv.shape
    n_unq, = prtcnt.shape
    
    # The following is only necessary for creating the "first occurance boolean mask".
    #prtinvind_sorted = torch.argsort(prtinv, stable=True) # stable is in torch version 1.13.0
    pp = prtinv + torch.arange(n_seedsamps, device=prtinv.device)/n_seedsamps # same functionality as stable
    prtinvind_sorted = torch.argsort(pp)
    assert prtinvind_sorted.shape == (n_seedsamps,)
    prtcnt_cs = prtcnt.cumsum(dim=0)
    assert prtcnt_cs.shape == (n_unq,)
    prtcnt_cszp = torch.cat([prtcnt.new_zeros(1), prtcnt_cs[:-1]])
    assert prtcnt_cszp.shape == (n_unq,)
    mask_idx = prtinvind_sorted[prtcnt_cszp].sort().values
    assert mask_idx.shape == (n_unq,)
    mask1d = torch.zeros(n_seedsamps, device=prtinv.device, dtype=torch.bool)
    mask1d[mask_idx] = 1
    assert mask1d.shape == (n_seedsamps,)
#     mask = mask1d.reshape(n_seeds, n_samps)
#     assert mask.shape == (n_seeds, n_samps)

    return mask1d

def get_unique_weight(prtinv2d, prtcnt):
    """
        Generates weights for the samples associated with 
        each unique sample, e.g., if there are n occurances
        of sample A then each one gets 1/n as weight.
        
        Parameters
        ----------
        prtinv2d: (torch.tensor) shape (n_seeds, n_samps)
        
        Output
        ------
        smp_w_unnorm: (torch.tensor) weights for each unique sample 
                shape (n_seeds, n_samps)
        n_smpunq: (torch.tensor) number of unique samples in each seed (n_seeds,)
        pi_min: (torch.tensor) the shift for unique class index (n_seeds,). This is
                due to flattening the tensor before using torch.unique
     
    """

    # Making sure that the seed groups remain together
    #prtinv2d = prtinv.reshape(n_seeds, n_samps)
    n_seeds, n_samps = prtinv2d.shape
    prtinv = prtinv2d.reshape((n_seeds*n_samps,))
    
    pi_min = prtinv2d.min(dim=1).values
    pi_max = prtinv2d.max(dim=1).values
    
    assert (pi_max[:-1] == (pi_min[1:]-1)).all()
    assert (pi_max[:-1]  <  pi_max[1:]   ).all()
    assert (pi_min[:-1]  <  pi_min[1:]   ).all()

    n_smpunq = (pi_max - pi_min) + 1
    assert n_smpunq.shape == (n_seeds,)
    assert (n_smpunq <= n_samps).all()

    # Simply, `smp_cnt_` will show how many times each sequence 
    # in prtn_2doph was repeated! 
    smp_cnt_ = prtcnt[prtinv]
    assert smp_cnt_.shape == (n_seeds * n_samps,)
    smp_cnt = smp_cnt_.reshape(n_seeds, n_samps)
    assert smp_cnt.shape == (n_seeds, n_samps)
    smp_w_unnorm = 1.0 / smp_cnt
    assert smp_w_unnorm.shape == (n_seeds, n_samps)
    assert torch.allclose(smp_w_unnorm.sum(dim=1), n_smpunq.float())
    
    return smp_w_unnorm, n_smpunq, pi_min

def get_split_weights(smp_w_unnorm, n_smpunq, pi_min, prtinv2d, split_props, device):
    """
    Parameters
    ----------
    split_props: proportion of different splits of the data, e.g., [0.7, 0.2, 0.1]
    
    smp_w_unnorm: weights for the unique samples with shape (n_seeds, n_samps)
    
    
    Output
    ------
    all_smpw: (list) a list of torch tensors with shape (n_seeds, n_samps)
            for splits of the data (train, test, val). In each split,
            only the samples associated with it have nonzero weights.
        
    """
    
    n_split = len(split_props)
    q_props_ = np.cumsum([0] + list(split_props))
    assert q_props_.shape == (n_split+1,)
    q_props = q_props_ / q_props_[-1]
    assert q_props.shape == (n_split+1,)

    #######################################################################
    ############# Performing the Train/Validation/Test Splits #############
    #######################################################################

    # If there was a single split, we would have been done by just 
    # normalizing the sample weights so that they sum to one for each seed.
    # 
    # if n_split == 1:
    #     smp_w = smp_w_unnorm / smp_w_unnorm.sum(dim=1, keepdim=True)
    #     assert smp_w.shape == (n_seeds, n_samps)
    #     all_smpw = [smp_w]

    n_seeds, n_samps = smp_w_unnorm.shape
    
    spprop_np = np.array([0] + list(split_props), dtype=np.float32)
    spprop_np = spprop_np.cumsum() / spprop_np.sum()
    q_props_ = torch.tensor(spprop_np).to(device=device)
    assert q_props_.shape == (n_split+1,)
    q_props = q_props_.reshape(1, n_split+1)
    assert q_props.shape == (1, n_split+1)

    aa = n_smpunq.unsqueeze(dim=1)
    assert aa.shape == (n_seeds, 1)
    cls_idxs = (q_props * aa).long()
    assert cls_idxs.shape == (n_seeds, n_split+1)
    all_smpw_unnorm = []
    for i_split in range(n_split):
        ll = cls_idxs[:, i_split] + pi_min
        assert ll.shape == (n_seeds,)
        hh = cls_idxs[:, i_split+1] + pi_min
        assert hh.shape == (n_seeds,)

        ll2d = ll.unsqueeze(dim=-1)
        assert ll2d.shape == (n_seeds, 1)
        hh2d = hh.unsqueeze(dim=-1)
        assert hh2d.shape == (n_seeds, 1)
        smpw_split_unnorm = smp_w_unnorm * (prtinv2d >= ll2d) * (prtinv2d < hh2d)
        assert smpw_split_unnorm.shape == (n_seeds, n_samps)
        all_smpw_unnorm.append(smpw_split_unnorm)

    tot_smp_w = smp_w_unnorm / smp_w_unnorm.sum(dim=1, keepdim=True)
    assert tot_smp_w.shape == (n_seeds, n_samps)

    aa = torch.stack(all_smpw_unnorm, dim=2).sum(dim=2)
    smp_w_consrvd = torch.allclose(aa, smp_w_unnorm)
    assert smp_w_consrvd

    all_smpw = [x / x.sum(dim=1, keepdims=True) for x in all_smpw_unnorm]
    
    return all_smpw

def weighted_shuffle_v0(w, n_q, device, tch_gen):
    (n_seeds, n_samps) = w.shape
    i_shuff = torch.multinomial(w, num_samples=n_q, generator=tch_gen)
    i_shuff = i_shuff.to(device)
    assert i_shuff.shape == (n_seeds, n_samps)
    return i_shuff

def weighted_shuffle(w, n_q, samp_strat, rng, device, tch_dtype):
    """
        generates n_q shuffled indices from a dataset with repeated samples (weights)
        for training n_q should be equal to training size
        sample_strat: iid, det
        
        n_q: number of samples to be taken
        out: i_shuff (n_seeds, n_q) => n_q is the partitions of the y axis (inverse cdf)
    
    """
    
    (n_seeds, n_samps) = w.shape

    urv = rng.uniform((n_seeds, n_samps))
    assert urv.shape == (n_seeds, n_samps)
    ui = urv.argsort(dim=-1)
    assert ui.shape == (n_seeds, n_samps)

    w_shuff = torch.take_along_dim(w, ui, dim=-1).contiguous()
    assert w_shuff.shape == (n_seeds, n_samps)

    cdf_shuff = w_shuff.cumsum(dim=1)
    assert cdf_shuff.shape == (n_seeds, n_samps)

    if samp_strat == 'det':
        q_o_ = torch.arange(n_q, device=device, dtype=tch_dtype) / n_q
        assert q_o_.shape == (n_q,)

        q_o = q_o_.reshape(1, n_q).expand(n_seeds, n_q)
        assert q_o.shape == (n_seeds, n_q)

        qii = rng.uniform((n_seeds, n_q)).argsort(dim=1)
        assert qii.shape == (n_seeds, n_q)

        q = torch.take_along_dim(q_o, qii, dim=1)
        assert q.shape == (n_seeds, n_q)
    elif samp_strat == 'iid':
        q = rng.uniform((n_seeds, n_q)).contiguous()
        assert q.shape == (n_seeds, n_q)
    else:
        raise ValueError(f'not implemented!')
        
    ii_shuff = torch.searchsorted(cdf_shuff, q, right=True)
    assert ii_shuff.shape == (n_seeds, n_q)

    i_shuff = torch.take_along_dim(ui, ii_shuff, dim=-1)
    assert i_shuff.shape == (n_seeds, n_q)
    
    return i_shuff
    
def get_unique_quantile(y_ds, unq_mask, q, tch_device):
    """
        Get the quantile of the y for the unique samples
        
        Parameters
        ----------
        y_ds: (torch.tensor) (ns, xs)
        
        unq_mask: (torch.tensor) mask specifying where the 
                unique values are located(ns, xs)
        
        q: (scalar) quantile in range (0,1]
        
    """
    
    ns, xs = y_ds.shape
    assert y_ds.shape == unq_mask.shape
    
    if q == 1:
        y_target = torch.max(y_ds, dim=-1)[0]
        assert y_target.shape == (ns,)
    else:
        y_tr = y_ds*unq_mask # unique train scores (y)
        assert y_tr.shape == (ns, xs)

        #y_target = torch.quantile(y_tr, q=.99, dim=-1) # attention: for pgvae this should be only unique samples (fix later)
        #assert y_target.shape == (n_seeds,)

        # computing the quantile of train activity scores for pgvae sampling
        # this is done by computing weighted cdf of the train data
        idx = torch.argsort(y_tr, dim=-1)

        norm = torch.sum(unq_mask, dim=-1, keepdim=True)
        assert norm.shape == (ns, 1)
        unq_mask_n = unq_mask/norm # should sum to one for cdf

        y_tr_sorted = torch.take_along_dim(y_tr, idx, dim=-1)
        unq_mask_n_sorted = torch.take_along_dim(unq_mask_n, idx, dim=-1)
        assert y_tr_sorted.shape == (ns, xs)
        assert unq_mask_n_sorted.shape == (ns, xs)

        unq_mask_cdf = torch.cumsum(unq_mask_n_sorted, dim=-1)
        assert unq_mask_cdf.shape == (ns, xs)

        qt = q*torch.ones((ns, 1), device=tch_device)
        ans_r = torch.searchsorted(unq_mask_cdf, qt, right=False)
        ans_l = torch.searchsorted(unq_mask_cdf, qt, right=True)
        assert ans_r.shape == (ns, 1)
        assert ans_l.shape == (ns, 1)
        qt_r= torch.take_along_dim(y_tr_sorted, ans_r, dim=-1)
        qt_l = torch.take_along_dim(y_tr_sorted, ans_l, dim=-1)
        y_target = (qt_r + qt_l)/2
        assert y_target.shape == (ns, 1)
        y_target = y_target.reshape((ns,))

    return y_target

def save_ds(ds, path, **kwargs):
    ds_save = {}
    for k in ds.keys():
        t = ds[k]
        if not(t is None):
            t = t.detach().to("cpu")
        ds_save[k] = t

    for k in kwargs.keys():
        ds_save[k] = kwargs[k]
    
    torch.save(ds_save, path)

class Dataset():
    def __init__(self, datadir=None, data_type="protein", 
                 dtype=torch.float32, device=None, 
                 split_rng=None, batch_rng=None, 
                 split_ratios=[0.8,0.1,0.1], 
                 add_w_optm=None, name="", max_hist=None, shuffle=True):
        """
            load the protein dataset from datadir
            data consists of protein sequences and their 
            measured property, e.g., fitness
        """
        self.name = name
        self.device = device
        self.dtype = dtype
        self.split_rng = split_rng # to split into train,test,val
        self.batch_rng = batch_rng # to sample batch of data for a batch of models
        self.data_type = data_type
        self.start_trav_ind = 0 # traversing the data
        self.add_w_optm = add_w_optm
        self.n_seeds, = batch_rng.shape # can be retrieved from batch_rng
        self.max_hist = max_hist # subsetting dsall with max history of max_hist
        
        data = self.load(datadir)
        # dsall contains all samples generated so far
        self.dsall = self.create_ds(data, add_w_optm, step=-1) # setting self.dsall
        
        # shuffle self.dsall
        if shuffle:
            ns, xs, *xd = self.dsall["x"].shape
            urv = self.batch_rng.uniform(sample_shape=(ns, xs))
            urv = torch.broadcast_to(urv, (ns, xs))
            assert urv.shape == (ns, xs)
            # range of uas values (0, n_tr)
            uas = urv.argsort(dim=-1)
            assert uas.shape == (ns, xs)
            for k in self.dsall.keys():
                if self.dsall[k] is None:
                    continue
                ns_k, xs_k, *xd_k = self.dsall[k].shape
                dim_ext = tuple([1]*len(xd_k))
                uas_ = uas.reshape((ns, xs, *dim_ext))
                self.dsall[k] = torch.take_along_dim(self.dsall[k], uas_, dim=1)          
        
        self.split(split_ratios) # setting self.all_smpw (list of weights for each partition)
        
    def load(self, datadir):
        """
            load the data and create a dataset
            datadir: (str) path to the data
            return: (dict) 
        """
        if not os.path.exists(datadir):
            raise Exception(f"{datadir} does not exist")
        
        data = np.load(datadir, allow_pickle=True)
        assert ("x" in data.keys()) and ("y") in data.keys(), "wrong data format!"
        return data
    
    @property
    def ds(self):
        """
            this is intended for the methods that
            work with a subset of dsall unlike pgvae
        """
        dssub = {}
        for k in self.dsall.keys():
            v = self.dsall[k]
            if v is None:
                dssub[k] = v
                continue
            ns, vs, *vd = v.shape
            if (self.max_hist is None) or (self.max_hist > vs):
                max_hist = vs
            else:
                max_hist = self.max_hist
            dssub[k] = v[:, -max_hist:, ...]

        return dssub
    
    def create_ds(self, data, add_w_optm, step=-1):
        """
            create the dataset used for training
            data: (dict) contains samples (x) e.g., protein strings and their associated property values (y)
            add_optm_w: (boolean) whether to add optimization weights to the dataset. The default is None 
            step: (int) MBO step in creating the dataset, -1 is associated with the starting dataset loaded from file
                  in numpy format
            return x_enc_dict: (dict) contains onehot and index encoded sequences 
                                as well as the property values (y)
        """
        assert ("x" in data.keys()) and ("y") in data.keys(), "data is incomplete!"
        x_enc_dict = self.encode(data["x"], step=step) # different encodings
        ns, xs, *xd = x_enc_dict["x"].shape

        y = data["y"]
        if step == -1:
            y = data["y"].reshape(1, -1) # adding scores
            assert isinstance(y, np.ndarray)
            y = np.broadcast_to(y, (self.n_seeds, y.size))
        assert y.shape == (ns, xs)
        
        x_enc_dict["y"] = y
            
        for k in x_enc_dict.keys():
            dtype_ = self.dtype
            if k == "x_index":
                dtype_ = torch.int64
            x_enc_dict[k] = torch.tensor(x_enc_dict[k], dtype=dtype_, device=self.device) \
                            if not(isinstance(x_enc_dict[k], torch.Tensor)) \
                            else x_enc_dict[k].to(dtype=dtype_, device=self.device)

        # adding weights for weighted optimization
        x_enc_dict["w_optm"] = None
        if add_w_optm:
            ns, xs, *xd = x_enc_dict["x"].shape
            if "w_optm" in data.keys():
                w_optm = data["w_optm"]
                w_optm = torch.tensor(w_optm, dtype=self.dtype, device=self.device) \
                                      if not(isinstance(w_optm, torch.Tensor)) \
                                      else w_optm.to(dtype=self.dtype, device=self.device)    
            else:
                w_optm = torch.ones((ns, xs), dtype=self.dtype, device=self.device)
            assert w_optm.shape == (ns, xs)
            x_enc_dict["w_optm"] = w_optm
            
        # adding the step in which the data is generated
        x_enc_dict["step"] = step*torch.ones((ns, xs), device=self.device)
        
        return x_enc_dict
        
    def encode(self, x, step=-1):
        """
            converts protein strings to one-hot or index encoded arrays
            x: (list or array) contains strings of protein sequences
            return x_enc_dict: (dict) onehot and index enoded arrays 
        """
        
        if self.data_type == "protein":
            if step == -1:
                # the data is not repeated for multiple seeds
                # x is a list/array of sequences
                x_onehot = encode_sequence(x, "one-hot", dataset_name="", lm_bsz= 64)
                org_shape = x_onehot.shape
                x_onehot = x_onehot.reshape((1, *org_shape))
                x_onehot = np.broadcast_to(x_onehot, (self.n_seeds, *org_shape))
            else:
                x_onehot_lst = []
                for it in x:
                    it_onehot = encode_sequence(it, "one-hot", dataset_name="", lm_bsz= 64)
                    it_s, *it_d = it_onehot.shape
                    it_onehot = it_onehot.reshape((1, it_s, *it_d))
                    x_onehot_lst.append(it_onehot)
                x_onehot = np.concatenate(x_onehot_lst, axis=0)
                assert x_onehot.shape == (self.n_seeds, it_s, *it_d)
                
            x_enc_dict = {"x": x_onehot}
        else:
            if step == -1:
                org_shape = x.shape
                x = x.reshape((1, *org_shape))
                x = np.broadcast_to(x, (self.n_seeds, *org_shape))
            if x.dtype == np.uint8:
                print("np.uint8 dtype for x")
                x = x/255
            x_enc_dict = {"x": x}
        return x_enc_dict
    
    def decode(self, x):
        """
            converts the index encoded array to protein strings
            x: (torch.tensor or np.array) indexed encoded protein sequences
            return x_dec: (arr object) array of protein amino acids
        """
        if type(x) == torch.Tensor:
            x = x.detach().to("cpu")
        x_dec = convert_idx_to_aas(x) if self.data_type == "protein" else x
        assert x_dec.shape == x.shape 
        return x_dec
        
    def size(self):
        """
            get the size of dataset
            which is the same for all seeds
        """
        ns, xs, *xd = self.ds["x"].shape
        return xs
    
    def append(self, data_ext):
        """
            appends data_ext to self.dsall
            data_ext (dict): dictionary of data to be appended
            return (dict): the appended dataset
        """
        # all self.ds keys should be in data_ext keys
        ovp = set(self.dsall.keys()) & set(data_ext.keys())
        assert len(ovp) == len(self.dsall.keys())

        for k in self.dsall.keys():
            if (data_ext[k] is None) and (not(self.dsall[k] is None)):
                raise Exception(f"encountered None value for {k} while current value is not None")
            elif ((self.dsall[k] is None)): #and (not(data_ext[k] is None)):
                print(f"attempting to add to a None value for {k}, skip append")
                continue 
            else:
                ns, xs, *xd = self.dsall[k].shape
                ns_ext, xs_ext, *xd_ext = data_ext[k].shape
                self.dsall[k] = torch.cat((self.dsall[k], data_ext[k]), dim=1)
                assert self.dsall[k].shape == (ns, xs+xs_ext, *xd)
    
    def update(self, ds_n):
        """
            Replacing dsall with the given ds_n
            for the common keys between the two dicts
        """
        for k in self.dsall.keys():
            if k in ds_n.keys():
                self.dsall[k] = ds_n[k]
            else:
                print(f"Update skipped for field {k}!")
 
    def split(self, split_ratios):
        """
            The functions used below are coded by ehsans2
            device: "cpu" or "cuda", sometime splitting should be done 
                    on cpu due to lack of gpu memory.
        """
        # dataset contains repetitive samples:

        # create the dataset
        # self.ds is initiated
        # find the unique elements
        
        self.all_smpw = self.mask1d = None
        
        if True: #data_type == "protein":
            ns, xs, *xd = self.ds["x"].shape
            prtinv, prtcnt = apply_unique(self.ds["x"], self.device, self.dtype)
            # generate the mask for where the unique elements happen
            mask1d = get_unique_mask(prtinv, prtcnt) # 1d mask
            assert mask1d.shape == (ns*xs,)
            assert prtinv.shape == (ns*xs,)
            # get the weight for unique elements, if elem a has n occurance
            # the weight assigned to its occurances is 1/n
            prtinv2d = prtinv.reshape((ns, xs))
            smp_w_unnorm, n_smpunq, pi_min = get_unique_weight(prtinv2d, prtcnt)
            # get the weights per split of the data
            # e.g., for train only train samples have nonzero weight

            # the data is split into train, val, test based on the count of uniqu values
            # e.g., if there are 10 unique samples in total and split ratio is [0.7, 0.2, 0.1]
            # the 7 unique samples will be assigned to train split
            all_smpw = get_split_weights(smp_w_unnorm, n_smpunq, pi_min, prtinv2d,
                                         split_ratios, self.device) # list output

            self.all_smpw = all_smpw # list of weights for splits
            self.mask1d = mask1d.to(self.device) # (ns*xs,)
            
        else:
            train_split, val_split, test_split = split_ratios
            ns, xs, *xd = self.ds["x"].shape[1]
            inds = torch.arange(0, xs)
            inds = inds.broadcast_tp(ns, xs)
            assert inds.shape == (ns, xs)
            
            ui = self.batch_rng.uniform((ns, xs))
            uis = torch.argsort(ui, dim=-1)
            assert uis.shape == (ns, xs)
            
            inds = torch.take_along_dim(inds, uis, dim=-1)
            
            # do this in a better way
            self.train_inds = inds[:, :int(train_split*ds_size)]
            self.val_inds = inds[:, int(train_split*ds_size):int((train_split+val_split)*ds_size)]
            self.test_inds = inds[:, int((train_split+val_split)*ds_size):]
        
    def get_split_unique_mask(self):
        """
            get the unique mask for each split of the data
        """
        ### not finished
        
        mask_splits = []
        for s, smpw_s  in enumerate(self.all_smpw):
            ns, ds = smpw_s.shape
            mask2d = self.mask1d.reshape((ns, ds))
            assert mask2d.shape == (ns, ds)
            smpw_sm = smpw_s * mask2d
            smpw_sm = smpw_sm > 0
            assert smpw_sm.shape == (ns, ds)
            mask_splits.append(smpw_sm)
        return mask_splits
        
    def __call__(self, batch_size, samp_strat, tch_gen):
        """
            get a batch of samples
            shuffle the samples when reached to the end of data
            
            Parameters
            ----------
            samp_strat: sampling strategy either "iid" or "det" (deterministic)
            
            return batch_dict: (dict) batch of samples
            
        """
        n_seeds, = self.batch_rng.shape
        
        if True: #self.data_type == "protein":
            w_tr = self.all_smpw[0] # get train weights
            w_tr_bin = w_tr > 0
            n_tr_s = torch.sum(w_tr_bin, dim=-1)
            assert n_tr_s.shape == (n_seeds,)
            n_tr = n_tr_s[0].item()
        else:
            n_tr = self.train_inds.size
            
        
        if self.start_trav_ind >= n_tr:
            self.start_trav_ind = 0

        # shuffle when reached to the end
        if self.start_trav_ind == 0:
            if True: #self.data_type == "protein":
                self.uti = weighted_shuffle_v0(w_tr, n_tr, self.device, tch_gen)
            else:
                urv = self.batch_rng.uniform(sample_shape=(n_seeds, n_tr))
                assert urv.shape == (n_seeds, n_tr)
                # range of uas values (0, n_tr)
                uas = urv.argsort(dim=-1)
                assert uas.shape == (n_seeds, n_tr)
                uas2 = uas.reshape(-1)
                assert uas2.shape == (n_seeds*n_tr,)
                # get the indices from the dataset
                uti = self.train_inds[uas2]
                assert uti.shape == (n_seeds*n_tr,)
                self.uti = uti.reshape((n_seeds, n_tr))
                assert self.uti.shape == (n_seeds, n_tr)
                

        start_ind = self.start_trav_ind
        end_ind = min(start_ind + batch_size, n_tr)
        n_mb = end_ind - start_ind

        batch_dict = {}
        for k in self.ds.keys():
            v = self.ds[k]
            if (v is None):
                batch_dict[k] = None # optm_w is None for pgvae
                continue

            ns, n_ds, *vd = v.shape
            assert v.shape == (n_seeds, n_ds, *vd)

            # get the batch of indices first (faster)
            ui = self.uti[:, start_ind:end_ind]
            assert ui.shape == (n_seeds, n_mb)

            vd_exp = tuple([1]*len(vd))
            uii = ui.reshape((n_seeds, n_mb, *vd_exp))
            vi = torch.take_along_dim(v, uii, dim=1)
            assert vi.shape == (n_seeds, n_mb, *vd)
            batch_dict[k] = vi #vii

        self.start_trav_ind = self.start_trav_ind + batch_size

        return batch_dict