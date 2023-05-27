import torch
import numpy as np
import pandas as pd


class Oracle_protein_gt():
    def __init__(self, orc_path=None, seq_len=None, 
                 num_cls=None, device=None, name=None):
        """
            Intended for oracles that return ground truth value 
            as the predicted score of the protein activity if the
            sequence is available in the protein dataset. Otherwise,
            the returned score is zero
            
            orc_path examples:
                "gb_gt" : "/home/sglhs/Projects/MBO_meta_epiAmb/protein_datasets_bench/CLADE/GB1/GB1.txt"
                "phoq_gt": "/home/sglhs/Projects/MBO_meta_epiAmb/protein_datasets_bench/CLADE/PhoQ/PhoQ.txt"
                "his7": "/home/sglhs/Projects/MBO_meta_epiAmb/protein_datasets_bench/DeepSequence/" \
                        "HIS7_yeast/HIS7_YEAST_Kondrashov2017.txt"
        """

        self.name = name
        self.seq_len = seq_len
        self.num_cls = num_cls
        assert not(orc_path is None)
        
        df_orc_gt = pd.read_csv(orc_path, sep="\t")
        df_orc_gt.columns = ["sequence", "score"]
        
        assert ("sequence" in df_orc_gt.columns) and ("score" in df_orc_gt.columns)
        # normalizing oracle scores
        df_orc_gt["score"] = df_orc_gt["score"]/max(df_orc_gt["score"])
        df_orc_gt.index = df_orc_gt["sequence"]
        self.orc = df_orc_gt

    def __call__(self, x):
        n_seeds, xs, *xd = x.shape
        # convert x from onehot encoding to string
        x = x.reshape((n_seeds, xs, self.seq_len, self.num_cls))
        _, labels = torch.max(x, dim=-1) # onehot to index (cls)
        assert labels.shape == (n_seeds, xs, self.seq_len)
        labels_arr = labels.detach().to('cpu')
        aa_arr = convert_idx_to_aas(labels_arr) # aa_arr is not a tensor
        assert aa_arr.shape == (n_seeds, xs, self.seq_len)
        aa_arr = aa_arr.reshape((n_seeds*xs, self.seq_len))
        aa_seqs = ["".join(aa_seq) for aa_seq in aa_arr]
        assert len(aa_seqs) == n_seeds*xs
        
        df_aa = pd.DataFrame(aa_seqs, columns=["sequence"])
        df_aa["score"] = 0.
        
        df_orc = self.orc
        comm_seqs = set(df_aa["sequence"]) & set(df_orc["sequence"])
        df_orc_results = df_aa[df_aa["sequence"].isin(comm_seqs)]
        
        df_orc_results["score"] = df_orc.loc[df_orc_results["sequence"]]["score"].values
        seq_diff = set(df_aa["sequence"]).difference(comm_seqs)
        if len(seq_diff) != 0:
            df_xt = df_aa[df_aa["sequence"].isin(seq_diff)]
            df_orc_results = pd.concat((df_orc_results, df_xt))
            
        assert len(df_orc_results["sequence"]) == n_seeds*xs
        
        x_raw = df_orc_results["sequence"].values
        x_raw = x_raw.reshape((n_seeds, xs))
        y = df_orc_results["score"].values
        y = y.reshape((n_seeds, xs))
        orc_dict = {"x_raw": x_raw, "y": y}
        
        return orc_dict
        
class Oracle_gmm():
    def __init__(self, mus=None, sigmas=None, weights=None, name="gmm"):
        """
            Uses gaussian mixture model as the 
            sequence to activity function y = gmm(x)
            Multivariate gaussian with diagonal covariance
            matrix is implemeted.
            
            mus: (torch.tensor) mean of gaussians (n_gmm, *xd)
            sigmas: (torch.tensor) std of gaussians (n_gmm, *xd)
            weights: (torch.tensor) mixture weights of gaussians
        """
        self.name = name
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights
        
    def __call__(self, x):
        (n_seeds, xs, *xd) = x.shape
        n_gmm = self.mus.shape[0]
        assert self.mus.shape == (n_gmm, *xd)
        assert self.sigmas.shape == (n_gmm, *xd)
        assert self.weights.shape == (n_gmm,)
        
        # for all mus and xs compute x-mu
        x = x.reshape((n_seeds, xs, 1, *xd))
        mus = self.mus.reshape((1, 1, n_gmm,  *xd))
        sigmas = self.sigmas.reshape((1, 1, n_gmm,  *xd))
        
        power = (x - mus)/sigmas
        power = power**2
        assert power.shape == (n_seeds, xs, n_gmm, *xd)
        power = torch.sum(power, dim=-1)
        assert power.shape == (n_seeds, xs, n_gmm)
        weights = self.weights.reshape((1, 1, n_gmm))
        y = weights*torch.exp(-0.5*power)
        assert y.shape == (n_seeds, xs, n_gmm)
        y = torch.sum(y, dim=-1)
        assert y.shape == (n_seeds, xs)
        
        x = x.reshape((n_seeds, xs, *xd))
        orc_dict = {"x_raw": x, "y": y}
        
        return orc_dict

class Oracle_pinn():
    def __init__(self, orc_path, dtype, device):
        """
            orc_path (str): path to the stored ground truth
                    of the differential equation solved by pinn
        """
        self.orc = np.load(orc_path)["gt"]
        self.orc = torch.tensor(self.orc[0], dtype=dtype, device=device)
        
    def __call__(self, x):
        
        (ns, xs, *xd) = x.shape
        dsr = int(np.prod(xd)**0.5)
        #orc = self.orc.reshape((1, 1, *xd))
        orc_re = self.orc.reshape(dsr, dsr)
        w_re = -orc_re
        w_re = w_re - w_re.min()
        w_re = w_re**2
        w_re = w_re/w_re.sum()
        w = w_re.reshape(1, 1, *xd)
        
        orc_n = self.orc.reshape((1, 1, *xd))
        orc_n = orc_n - torch.mean(orc_n)
        dim_red = list(np.arange(2, 2+len(xd)))
        y_wmse = ((x - orc_n).square()*w).sum(dim=dim_red)
    
        #y_mse = torch.mean((x - orc)**2, dim=dim_red)
        #y_mae, _ = torch.max(torch.abs(x - self.orc), dim=-1)
        #y_mae = torch.mean(torch.abs(x - self.orc), dim=dim_red)
        
        y = y_wmse
        y = -torch.log10(y)
        assert y.shape == (ns, xs)
        
        orc_dict = {"x_raw": x, "y": y}
        return orc_dict