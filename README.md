# Robust Model-Based Optimization for Challenging Fitness Landscapes

This repository contains the implementation for **<ins>P</ins>roperty-<ins>P</ins>rioritized <ins>G</ins>eneraive <ins>V</ins>ariational <ins>A</ins>uto <ins>E</ins>ncoder (PPGVAE)**.
Model-based optimization with PPGVAE robustly finds improved samples regardless of 1) the imbalance between low- and high-fitness training samples, and 2) the extent of their separation in the design space. MBO with our PPGVAE can be used for both discrete and continuous design spaces. Details of our comprehensive benchmark are covered in the following. 

![image](https://github.com/sabagh1994/PGVAE/assets/33433428/551dccb8-15a8-4f44-b590-e4dd4266cf23)


<details open>
<summary><h2>Guide to MBO with PPGVAE</h2></summary>

+  <details>
   <summary><strong>Cloning the Repository</strong></summary>
   
    1. `git clone --recursive https://github.com/sabagh1994/PGVAE.git`
    2. `cd PGVAE`
    </details>

+  <details>
   <summary><strong>Download the Datasets and Configs</strong></summary>
   
   To download the datasets used to create the oracles, and generate train sets at varying separation and imbalance ratios, 
   run `./datasets/download.sh`. The downloaded files will be `./datasets/aav.csv`, `./datasets/GB1.txt`, `./datasets/PhoQ.txt`, and `pinn_poisson.npz`.
   AAV dataset was retreived from https://benchmark.protein.properties/landscapes. PINN dataset was originally generated in experiments of https://arxiv.org/abs/2305.17387,
   however the `.npz` format can only be accessed from here.

   To download the config files run `./configs/download.sh`. The configs will be stored at the `configs` directory. There will be a sample config for each benchmark
   task. You can modify the config file to include more methods and train sets. Read **"Running Configuration"** for more details.
   
    </details>

+  <details>
   <summary><strong>Make a Virtual Environment</strong></summary>
   
   Before running MBO with PPGVAE (or other methods) make sure that all the required packages are installed.
   To create a virtual environment with all the required packages installed,
   
    1. Install Python version 3.9 or higher. We used Python 3.9.
    2. Run `make venv`. This step creates a folder named `./venv` which contains all the required packages.
    3. Run `source venv/bin/activate` to activate the venv
    </details>

+  <details>
   <summary><strong>Generating the Train Sets and Oracles</strong></summary>
   
   For each benchmark task, trains sets and oracles should be generated before running MBO. Note that
   this step requires the datasets included in the `datasets` folder.
   Navigate to `notebooks` folder and run the jupyter notebook `ds_generator.ipynb`. This will create,
   
    1. Train sets for semi-synthetic GB1 and PhoQ, AAV, PINN and GMM benchamrk tasks, at varying imbalance ratios and separation levels.
       Train sets will be stored at a separate folder for each benchmark task in `sample_trainset` directory.
    2. Oracles used for protein benchmark tasks. Oracles will be stored in `oracles` directory including `oracles/protein_aav`, `oracles/protein_gb_synth`, and `oracles/protein_phoq_synth`  
   
   **Note 1:** GMM and PINN won't have any oracles stored. GMM oracle can be constructed with its parameter specification, which is stored within each instance of
   train set, e.g., `sample_trainset/gmm/ds0.npz`. PINN oracle is generated from `datasets/pinn_poisson.npz` when its instance is created in `scripts/run_mbo.py`. \
   **Note 2:** For semi-synthetic GB1 and PhoQ datasets, train sets and oracles are generated with appended length of three corresponding to the lowest separation. For higher
   separation, set the variable `ext_len` to higher integer values (default 3) in `notebooks/ds_generator.ipynb`.
    
    </details>

+  <details>
   <summary><strong>Running MBO</strong></summary>
   
   To perform MBO, one config file is needed. An example of the config file is included in `configs/run_config.json`. Read **"Running Configuration"** for the
   description of each field in the config file. To run MBO with the example config file, execute
   
      ```bash
      python scripts/run_mbo.py --run_config configs/run_config.json &> log_mbo
      ```
   This runs 10 MBO steps using PPGVAE on the example GMM train set located at `sample_trainset/ds0.npz`. The results will be stored at `results/ds0/*.pt`.
   Read **"Train Set and Output Format"** for the contents of train set `*.npz` and output `*.pt`, for each benchmark task.
    </details>

+  <details>
   <summary><strong>Summarizing the Results</strong></summary>
   
   After running MBO for each benchmark task, there will be multiple `*.pt` files in the `results` directory. To compute various statistics,
   e.g., maximum property relative to train/initial set, for the generated samples from MBO, run summary notebooks in the `notebooks` directory, e.g., `summary_gmm.ipynb` summarizes
   the results for the GMM benchmark. In each summary notebook, the following three dataframes are constructed and saved in the `summary` directory.
   1) `df_stats` each row represents a unique configuration of (imbalance ratio, separation level, seed number, method name, MBO step). Various statistics
       are computed for each unique configuration.
   2) `df_bs` contains the statistics computed in (1) as well as their 95% bootstrap confidence intervals for each unique configuration of
      (imbalance ratio, separation level, method name, MBO step). Note that "seed number" is not in the configuration. This dataframe was used
      to study the impact of varying imbalance for a given separation level in each benchmark task.
   3) `df_bsg` contains the statistics computed in (1) as well as their 95% bootstrap confidence intervals for each unique configuration of
      (separation level, method name, MBO step). Note that both "seed number" and "imbalance ratio" are not in the configuration. This dataframe was used
      to generate the plots representing the impact of separation level aggragated over all imbalance ratios.
    </details>

</details>

<details>
<summary><h2>Running Configuration</h2></summary>

+ <details open>
  <summary><strong>Example</strong></summary>
   
   An example of the configuration file `configs/run_config.json` is,
   ```json
   {
       "description": "sample config file to run MBO with ppgvae or other methods",
       "ds_rootdir": "sample_trainset",
       "ds_names": ["ds0.npz"],
       "method_names": ["pgvae"],
       "weighted_opt_firststeps": [false],
       "n_samples_gens": [100],
       "savedir": "results",
       "vae_type": "mlp",
       "n_seeds": 10,
       "mbo_steps": 10
   }   
   ```
   </details>
 
+ <details>
  <summary><strong>Description of the Arguments</strong></summary>
   
   * `"description"` is the notes about the configuration file or whatever notes you want to keep for the configuration you are using.
   * `"ds_rootdir"` the root directory containing the train sets.
   * `"ds_names"` is a list containing the file names for the train sets. A single train set `ds_name` is read from `ds_rootdir/ds_name`.
   * `"method_names"` is a list containing the name of the methods, e.g., `["pgvae", "rwr", "cem-pi", "dbas", "cbas"]`
   * `"weighted_opt_firststeps"` if false the first MBO step uses uniform nonzero weights in weighted optimization as done in CbAS paper. If True, weighted
     optimization with non-uniform weights is performed in the first step as well. Both CbAS and PPGVAE run with `false`. Leave this as `[false]` for simplicity.
   *  `"n_samples_gen"` is a list containing the integer number of samples generated per MBO step.
   *  `"vae_type"` is a string specifying the type of VAE. This should be set to `"mlp"` for all experiments in the paper.
   *  `"n_seeds"` determines the number of models to be trained in parallel, each leading to a different chain of samples generated from MBO. See **"Stacked Model and Data Training"** for more details.
   *  `"mbo_steps"` is the number of MBO steps performed.
   
   </details>
   
</details>


<details>
<summary><h2>Train Set and Output Formats</h2></summary>

+ <details>
  <summary><strong>Train Set Format</strong></summary>
   
   Train sets are in `*.npz` format. Each file consists of three fields `x`, `y`, and `orc_spec`. Both `x` and `y` are numpy arrays containing the samples from the design space and      their associated properties. `x` is an array of strings for protein benchmarks.`orc_spec` is a dictionary containing the oracle specifications and variables involved in train set     generation. These depend on the benchmark task (gmm, pinn, aav, ...) as explained below.
  1) In GMM `orc_spec` consists of
     * `"mu_1st"` mean of the first Gaussian mode (less desired mode)
     * `"mu_2nd"` mean of the second Gaussian mode (more desired mode). Specifies the extent of separation as `"mu_1st"` is set to zero.
     * `"ro"` imbalance ratio between the less desired and more desired train samples.
     * `"data_type"` type of the benchmark task, i.e., `"gmm", "protein", "pinn"`. This affects the initialization of `Dataset` object in `run_mbo.py`
     * `"sigmas_gmm"` numpy array containing the standard deviations for the two modes.
     * `"weights"` peak height of each Gaussian mode.
     * `"N1"` number of training samples taken from the less desired mode.
     * `"N"` size of the train set.
  2) In AAV `orc_spec` consists of
     * `"mut_thr"` an integer indicating the minimum number of mutated sites in less desired samples. Specifies the extent of separation.
     * `"ro"` imbalance ratio
     * `"data_type"` type of the benchmark task, i.e., `"protein"`
     * `"N"` size of the train set.
     * `"orc_path"` path to the oracle.
  3) semi-synthetic GB1 and PhoQ, have the same `orc_spec` as the AAV, with the exclusion of `mut_thr`.
  </details>
   
+ <details>
  <summary><strong>Output Format</strong></summary>

   Outputs are in `*.pt` format. The output dictionary consists of the following fields,
  * `x` is a `torch.tensor` with shape `(n_seeds, N_total, dim_rest)`. `dim_rest` varies depending on the type of dataset. For one-hot encoded
    protein sequences `dim_rest = (sequence length, number of amino acids)`. For the gmm dataset `dim_rest = 1`
  * `y` is a `torch.tensor` with shape `(n_seeds, N_total)` containing the property (fitness) values.
  * `w_optm` is a `torch.tensor` with shape `(n_seeds, N_total)` containing the optimization weights. This is `None` for PPGVAE as it does not
    perform weighted optimization.
  * `step` is a `torch.tensor` with shape `(n_seeds, N_total)` containing the MBO step values.
  * `orc_spec` is a dictionary containing oracle specifications and variables used for train set generation, as explained in **"Train Set Format"**.
  * `method_name` is a string specifying the method used for MBO, e.g., `"pgvae", "rwr", "cem-pi"`.
  * `n_samples_gen` is the integer number of samples generated per MBO step.
  * `weighted_opt_firststep` is a bool determining whether non-uniform weighted optimization was used in the first MBO step. This was set to
    `false` in all experiments of the paper.
  * `datadir` is the path to the train set.
    
  `n_seeds` is the number of models ran in parallel to perform MBO with different seeds. \
  `N_total` is the total number of samples generated in MBO which is equivalent to the number of MBO steps times the number of samples generated per MBO step.  

  </details>

</details>


<details>
<summary><h2>Stacked Model and Data Training</h2></summary>
   
   The entire training pipeline, including the models, the data, and the random number generators, were stacked along a first dimension across multiple indpendent runs. This allows our code to efficiently run     many independent instances in parallel on a single GPU device. For further details, you can check the BMLP and BatchRNG classes in the `tch_utils.py` script. 

   Note that the shape of most tensors in our implementation starts with a `(n_seeds, ...)` prefix, where n_seeds is the number of independent runs specified in the input config files. 
   This approach was first implemented in "On the Importance of Firth Bias Reduction in Few-Shot Classification (https://github.com/ehsansaleh/firth_bias_reduction)"
   and used in the following two studies as well,
   1. BEDwARS: a robust Bayesian approach to bulk gene expression deconvolution with noisy reference signatures (https://github.com/sabagh1994/BEDwARS)
   2. Learning from Integral Losses in Physics Informed Neural Networks (https://github.com/ehsansaleh/btspinn)

   **If you use our implementation in your work, please Firth Bias Reduction paper (https://arxiv.org/abs/2110.02529) or this study.**
</details>

## References

* The bioRxiv link to the paper:
  * PDF link: https://browse.arxiv.org/pdf/2305.13650v2.pdf
  * Web-page link: https://arxiv.org/abs/2305.13650v2

* Here is the bibtex citation entry for our work:
```
@article{ghaffari2023,
  title={Robust Model-Based Optimization for Challenging Fitness Landscapes},
  author={Ghaffari, Saba and Saleh, Ehsan and Schwing, Alexander G and Wang, Yu-Xiong and Burke, Martin D and Sinha, Saurabh},
  journal={arXiv preprint arXiv:2305.13650v2},
  year={2023}
}
```
