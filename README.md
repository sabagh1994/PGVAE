# Robust Model-Based Optimization for Challenging Fitness Landscapes

This repository contains the implementation for **<ins>P</ins>roperty-<ins>P</ins>rioritized <ins>G</ins>eneraive <ins>V</ins>ariational <ins>A</ins>uto <ins>E</ins>ncoder (PPGVAE)**.
Model-based optimization with PPGVAE robustly finds improved samples regardless of 1) the imbalance between low- and high-fitness training samples, and 2) the extent of their separation in the design space. MBO with our PPGVAE can be used for both discrete and continuous design spaces. Details of our comprehensive benchmark are covered in the following. 

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
   Read **"Train Set and Output Format"** for the contents of train set `*.npz` and output `*.pt`, for each benchmark task .
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
     * `"data_type"` type of the benchmark task, i.e., `"gmm"`
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
   Stacked model and data training was first used in (firth github). Brief explanation. Mention tch_utils. Please cite (these paper or githubs) if you use ... for your research.
</details>

