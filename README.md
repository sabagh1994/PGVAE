# Robust Model-Based Optimization for Challenging Fitness Landscapes

This repository contains the implementation for <ins>P</ins>roperty-<ins>P</ins>rioritized <ins>G</ins>eneraive <ins>V</ins>ariational <ins>A</ins>uto <ins>E</ins>ncoder (PPGVAE).
Model-based optimization with PPGVAE robustly finds improved samples regardless of 1) the imbalance between low- and high-fitness training samples, and 2) the extent of their separation in the design space. MBO with our PPGVAE can be used for both discrete and continuous design spaces. Details of our comprehensive benchmark are covered in the following. 

<details open>
<summary><h2>Guide to MBO with PPGVAE</h2></summary>

+  <details>
   <summary><strong>Cloning the Repository</strong></summary>
   
    1. `git clone --recursive https://github.com/sabagh1994/PGVAE.git`
    2. `cd PGVAE`
    </details>

+  <details>
   <summary><strong>Download the Datasets</strong></summary>
   
   To download the datasets used to create the oracles, and generate train sets at varying separation and imbalance ratios, 
   run `./datasets/download.sh`. The downloaded files will be `./datasets/aav.csv`, `./datasets/GB1.txt`, `./datasets/PhoQ.txt`, and `pinn_poisson.npz`.
   
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
    2. Oracles used for protein benchmark tasks.
   Oracles will be stored in `oracles` directory, and train sets will be stored at a separate folder for each benchmark task in `sample_trainset` directory.
   Note that GMM and PINN won't have any oracles stored. GMM oracle can be constructed with its parameter specification which is stored within each instance of
   train set, e.g., `sample_trainset/gmm/ds0.npz`. PINN oracle is generated from `datasets/pinn_poisson.npz` when its instance is created in `scripts/run_mbo.py` script.
    
    </details>

