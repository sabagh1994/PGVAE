#!/bin/bash 


# Usage: download datasets from google drive

datasets_ppgvae_bench="1nT8u7QnOvl0ZCmNljNf7Wk8luBPIqnDb"
gdown $datasets_ppgvae_bench

# sanity check
if ! md5sum -c datasets_ppgvae_bench.md5; then echo "corrupted files"; fi


tar -xzvf datasets_ppgvae_bench.tar.gz

rm *.tar.gz
