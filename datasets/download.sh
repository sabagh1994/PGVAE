#!/bin/bash 


# Usage: download datasets from google drive

datasets_ppgvae_bench="1nT8u7QnOvl0ZCmNljNf7Wk8luBPIqnDb"
gdown $datasets_ppgvae_bench

# sanity check
if ! md5sum -c datasets_ppgvae_bench.md5; then echo "corrupted files"; fi


tar -xzvf datasets_ppgvae_bench.tar.gz
cp datasets_ppgvae_bench/* ./
# mnist train set used in the paper is downloaded, not the entire mnist dataset
mv toy_mnist.npz ../sample_trainset

rm -rf datasets_ppgvae_bench
rm *.tar.gz
