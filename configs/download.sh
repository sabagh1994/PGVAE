#!/bin/bash 


# Usage: download configs from google drive

configs_bench="11w1DYgszSUioICf99bx4PGJVBrfiF-tZ"
gdown $configs_bench

# sanity check
if ! md5sum -c configs.md5; then echo "corrupted files"; fi


tar -xzvf configs_bench.tar.gz

rm *.tar.gz
