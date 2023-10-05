.PHONY: venv
SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

##########################################################
####################      VENV     #######################
##########################################################

venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && python -m pip install torch==1.12.1+cu116 \
	        --extra-index-url https://download.pytorch.org/whl/cu116
	source venv/bin/activate && python -m pip install gdown
	source venv/bin/activate && python -m pip install -r requirements.txt
