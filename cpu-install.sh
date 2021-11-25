#!/bin/bash

# un-comment to install anaconda
install_dir=$HOME/experiment/anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p  $install_dir
export PATH=$install_dir/bin:$PATH


# create conda env
conda init bash
. ~/.bashrc
conda env create -f environment.yml # Install dependencies
conda activate modelkeeper


