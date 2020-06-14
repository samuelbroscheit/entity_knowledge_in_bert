#!/usr/bin/env bash

git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout ec6f8ef99a8c6942133e01a610def197e1d6d9dd
git apply ../fairseq_patch_01.patch
git apply ../fairseq_patch_02.patch
