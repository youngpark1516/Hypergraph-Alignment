#!/usr/bin/env bash
set -e
mkdir -p data/raw
mkdir -p data/processed/deformingthings4d/{train,val,test}
mkdir -p data/processed/faust/{train,val,test}
mkdir -p data/processed/sapien/{train,val,test}
echo "Initialized data/ layout."
