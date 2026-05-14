#!/bin/bash
set -euo pipefail

#Load secrets
if [ ! -f secrets.sh ]; then
    echo "ERROR: secrets.sh not found."
    exit 1
fi
source secrets.sh

#Git config
git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_USERNAME"
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"

#Conda env
conda env create -f environment.yaml
conda run -n amrcc python -m ipykernel install --user --name securespace --display-name "Python (securespace)"