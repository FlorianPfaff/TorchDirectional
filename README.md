# TorchDirectional
Some functions for modeling directional distributions for the use in PyTorch (including GPU). The syntax is similar to that of
[libDirectional](https://github.com/libDirectional/libDirectional).
To test TorchDirectional, create an empty folder, e.g. testFolder, and assign WORKSPACE to it, e.g., via `WORKSPACE=testFolder`.
Then, run the test cases via

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $WORKSPACE/miniconda.sh
    bash $WORKSPACE/miniconda.sh -b -p $WORKSPACE/miniconda
    . $WORKSPACE/miniconda/bin/activate
    conda create -n "pl" python=3.10
    conda activate pl
    conda install -c conda-forge pytorch-lightning matplotlib scipy
    pip install tap.py
    python -m tap | tee taptestresults.tap

Author: Florian Pfaff, pfaff@kit.edu