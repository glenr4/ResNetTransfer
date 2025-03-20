# ResNetTransfer

From following the tutorial at: https://medium.com/@paravisionlab/supercharge-your-ai-resnet50-transfer-learning-unleashed-b7c0e40976c4

## Restore and Run
1. Make sure you have Miniconda installed: https://www.anaconda.com/docs/getting-started/miniconda/install
1. Setup the environment: `conda env create -f environment.yml`
1. Activate the environment: `conda activate ResNetTransfer`
1. Run the script: `python main.py`

## Enable GPU on Windows
Creating the environment from `environment.yml` should mean that this isn't necessary but for reference:
1. `pip install tensorflow==2.10.0` # last version that supports Windows
1. `conda install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge`
1. `pip install --upgrade tensorflow-gpu==2.10.0`

https://stackoverflow.com/a/78351204
