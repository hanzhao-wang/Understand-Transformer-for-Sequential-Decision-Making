# Understand Transformer for Sequential Decision Making

This repository contains all the code used in the paper: ["Understanding the Training and Generalization of Pretrained Transformer for Sequential Decision Making"](https://www.arxiv.org/abs/2405.14219).


## Usage

0. This repository uses wandb for experiment tracking. Please make sure your account is correctly configured. See https://docs.wandb.ai/quickstart for more details.
1. Clone the repository to your local machine.
2. Prepare the environment with conda: `conda env create -f environment.yml`.
3. Execute `experiment/data_generate/generate.sh` to synthesize the dataset.
4. Use bash scripts to run the main experiments. They are respectively available at:
    - Dynamic Pricing: `experiment/DP_cts/DP_cts.sh`
    - Multi-Arm Bandit: `experiment/MAB/MAB.sh`
    - Linear Bandit: `experiment/LinB/LinB.sh`
    - Newsvendor: `experiment/NV_cts/NV_cts.sh`

    Note:
    - Calling sequence: bash script -> py script in the same dir -> src/train.py
    - The bash scripts are configured to run on both normal environments and slurm environments. If you are using slurm, please modify the scripts accordingly.
5. Go into `src` directory and run `python test.py` to test the trained models and plot the results.

Feel free to leave commments or open issues if you have any questions.

