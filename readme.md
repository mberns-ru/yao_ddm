# Yao Lab DDM

A drift diffusion model with fittable bounds and drift, which is scaled by log(AM Rate) - log(6.25).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

To install, first download the files and place them in a project folder.
Then, open Anaconda Prompt and type the following to install the dependencies in a new Conda environment:

```bash
cd project_folder
conda env create --file requirements.yml
conda activate ddm_env
```

## Usage

To run the model, type the following command into the active Conda environment (ddm_env):

```bash
python gui.py
```

This will open a GUI with the following inputs:
- CSV File: Path to .csv file containing behavior data
- Project Path: Path of the folder where you want the results saved to, such as the folder with these code files.
- Model Title: Name of the current model, i.e. "M16 Hearing Loss"

Then click "Run" and you're good to go!
