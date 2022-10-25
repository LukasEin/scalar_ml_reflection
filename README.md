# Group equivariant networks for a scalar field theory

This repository contains a Group equivariant convolutional neural network (G-CNN) that includes translations and reflections of a 2 dimensional lattice.

## Setup

Download the dataset from [this repository](https://zenodo.org/record/4644550) at Zenodo. Adjust the `path_data` variable in `Training and Testing.ipynb` and `Optuna.ipynb` accordingly. The path is initialized to the `./datasets` folder in this repository.

The conda environment that we used for this task is provided in the `env.yml` file. It is the same one as for the counting task. Thus, if it already exists, it only needs to be activated via shell with:

```
conda activate scalar_ml_reflection
```

If not, it needs to be created first:

```
conda env create -f env.yml
conda activate scalar_ml_reflection
```

## Source files

In `./src` the two files `refconv2d.py` and `transform_featuremaps.py` are stored. 

- `refconv2d.py`: Here the equivariant convolutional layer is defined. It is equivariant under translations and reflections of the underlying lattice.
- `transform_featuremaps.py`: This stores the functions transforming the feature maps under the reflections in t, x, and t+x direction. It transforms featuremaps that represent different fluxvariables each according to the underlying transformation behavior. Translations are not included her but can be accomplished by the function `torch.roll` if needed.

Nothing should be changed here.

## Training and testing models

In the jupyter notebook `Training and Testing.ipynb` various models can be trained and tested. Additional classes and functions are stored in the corresponding python file `training_and_testing.py`.

If the notebook is executed, it will create two `.pickle` files: one containing the test results on the $`60 \times 4`$ lattice and one containing the test results on other lattice sizes.

- `path_data` should be pointing to the directory containing the datasets from [this repository](https://zenodo.org/record/4644550).
- `name_string_helper` defines the name of the `.pickle` file where the testing data will be stored.

The test results are displayed in the jupyter notebook `Test Plots.ipynb`. In order to do that, the `.pickle` files that shall be plotted have to be added to the lists `filenames` and `ls_filenames`.

## Looking for good architectures with Optuna

We use `optuna` to look for well-performing architectures. The jupyter notebook `Optuna.ipynb` contains such a search. It produces one `.pickle` file for each number of training samples that is chosen. Additional classes and functions are stored in `Optuna.py`.

## Code overview

* `Optuna.ipynb` launches an optuna search
* `Optuna.py` contains classes and functions for `Optuna.ipynb`
* `Test Plots.ipynb` displays the test results of models trained in `Training and Testing.ipynb`
* `Training and Testing.ipynb` trains and tests models of an architecture that has to be specified
* `training_and_testing.py` contains classes and functions for `Training and Testing.ipynb`