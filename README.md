# fashion-mnist-classificator

Сlassification of clothing on the Fashion-MNIST dataset

[![pre-commit](https://github.com/Kulikov17/fashion-mnist-classificator/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Kulikov17/fashion-mnist-classificator/actions/workflows/pre-commit.yml)

## Required depending:
  - Python >=3.10
  - Poetry

## Сommands to run
1. `poetry install` -  installing dependencies
2. `pre-commit install` - install pre-commit hooks
3. `pre-commit run -a` - check black, isort, flake8
4. `python train.py` - train model
5. `python infer.py` - inference model (on test dataset from MNIST). Results will be save in a csv file

Parameters for training can be changed in `config.yaml`.
