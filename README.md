# Companion code for VSE paper, ICASSP 2026

## Environment

- Python version 3.10.2
- CUDA version 12.2

### Python packages

Packages are listed with versions in `requirements.txt` aswell.
Two packages in `requirements.txt`, `ruff` and `basedpyright` are only used for
formatting and type checking, they are not necessary at all.

It is possible and likely that the code works with newer versions of the extensions.
These are just the versions I used.

- numpy==2.2.6
- torch==2.7.0
- scipy==1.15.3
- matplotlib==3.10.3
- typing-extensions==4.13.2

## Running
Simply run `python main.py`. All configurations are in the code.
This will generate data, train and test code. All outputs are put in `out` folder
in workspace directory.
