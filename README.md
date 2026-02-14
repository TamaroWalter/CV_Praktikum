### Repo für das Praktikum von Computer Vision

## Setup of python enviroment (linux):

To install pip packages, a `venv` enviroment is needed:

**If there is no venv environment:**

`python3 -m venv .venv`

**If there already is a venv environment:**

- Start python enviroment: `source .venve/bin/activate`
- Install packages with: `pip install -r .requirements.txt`
- Write to the requirements: `pip freeze > .requirements.txt`

The `.venv` should be included in the `.gitignore`, as it is too big for a repo.

## Start

Start the main function of Aufgabe 1 with an image number to test (adapt your path to BSDS500 previously) `python3 Aufgabe1/main.oy 2018`

### Data

Get image data on: `git@github.com:BIDS/BSDS500.git` and clone into this repository into a folder 'BSDS500'


### How to work with matrices

A little help on how to work with numpy-ndarrays
```
a: np.ndarray = np.array([1, 2, 3])              # 1D
b: np.ndarray = np.array([[1, 2], [3, 4]])        # 2D
c: np.ndarray = np.zeros((3, 3), dtype=np.uint8)  # 3x3 of zeros
d: np.ndarray = np.ones((2, 4), dtype=np.float64) # 2x4 of ones
e: np.ndarray = np.empty((5, 5))                  # uninitialized
```