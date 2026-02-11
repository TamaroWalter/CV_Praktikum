### Repo für das Praktikum von Computer Vision

## Setup of python enviroment (linux):

To install pip packages, a `venv` enviroment is needed:

**If there is no venv environment:**

`python3 -m venv .venv`

**If there already is a venv environment:**

- Start python enviroment: `source .venve/bin/activate`
- Install packages with: `pip install -r requirements.txt`
- Write to the requirements: `pip freeze > requirements.txt`

The `.venv` should be included in the `.gitignore`, as it is too big for a repo.

### Data

Get image data on: `git@github.com:BIDS/BSDS500.git` and clone into this repository
