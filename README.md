[![Paper shield](https://img.shields.io/badge/docker-container-blue)](https://hub.docker.com/r/neurodata/graph-stats-book)

# Network Machine Learning

This repository provides the programming component for an introductory book to network machine learning with python. Our book will be published with Cambridge University Press in early 2025. This book assumes a recent version of python (we use Python `3.11` and `3.12`).

# Usage

Begin by cloning this repository locally:

```
git clone git@github.com:ebridge2/textbook_figs.git <destination>/<directory>/
```

## Compiling the figures via docker

The simplest way to compile the figures associated with this work is via the docker container. You can obtain the docker container locally with a properly configured docker engine using:

```
docker pull neurodata/graph-stats-book
```


Finally, enter the docker container while provisioning the repository, navigate to the appropriate directory, and then compile the book:

```
docker run -ti --entrypoint /bin/bash -v <destination>/<directory>/textbook_figs:/book -p neurodata/graph-stats-book
cd /book
jupyter-book build textbook_figs/
```

A fully-rendered HTML version of the book will be built in `<destination>/<directory>/textbook_figs/_build/html/index.html`. You can open this with a web browser from your computer.

## Compiling the figures locally

We would recommend setting up a virtual environment for the book, using `virtualenv` with a recent version of `python3`. Within this virtualenv, install the dependencies:

```
cd <destination>/<directory>/textbook_figs
pip install -r requirements.txt
```

and then compile the book:

```
jupyter-book build textbook_figs/
```

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).

## Code

Functions specific to this book - e.g., plotting functions we use regularly - has been stored in the corresponding package [graphbook-code](https://github.com/neurodata/graphbook-code/tree/main).

