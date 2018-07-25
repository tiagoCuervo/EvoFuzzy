# EvoFuzzy
This is a very simple Python implementation of the [Differential Evolution Algorithm](http://www1.icsi.berkeley.edu/~storn/TR-95-012.pdf) for tuning Fuzzy Inference Systems.

## Code Structure

- `anfis.py`: contains a python ANFIS implementation.
- `diffevo.py`: contains a python implementation of the Differential Evolution algorithm (based on [this tutorial](https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/)).
- `fobj.py`: contains several objective functions.
- `mackey.py`: contains an example that uses Differential Evolution for tuning an ANFIS for the prediction of the Mackey Glass series. This example trains the system on 1500 points of the series and plots the real vs. predicted series.

## Requirements
Known dependencies:
- Python (3.5.5)
- Numpy (1.14.2)
- Matplotlib (2.2.2)

## TODO:
- Implement membership functions other than Gaussians.
- Implement other evolutionary algorithms for tuning Fuzzy Systems (I would like to implement the Covariance Matrix Adaptation Evolution Strategy).