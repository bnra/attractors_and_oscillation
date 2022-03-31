# Attractors and Oscillation

## Description
Emergence of Attractor Dynamics in Stochastically Synchronized Networks 
and Interplay between Attractor Dynamics and Oscillatory Dynamics
- Hopfield-like Conductance Scaling in E-I Networks with EIF neurons
- Stochastic Synchronization
- Gamma-breadth snapshots of network activity as network state

## Documentation 
Documentation is provided in ./document.pdf.

Managing documentation:
- generate html documentation

    $sphinx-build -b html .docs docs

- view documentation
    
    $firefox docs/index.html

- generate single pdf file (requires latexmk, texlive-latex-recommended, texlive-latex-extra)
   
    $sphinx-build -b latex .docs docs_pdf

- test documentation code snippets
    
    $sphinx-build -b doctest .docs docs

## Basic Usage ./notebooks/oscillation_poisson_input.ipynb
- running simulations
- analyzing data
- plots for exploring attractor and oscillatory dynamics


## Running simulations via cli (batch mode - multiprocessed)
- values can be specified as
  - as single value:                  `x`
  - comma separated list of values:   `a,d,g`
  - slice (start,end,step):           `[start:end:step]` 
- examplary cmd 
    $ python mp_run.py --sim eif_attr_stim --path path/to/save_directory --simtime 2250.0 --perturbation [0.0:0.205:0.1] --norm 4.25 --weighted True,False


## Installing dependencies
- apt dependencies: build-essential, python3-tk
- python3 -m pip install -r requirements.txt

## Testing
- to run tests
    
    $python -m unittest

- see test/Readme.md for more info
