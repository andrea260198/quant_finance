# Quantitative Finance Package


## Introduction

This repo contains some example of quantitative finance topics based on what I 
learned at EPFL during the Master in Financial Engineering (MFE).


## Observations

Python is not the best language to use for a big quant finance repo, because the
fact that it is not statically typed makes it more difficult to understand in 
big teams and big repos, as well as, less robust (more prone to runtime errors 
and crashes while the model is running).

Despite these limitations, the static checker Mypy does a good job in preventing
many errors that might arise during runtime.


## CI pipeline

Every time `git push` is executed, a static type check is performed using 
**mypy**.


## CD pipeline

In the future I would like to develop a CD pipeline to run the model in a 
server.


## Multiprocessing

Note that **multiprocessing** module is used to make full use of multi-core 
CPUs.


## Testing

I also created a directory for unit testing of the code.


## Risk Management


## Option pricing

With `plot_approx_error_european_option.py` script I want to show that binomial
tree approximation of European options price has an approximation error which is
**O(dt)**, with **dt** being the timestep.


## Interest rate

With `plot_yield_curve.py` script I want to show how the exact formula for the 
yield curve can be approximated by using Monte Carlo. The Monte Carlo simulation
make use of "multiprocessing" package. Every simulation is run in a separated
process.

