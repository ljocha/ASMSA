# ASMSA
Analysis and Sampling of Molecular Simulations by adversarial Autoencoders

Use `nbstrioput` to commit so that diffing/merging is not nightmare.

Binder is likely to be broken right now :-(
[![Binder](https://binderhub.cloud.e-infra.cz/badge_logo.svg)](https://binderhub.cloud.e-infra.cz/v2/git/https%3A%2F%2Fgitlab.ics.muni.cz%2F485413%2FASMSA/refs/heads/kl_divergence)

## Description
TODO

## Badges
TODO

## Getting started
TODO

## Test and Deploy

Full support for distributed hyperparameter tuning is available at CERIT-SC Jupyterhub:
1. Go to https://hub.cloud.e-infra.cz/ and log in with your [Metacentrum](http://metacentrum.cz) account
1. Either click on **Start my server** for the default, or type a specific name and click **Add New Server**
1. Fill in the submit form:
    - Select an image: **Custom**
    - Custom image name: **ljocha/asmsa:2023-19**
    - Select persistent home type: **New** at the first time, **Existing** is prefered afterwards
    - Select persistent home (when *Existing* in the previous choice): pick you prefered one, or stick with the only one offered
    - I want MetaCentrum home: **no**
    - Would you like to connect project directory: **no**
    - Select number of CPUs: **2** (it's enough for the notebook, parallel workers are not counted here)
    - Memory: **8 GB** (appears to be enough for this usecase)
    - GPU: **None** (it turns out that our models are too small to leverage GPU accelleration)
1. Click on **Start**
1. Depending on the container image cache status Jupyterlab starts in few seconds (the image was cached) or several minutes (it must be downloaded)
1. click on **asmsa.ipynb** in the left panel and follow instructions in the notebook

 
## Visuals
TODO (screenshots, ...)

## Installation
TODO

## Usage
TODO

## Support
TODO

## License
TODO

## Project status
In development... No main usable version released yet
