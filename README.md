# RectifiedFlow
The smallest and the easiest RectifiedFlow Implementation based on SD3 paper.

## Requirements
- Python 3.12
- Install pytorch, lightning, cv2, numpy, einops (supports pytorch 2.5). Very easy to install.

## Features
- Supports advanced config file instantiation, supports methods and references in .yaml files.
- Uses the same UNet model from stable diffusion.
- Uses linear flow **xt = (1-t) * x0 + t * x1** where x0 is our data distribution and x1 is random noise.
- LogNormalSampler, log normal distribution implemented for time steps.
- Supports Ema weights.

# Run
- Before running it, please fill in config and dataset.py file.
- Just use main.py!

This repo has been inspired by SD3 paper! This repo doesn't relate to any code base except latent diffusion models repo, written from scratch all manually. 

If you use this repo in your work, don't forget to cite me. Thanks!
