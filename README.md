# Computer Vision Learning Project

## Scope

For this project project, I wanted to get familiar with a couple of new things:

1. **Setting up a development environment using the Nix Package Manager**: The Nix Package Manager uses the declarative language Nix to produce reproducible development environments. I found that having the right system dependencies can be a pain sometimes. For example, one of your project might need a newer version of a library. By updating it system wide, it might break other projects not compatible. With Nix, you define which packages to use in your environment and you can pin the versions so that you will always get a specific version of a library, and so, without affecting the overall system. I wanted to explore this as I feel it could help eradicate the "but it works on my machine" problem in a dev team. It could also streamline the integration of a new employee since you could get a working development quickly and without the need to remember every little things that were customised for the company's needs.
2. **Build and train a simple CNN model to solve the SVHN dataset (the full house numbers with bounding boxes, not the digits)**: I wanted to build a simple CNN model with a LSTM on top in order to predict an arbitrary number of bounding boxes and digits "dynamically" (as opposed to YOLOv1 for example that predicts a fixes amounts of bounding box all the time). The idea was not to get a new SOTA result, but to have a model with relatively good performance to try and prune (and maybe quantize) to make it as small and efficient as possible without losing too much accuracy. I could've used a model in Huggingface, but for this project I wanted to experiment at a lower level to get a better understanding.
3. **Use Pytorch Lightning for the training logic boilerplate**: Wanting to avoid HuggingFace here but wanting a similar training API to avoid writing the whole training logic boilerplate, I wanted to explore PyTorch Lightning. It streamlines the training of models using pytorch and it is very powerful (e.g distributed training, model precision control, global seeding, etc.) so I wanted to get to know this tool.
4. **Implement the hessian-based pruning algorithm in [Hessian-Aware Pruning and Optimal Neural Implant](https://arxiv.org/pdf/2101.08940)**: I wanted to get a feel of different pruning method. LeCun was the first to measure the usefulness of neurons using second order gradients. This paper built on this and offered an algorithm I found clever and seemed to be a good starting point to understand pruning at a lower level. I also didn't want to simply do unstructured magnitude pruning as I felt it was too "simplistic" and naive to really work (which I later confirmed with [Pruning vs Quantization: Which is Better?](https://arxiv.org/pdf/2307.02973))

## Current state

### Nix shell for development environment
It works, but the (stable) nixpkgs version is based on the system's channel, which can change if the channel is updated. Ideally, like I did it for the unstable channel, I would need to point to a specific commit sha to guarantee the reproducibility.

The biggest problem was managing python packages independently of nix. I'm using NixOS, and on NixOS, there is no global `/usr/lib` directory. Everything is isolated, so using a tool like pyenv was painful to make work because it would never find the dependencies to build python. The intended way is to define the python package we need in the python package derivation, but I didn't want to depend on nix for the python environment because I know it is likely in the real world that other devs might use a different development environment. I ended up finding a nix package repository with all the python version to replace pyenv and define whatever python version I needed in my `shell.nix`. I also added `uv` to my dev env to manage the python dependencies afterward. Again, because of NixOS quirks, I needed to create the virtual environment with `venv` so that the site package was copied and not simply linked and then uv worked nicely.

### CNN Model
I ended up using a resnet base with a small 1 layer LSTM on top. I tried many different configurations, but somehow a smaller resnet (e.g. resnet-18) and a smaller LSTM seemed to generalize better. The CNN output is flatten and repeated for each "timestep" and then fed into the LSTM. The output of the LSTM is a vector containing the bounding box coordinate, the number class logits and the oes logit. The problem is that I could never get good results out of the model. I tried modifying the task to simply predicting the center of the bounding boxes and it performed well, but when it needs to predict the bounding boxes and the numbers, it failed. My hypothesis is that the LSTM models the distribution of number sequences in the images, which is somewhat irrelevant here as each number should be considered independently. This hypothesis came to me as I was probing the model at each epoch and I could see the model would default to predict bounding boxes in a particular pattern which was repeated for each prediction (in the early epochs). My initial idea was that the by repeating the flattened CNN output at each "timestep", the LSTM would "map" the CNN feature space to the amount of bounding boxes and would always know what bounding box was already predicted. In the end, as I was saying, I think it also tries to model the number sequences distribution (meaning that if all the training addresses would start with a 1, but all the test adresses would start with a 2, the model would perform poorly)

### Pruning
TODO