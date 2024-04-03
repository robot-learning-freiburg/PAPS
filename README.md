# Perceiving the Invisible: Proposal-Free Amodal Panoptic Segmentation

PAPS is a bottom approach for amodal panoptic segmentation, where the goal is to concurrently predict the pixel-wise semantic segmentation labels of visible regions of "stuff" classes (e.g., road, sky, and so on), and instance segmentation labels of both the visible and occluded regions of "thing" classes (e.g., car, truck, etc).

![Overview of PAPS Architecture](/projects/PAPS/images/overview.png)

This repository contains the **PyTorch implementation** of our RA-L'2022 paper [Perceiving the Invisible: Proposal-Free Amodal Panoptic Segmentation](https://arxiv.org/pdf/2205.14637.pdf). The repository builds on [Detectron2](https://github.com/facebookresearch/detectron2).

If you find this code useful for your research, we kindly ask you to consider citing our papers:

```
@article{mohan2022perceiving,
  title={Perceiving the invisible: Proposal-free amodal panoptic segmentation},
  author={Mohan, Rohit and Valada, Abhinav},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={9302--9309},
  year={2022},
  publisher={IEEE}
}
```

## System Requirements
* Linux 
* Python 3.9
* PyTorch 1.12.1
* CUDA 11
* GCC 7 or 8

**IMPORTANT NOTE**: These requirements are not necessarily mandatory. However, we have only tested the code under the above settings and cannot provide support for other setups.

##  Installation
Please refer to the [installation documentation](https://github.com/robot-learning-freiburg/PAPS/blob/main/projects/PAPS/docs/INSTALLATION.md) for detailed instructions.

## Dataset Preparation
Please refer to the [dataset documentation](https://github.com/robot-learning-freiburg/PAPS/main/projects/PAPS/docs/DATASET.md) for detailed instructions.

## Usage
For detailed instructions on training, evaluation, and inference processes, please refer to the [usage documentation](https://github.com/robot-learning-freiburg/PAPS/blob/main/projects/PAPS/docs/USAGE.md).


## Pre-Trained Models
Pre-trained models can be found in the [model zoo](https://github.com/robot-learning-freiburg/PAPS/blob/main/projects/PAPS/docs/MODELS.md).

## Acknowledgements
We have used utility functions from other open-source projects. We espeicially thank the authors of:
- [Detectron2](https://github.com/facebookresearch/detectron2)

## Contacts
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
* [Rohit Mohan](https://github.com/mohan1914)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.

