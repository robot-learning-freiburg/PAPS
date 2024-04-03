# USAGE

This guide covers the basic commands for training and evaluating models within the project. Ensure you have followed the installation and dataset preparation steps before proceeding.

## Training

To train a model on 8 GPUs, use the following command. Make sure you are in the correct directory and specify your configuration file:

```bash
cd /path/to/detectron2/projects/PAPS
python amodal_train_net.py --config-file config.yaml --num-gpus 8
```

## Evaluation

For model evaluation, execute the command below. This will use the specified model checkpoint for evaluation. Ensure the MODEL.WEIGHTS path is correctly set to the location of your model checkpoint:

```bash
cd /path/to/detectron2/projects/PAPS
python amodal_train_net.py --config-file config.yaml --eval-only --model-weights /path/to/model_checkpoint
```

For tasks related to panoptic segmentation, replace amodal_train_net.py with train_net.py in the commands above. This script is tailored for the panoptic segmentation task:
```bash
cd /path/to/detectron2/projects/PAPS
python train_net.py --config-file config.yaml --num-gpus 8
# For training

cd /path/to/detectron2/projects/PAPS
python train_net.py --config-file config.yaml --eval-only --model-weights /path/to/model_checkpoint
# For evaluation
```