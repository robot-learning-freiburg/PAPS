# Dataset Preparation

Ensure your datasets are organized correctly for use with the models. Below are the guidelines for preparing your datasets for both Amodal Panoptic Segmentation and Panoptic Segmentation tasks.

## Amodal Panoptic Segmentation

For amodal panoptic segmentation tasks, the dataset directory should be structured as follows:

```plaintext
DATASET
├── amodal_panoptic_seg
├── images
```

**Setting the Environment Variable**: Set the DETECTRON2_DATASETS environment variable to the path containing your dataset folder. This allows Detectron2 to locate your dataset.
```shell
export DETECTRON2_DATASETS=/path/to/dataset/containing/folder
```

## Panoptic Segmentation
**Cityscapes Dataset**: For those using the Cityscapes dataset for panoptic segmentation, please refer to the official [Detectron2 tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html#expected-dataset-structure-for-cityscapes). This tutorial provides detailed instructions on how to prepare the Cityscapes dataset for use with Detectron2.
