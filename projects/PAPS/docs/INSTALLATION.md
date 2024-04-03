# Installation

This guide provides a step-by-step process for setting up and installing PAPS. Follow these instructions to prepare your environment.

## Prerequisites

Ensure you have Conda installed on your system. If not, refer to the [Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Steps

1. **Create and Activate a Conda Virtual Environment**

    Start by creating a Conda environment named `paps` with Python 3.9. Activating this environment will isolate your PAPS installation and dependencies.

    ```shell
    conda create --name paps python=3.9
    conda activate paps
    ```
2. **Install PyTorch**
    Install PyTorch, torchvision, and torchaudio with CUDA 11.3 support. This step ensures that you have the correct versions for running PAPS.

    ```shell
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

3. **Download the PAPS Repository**
    Clone the PAPS repository from GitHub and navigate into the project directory.

    ```shell
    git clone https://github.com/robot-learning-freiburg/PAPS.git
    cd PAPS
    ```

4. **Install Dependencies**
    Install the required Python packages, including a specific version of PyYAML and dependencies from the panopticapi and OpenCV.

    ```shell
    pip install pyyaml==5.1
    pip install git+https://github.com/cocodataset/panopticapi.git
    pip install opencv-python
    pip install git+https://github.com/robot-learning-freiburg/amodal-panoptic.git
    ```

5. **Install PAPS**
    Finally, install PAPS in editable mode to facilitate easy updates.
    ```shell
    pip install -e .
    ```
    
