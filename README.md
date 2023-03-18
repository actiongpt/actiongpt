<p align="center">

  <h1 align="center">Action-GPT: Leveraging Large-scale Language Models for Improved and Generalized Action Generation</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/sai-shashank-54288219b"><strong>Sai Shashank Kalakonda</strong></a>
    ·
    <a href="https://shubhmaheshwari.github.io/"><strong>Shubh Maheshwari</strong></a>
    ·
    <a href="https://ravika.github.io/"><strong>Ravi Kiran Sarvadevabhatla</strong></a>
  </p>
  
  <h2 align="center">ICME 2023</h2>
  <div align="center">
  </div>

  <a href="">
    <img src="./images/teaser.gif" alt="Logo" width="100%">
  </a>

<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href='https://arxiv.org/abs/2211.15603'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://actiongpt.github.io/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'><br></br>
  </br>
    
   
  </p>
</p>


[Action-GPT](http://actiongpt.github.io) provides a plug and play framework for incorporating Large Language Models (LLMs) into text-based action generation models.

For more details please refer to the [Paper](https://arxiv.org/abs/2211.15603) or the [project website](https://actiongpt.github.io/).


## Table of Contents
  * [Description](#description)
  * [Getting Started](#getting-started)
  * [Running the Demo](#demo)
  * [Training](#training)
  * [Sampling and Evaluation](#sampling-evaluation)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)


## Description

This implementation:

- Can generate motion sequences using Action-GPT framework on TEACH, TEMOS and MotionCLIP for arbitrary text descriptions provided by users.
- Can retrain Action-GPT on TEACH, TEMOS and MotionCLIP, allowing users to change details in the training configuration.


## Getting started

To install the dependencies please follow the next steps:

- Clone this repository: 
    ```Shell
    git clone https://github.com/actiongpt/actiongpt.git
    cd actiongpt
    ```
- Install the dependencies of respective models by following the steps below:
    -   Action-GPT-TEACH
        -  ```
            cd Action-GPT_TEACH_k-4/
            ```
        -   Install DistillBERT and requirements from [here](https://github.com/athn-nik/teach#getting-started)
        -   Download and setup the data as mentioned [here](https://github.com/athn-nik/teach#data)

    -   Action-GPT-TEMOS
        -  Since TEACH is an extension of TEMOS, the same installation and data setups used for Action-GPT-TEACH can be used here.

    -   Action-GPT-MotionCLIP
        -  ```
            cd Action-GPT_MotionCLIP_k-4/
            ```  
        -   Install requirements from [here](https://github.com/GuyTevet/MotionCLIP#getting-started)
        -   Download and setup the data as mentioned [here](https://github.com/GuyTevet/MotionCLIP#1-create-conda-environment)

- After installing the dependencies of the respective models, install openai as mentioned below to use GPT,
    ```Shell
  pip install openai
    ```

#### GPT descriptions
- We provided `gpt3_annotations.json` for all the models which consists of the gpt3 descriptions for the train and test action phrases of the respective data loaders.
- OpenAI API key
  - OpenAI API key need to be provided to generate GPT-3 descriptions for the action phrases which doesn't exist in `gpt3_annotations.json`.
  - Update your API-Key in `gpt_annotator.py`


## Running the Demo
- Action-GPT-TEACH (or) Action-GPT-TEMOS
  - Check out the steps to run the demo on any arbitary text descriptions from [here](https://github.com/athn-nik/teach#running-the-demo)
  - The `path/to/experiment` directory is `pretrained_model` in the respective `Action-GPT_TEACH_k-4` or `Action-GPT_TEMOS_k-4` directory, which can be downloaded from [here](https://zenodo.org/record/7747323#.ZBW4YdJBxuT).
  - NOTE : As Action-GPT-TEMOS is trained for single text descriptions, the demo can be executed for only single text prompts. 

- Action-GPT-MotionCLIP:
  - Check out the steps to generate motion from text [here](https://github.com/GuyTevet/MotionCLIP#1-text-to-motion)
  - The `./exps/paper-model` directory is `pretrained_model` in `Action-GPT_MotionCLIP_k-4` directory, which can be downloaded from [here](https://zenodo.org/record/7747323#.ZBW4YdJBxuT)

## Training
- Action-GPT-TEACH (or) Action-GPT-TEMOS
  - Check out the steps to train the model from [here](https://github.com/athn-nik/teach#training)
  
- Action-GPT-MotionCLIP:
  - Check out the steps to train the model from [here](https://github.com/GuyTevet/MotionCLIP#train-your-own)

## Sampling and Evaluation
- Action-GPT-TEACH (or) Action-GPT-TEMOS
  - Check out the steps to sample and evaluate the model from [here](https://github.com/athn-nik/teach#sampling--evaluation)
  - NOTE : For Action-GPT-TEMOS there are no align and slerp parameters to be passed as the model is trained for single text descriptions.
  
- Action-GPT-MotionCLIP:
  - Follow the below command to sample from the test set. The below command creates two directories named `ground_truth` and `action_gpt` in the path provided for the parameter `generations`. Both the directories `ground_truth` and `action_gpt` contain the npy files of the motion sequences corresponding to the test set text descriptions. Using the npy files created and the evaluation code provided in [TEACH](https://github.com/athn-nik/teach#sampling--evaluation) one can generate the metrics as provided in the paper.
  - ```python sampling.py ./exps/paper-model/checkpoint_0100.pth.tar --generations path/to/store/sampled/generations```

## Citation

```
@inproceedings{Action-GPT,
  title = {Action-GPT: Leveraging Large-scale Language Models for Improved and Generalized Action Generation},
  author = {Kalakonda, Sai Shashank and Maheshwari, Shubh and Sarvadevabhatla, Ravi Kiran},
  booktitle = {IEEE International Conference on Multimedia and Expo ({ICME})},
  year = {2023},
  url = {https://actiongpt.github.io/}
}
```

## Acknowledgments

This work is an implementation of Action-GPT framework on T2M-models such as [TEACH](https://github.com/athn-nik/teach), [TEMOS](https://github.com/Mathux/TEMOS) and [MotionCLIP](https://github.com/GuyTevet/MotionCLIP). Many part of this code were based on the official implementation of the respective T2M-Models. We thank all the corresponding authors for making thieir code available.

This template was adapted from the GitHub repository of [GOAL](https://github.com/otaheri/GOAL).