# MLMI VQA model

*Explaining Medical Image Classifiers with Visual Question Answering Models:*<br>
a Visual Question Answering (VQA) Model trained  on medical data


# Description
Deep learning has shown promising potential for Medical Image Classification and Diagnosing.  But added to the limitations of annotated training data in the medical domain, explanations for the models's predictions are also desired in this field of application.

Using [Flamingo, a Visual Language Model for Few-Shot Learning](https://doi.org/10.48550/ARXIV.2204.14198), we leverage big pre-trained language models and vision encoders to build a new VQA model that can answers question for Xray images.

You can find out available pre-trained Models under the [following link](https://drive.google.com/drive/folders/1WYwDez52QNDBsYQPPh5tsSDyV1hs8eJs?usp=sharing)

# Requirements
- [Python](https://www.python.org/downloads/) (Python `>= 3.8`)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) (Virtual Environment)

# Table Of Contents
-  [Model Architecture](#model-architecture)
-  [Training and Testing](#training-and-testing)
-  [Getting Started](#getting-started)
-  [Demo and Deploy](#demo-and-deploy)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)


# Model Architecture
Using Flamingo's architecture elements, we built a model capable of taking an Xray image and any question as inputs in order to generate an answer to the asked question. A simplified overview of our model architecture is given in the following figure :   

![image](docs/imgs/flamingo-own.jpg)

# Training and Testing
TOOD: For the training process, we trained our model for ..... on ...... using the dataset .............
TODO: soe test results

# Getting started

To make it easy for you to get started with our model, here's a list of recommended next steps:

- [ ] Clone this repository into a local folder.
```
cd local/path
git clone https://gitlab.lrz.de/CAMP_IFL/diva/mlmi-vqa
```
- [ ] Setup the python virtual environement using `conda`.

```
conda env create -f environment.yml
conda activate mlmi_caghan
```
- [ ] Check the notebooks for usage examples


# Demo and Deploy

You can check and try out our model in our demo page using this [Hugging Face Space](https://huggingface.co/spaces/alamellouli/demo-mlmi-vqa)

***

## Future Work
TODO


## Contributing
At the moment still closed for contributions.


## Acknowledgments

Authors: **Fabian Scherer - Andrei Mancu - Alaeddine Mellouli - Çağhan Köksal**

We thank the MLMI team and both Matthias Keicher and Kamilia Zaripova for their help and support.

## License
Private Repository until further development.

