# SIIM-ISIC Melanoma Classification

![Python](https://img.shields.io/badge/python-3.x-orange.svg)
![Type](https://img.shields.io/badge/Deep-Learning-red.svg)
![Type](https://img.shields.io/badge/Type-PYTORCH-green.svg)
![Type](https://img.shields.io/badge/Type-KERAS-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

## Introduction

<p>Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least
common skin cancer.
The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020.
It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.<br>
Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma.
Existing AI approaches have not adequately considered this clinical frame of reference. Dermatologists could enhance their diagnostic accuracy if detection algorithms take into account “contextual” images within the same patient to determine which images represent a melanoma. If successful, classifiers would be more accurate and could better support dermatological clinic work.<br>
As the leading healthcare organization for informatics in medical imaging, the Society for Imaging Informatics in Medicine (SIIM)'s mission is to advance medical imaging informatics through education, research, and innovation in a multi-disciplinary community. SIIM is joined by the International Skin Imaging Collaboration (ISIC), an international effort to improve melanoma diagnosis. The ISIC Archive contains the largest publicly available collection of quality-controlled dermoscopic images of skin lesions.<br>
Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.</p><br>

# Dataset

This dataset contains a balanced dataset of images of benign skin moles and malignant skin moles.
The data consists of two folders with each 1800 pictures (224x244) of the two types of moles.
You can ![Download](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign) the dataset from Kaggle.

# Implementation of Image Classification

The implementation is done with Keras and Pytorch.In Keras, I did Image Data Augmentation using ImageDataGenerator and creates a CNN model to achieve an accuracy of 77% on the test dataset.
In PyTorch, I created two files one with simple CNN architecture and achieved an accuracy of 81% on the test dataset while on the other file I use Densenet161 model and achieved an accuracy of 85% on test data in 5 epochs.

# Densenet 161 Architecture

Dense Networks are a relatively recent implementation of Convolutional Neural Networks, that expand the idea proposed for Residual Networks, which have become a standard implementation for feature extraction on image data.
Similar to Residual Networks that add a connection from the preceding layer, Dense Nets add connections to all the preceding layers, to produce a Dense Block. The problem that Residual Nets addressed was the vanishing gradient problem that was due to many layers of the network. This helped build bigger and more efficient networks and reduce the error rate on classification tasks on ImageNet.
So the idea of Densely Connected Networks, is that every layer is connected to all its previous layers and its succeeding ones, thus forming a Dense Block. The authors of the paper reported that their implementation performed better than previous state of the art on classification on ImageNet, which seems compelling.

![](https://miro.medium.com/max/3000/1*04TJTANujOsauo3foe0zbw.jpeg)



<b>This GIF can give us an idea how Densenet works</b>

![](https://miro.medium.com/max/875/1*rv-_-8LemZW6m9YrWBQx9w.gif)

# What problem DenseNets solve?

Counter-intuitively, by connecting this way DenseNets require fewer parameters than an equivalent traditional CNN, as there is no need to learn redundant feature maps.
Furthermore, some variations of ResNets have proven that many layers are barely contributing and can be dropped. In fact, the number of parameters of ResNets are big because every layer has its weights to learn. Instead, DenseNets layers are very narrow (e.g. 12 filters), and they just add a small set of new feature-maps.
Another problem with very deep networks was the problems to train, because of the mentioned flow of information and gradients. DenseNets solve this issue since each layer has direct access to the gradients from the loss function and the original input image.



# Acknowledgements
All the rights of the Data are bound to the ISIC-Archive rights (https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main). I do not take any responsibility for the right-infringement of any kernels.