# This Looks Like That, Because ... Explaining Prototypes for Interpretable Image Recognition

This repository contains Python code and Jupyter notebooks to explain prototypes learned by the Prototypical Part Network [(ProtoPNet)](https://github.com/cfchen-duke/ProtoPNet). Only visualizing prototypes can be insufficient for understanding what a prototype exactly represents, and why a prototype and an image are considered similar. We improve interpretability by automatically quantifying the influence of color hue, shape, texture, contrast and saturation in a prototype. 

This is a fork of [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) with a new folder `preprocessing` and an extra notebook `prototype_activation.ipynb`. 

<!-- Corresponding paper: ["This Looks Like That, Because ... Explaining Prototypes for Interpretable Image Recognition"]() -->

![Overview of explaining prototypical learning for image recognition.](overview_explaining_prototypes.png "Explaining Prototypes")
Overview of explaining prototypical learning for image recognition. 1) Dataset with bird species for image recognition. 2) Image classification based on similarity with prototypical patches obtained by [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet). 3) Our contribution: Explaining why the classification model considered the test image and the prototype to be similar. Importance scores quantify the importance of visual characteristics.

## Usage
1. Download and unpack the dataset [CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. Run `preprocessing/crop_images.py` to crop images according to the provided bounding box annotations
3. Create images with modified hue, shape, texture, contrast and saturation by running `preprocessing/create_modified_datasets.ipynb`.
4. Train a ProtoPNet as presented by the original authors: https://github.com/cfchen-duke/ProtoPNet with `main.py` and the appropriate parameters in `settings.py`.
5. Run `prototype_activation.ipynb` to get global and local importance scores which explain prototypes.