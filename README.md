This is the repo for the ECCV 2020 paper "Describing Textures using Natural Language".  
More contents are shared on [the project webpage](https://people.cs.umass.edu/~chenyun/texture/).

Our collected annotations are in `data_api/data/image_desciptions.json`. `data_api` includes all basic data handling code. 
DTD images should be put under `data_api/data/images`. They can be downloaded from [the DTD website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Code for training and evaluating 3 baseline models are under `models/naive_classifier/`, `models/show_attend_tell/`, `models/triplet_match` respectively. 
We also include additional application and visualization code.

More documentation will be added to this repo. Feel free to contact `chenyun _AT_ cs.umass.edu` if you have any questions.
