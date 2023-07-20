# Contrails Segmentation
This repository is an attempt to develop a competing solution for [this](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/overview/description) Kaggle competition.

## Problem Statment:
This competition on Kaggle is a segmentation challenge. The dataset contains a bunch of satellite images of dimension 256x256. A bunch of such satelite images form a record and each record has a corresponding ground truth image which is a binary mask where 1 denotes presence of a contrail and 0 denotes its absence.

### Setup
Please ensure that the following packages has been installed.
* Python3
* Tensorflow 1.12.x
* OpenCV 4.x
* NumPy
* pandas
* tqdm
* imutils

### Dataset
If you look at the dataset on Kaggle, you will see that the data is stored in NumPy (.npy) files. These NumPy files are nothing but normalised NumPy arrays. Due to the size of this dataset (approx. 450 GB), I do not recommend downloading the `train` dataset. Instead, a competing member has already converted those `np.float32` arrays into `np.uint8`. Please click [here](https://www.kaggle.com/datasets/thejavanka/google-research-identify-contrails-preprocessing) to download this dataset. 
Once downloaded, please download the `validation` dataset as it is from the competition page and extract it.
Now we are ready to generate `.jpg` images from those NumPy arrays. To do so, please run the below script,
```commandline
python image_generator.py
```

The above script will convert those `.npy` arrays into `jpg` images. This script will store the training data in `data` directory and validation data in `val_data` directory. Within these folders, there are two additional folders, viz. 
* `images` - This folder contains the satellite images.
* `ground_truth` - This folder contains the ground truth for the corresponding satellite images.

### Models
In order to successfully compete in this competition, I have developed three different segmentation models. The source code of these models resides inside `model` package. Following models are developed
1. UNet
2. PSPNet (with ResNet50 as backbone)
3. DeepLabv3 (with UNet encoder as the backbone)

### Training
In order to train a model, simply run the following command,
```commandline
python main.py
```

### Testing
Please download the `test` dataset from the competition page and extract it. Once extracted, run the below command,
```commandline
python test.py
```

**Note: Please train a model using the above script before running testing. Once a model has been trained, it will be saved in `saved_model` directory. The `test.py` script will load the model from this directory.**