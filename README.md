# Object-Detection-with-SSD
This is my first object detection project built and run locally using SSD, with references to the implementation of SSD from [amdegroot](https://github.com/amdegroot/ssd.pytorch/commits?author=amdegroot) and [huutrinh68](https://github.com/huutrinh68/dl-pytorch) with minor adjustment. SSD is an old object detection algorithm, released in 2016, but it still covers a lot of useful ideas of how an object detection algorithms may work.
# Outline
1. Data Processing
2. SSD Modules
3. Usage

## Data Processing
### Dataset
Here, I am using the VOC2012 dataset for training my SSD model. You can download it automatically by using the **prepare_data.py**, which will create a data folder and install the VOC2012 dataset from the internet and untar the dataset (in case the dataset does not exist).
```
python prepare_data.py
```
After having our dataset, we continue to use **make_data_path.py** (which will create lists, each containing images and corresponding annotation links for the later process). The **extract_anno_info.py** extracts information from annotation links, which are in xml format. The **transform.py** uses some basic transformations for data augmentation. All of the 3 files will then be used in the **dataset.py**, which will create train and val DataLoaders containing images after transformation and the annotations (bounding boxes coordinate, class) for training the model.

## SSD Modules
The SSD Modules contain VGG layers, the extra feature layers, the L2Norm, and loc and conf layers.
<p align = "center">
<img src = "https://github.com/TaiQuach123/Object-Detection-with-SSD/blob/main/%2308_SSDmodel.png" height=400/>
</p>

View the **model.py** for more architecture information

## Usage
Firstly, clone this repository by executing:
```
git clone https://github.com/TaiQuach123/Object-Detection-with-SSD.git
```

Then, get the pre-trained weights for the VGG part as initial weights to help the model converge faster

```
cd './data'
mkdir 'weights'
cd './weights'
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
cd ../..
```

Finally, executing the **train.py**:
```
python train.py
```



