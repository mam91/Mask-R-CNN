## Commands to train and evaluate are as follows:

#### 1. In terminal, navigate to Mask-R-CNN/samples/gdxray directory

#### 2. To train, execute one of the following commands:

  To train a dataset on weights pretrained on the COCO dataset: <br><br>
  ` python gdxray.py train --dataset=PATH_TO_/Mask-R-CNN/datasets/gdxray --weights=coco `

  To train a dataset on weights pretrained on the ImageNet dataset: <br><br>
  ` python gdxray.py train --dataset=PATH_TO_/Mask-R-CNN/datasets/gdxray --weights=imagenet `

  To resume training from a pevious execution: <br><br>
  ` python gdxray.py train --dataset=PATH_TO_/Mask-R-CNN/datasets/gdxray --weights=last `

#### 3. To evaluation a trained model, execute one of the following commands:

  To evaluate last model trained: <br><br>
  ` python gdxray.py eval --weights=last `
  
  To evaluate a specific set of weights: <br><br>
  ` python gdxray.py eval --weights=/path/to/weights.h5 `

  Example weights location = PATH_TO_/Mask-R-CNN/logs/shuriken_gun20190407T0317/mask_rcnn_shuriken_gun_0030.h5

## To run tensorboard, execute the following on a log folder
  ` tensorboard --logdir=/path/to/log/folder `
  
  Example log directory = PATH_TO_/Mask-R-CNN/logs/shuriken_gun20190409T0146
  
## Modifying layers

To make changes to the layers, check the PATH_TO_/Mask-R-CNN/mrccn/model.py file.

## Steps to create new training flow with new dataset
Follow these steps if you want to preserve the gdxray dataset and scripts and create a different workspace.

#### 1. Annotate the dataset
Annotations must at least contain the following fields:
  * class label
  * id (can be filename)
  * path (full path to image)
  * height
  * width
  * segmentation mask (can be stored in any format as long as it can be converted to a binary mask when loading) 
  
#### 2. Create directory structure for new dataset
1. Create dataset folders under datasets:  
  */Mask-R-CNN/datasets/dataset_name/
  */Mask-R-CNN/datasets/dataset_name/train
  */Mask-R-CNN/datasets/dataset_name/val
  
2. Move training data
  *Move training image-annotation pairs into /Mask-R-CNN/datasets/dataset_name/train
  *Move test image-annotation pairs into /Mask-R-CNN/datasets/dataset_name/val
  
#### 3. Create directory structure for new scripts if necesssary
1. Create copy of gdxray folder in /Mask-R-CNN/samples
2. (Optional) Rename folder and script file (gdxray.py)
3. (Optional) Rename class objects such as GDXrayConfig and GDXrayDataset in the script file (previously gdxray.py).
4. Update references in both jupyter notebooks to reference the new dataset and configs

#### 4. Validate data
To ensure data is in correct format, run the jupyter notebook "inspect_data.ipynb".  This notebook will load the dataset and the ground truth masks.  Any issues should be evident.

##### NOTE: The gdxray dataset and scripts should work out of the box.  You can simply change the training data and, as long as the annotations are in the same format, successfully train on different data.

