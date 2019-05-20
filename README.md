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
  
### To make changes to the layers, check the PATH_TO_/Mask-R-CNN/mrccn/model.py file.
