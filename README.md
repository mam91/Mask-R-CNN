python shuriken_gun.py train --dataset=PATH_TO_/MaskRCnn/datasets/shuriken_gun --weights=last

python shuriken_gun.py train --dataset=PATH_TO_/MaskRCnn/datasets/shuriken_gun --weights=coco

python supervisely.py train --dataset=PATH_TO_/MaskRCnn/datasets/supervisely --weights=coco

***MASK R CNN DETECT AND EVAL***
python shuriken_gun.py detect --weights=last

python shuriken_gun.py eval --weights=last


python shuriken_gun.py detect --weights=PATH_TO_/MaskRCnn/logs/shuriken_gun20190407T0317/mask_rcnn_shuriken_gun_0030.h5
python shuriken_gun.py eval --weights=PATH_TO_/MaskRCnn/logs/shuriken_gun20190331T2335/mask_rcnn_shuriken_gun_0135.h5

tensorboard --logdir=PATH_TO_/MaskRCnn/logs/shuriken_gun20190409T0146

**YOLO***

flow --model cfg/yolo-xray.cfg --load bin/yolo.weights --train --annotation shuriken/annotations --dataset shuriken/Images --labels shuriken/labels.txt --load -1

flow --model cfg/tiny-yolo-voc-xray.cfg --load bin/tiny-yolo-voc.weights --train --annotation shuriken/annotations --dataset shuriken/Images --labels shuriken/labels.txt --gpu 0.7

flow --imgdir shuriken/eval --model cfg/tiny-yolo-voc-xray.cfg --load bin/tiny-yolo-voc.weights --labels shuriken/labels.txt --thresh 0.3