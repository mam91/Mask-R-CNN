
"""
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python gdxray.py train --dataset=/path/to/dir/Mask-R-CNN/datasets/gdxray --weights=coco

    # Resume training a model that you had trained earlier
    python gdxray.py train --dataset=/path/to/dir/Mask-R-CNN/datasets/gdxray --weights=last

    # Train a new model starting from ImageNet weights
    python gdxray.py train --dataset=/path/to/dir/Mask-R-CNN/datasets/gdxray --weights=imagenet

    # Calculate AP of trained model
    python gdxray.py eval --weights=last
"""
import time
import os
import sys
import json
import random
import re
import datetime
import numpy as np
import skimage.draw
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
import tensorflow as tf
import cv2
from skimage.measure import find_contours
import skimage
from skimage.measure import compare_ssim as ssim

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
GDXRAY_WEIGHTS_PATH = "//mask_rcnn_gdxray.h5"  

############################################################
#  Configurations
############################################################

############################################################
#  A new config class must be created that derives from the base Config class 
#  Config settings will be modified within this new class instead of the base config class
############################################################
class GDXrayConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "gdxray"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1 # Background + shuriken + gun

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

############################################################
#  A new dataset class must be created that derives from the base Dataset class
#  This is how we control the loading of image and annotation data
#  must override load_mask and image_reference functions
#  The load_dset function is where we customize how we load our custom dataset
############################################################
class GDXrayDataset(utils.Dataset):

    def load_dset(self, dataset_dir, subset):
        """Load a subset of the GDXray dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only two classes to add.
        self.add_class("shuriken", 1, "shuriken")
        self.add_class("gun", 2, "gun")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        #Iterate through our annotations, fetching image meta data
        for mask_file in os.listdir(dataset_dir):
            if(mask_file.endswith('.json')):
                mask_path = os.path.join(dataset_dir, mask_file)
                annotations = json.load(open(mask_path))
                annotations = list(annotations.values())
                for a in annotations:
                    if type(a['regions']) is dict:
                        polygons = [r['shape_attributes'] for r in a['regions'].values()]
                    else:
                        polygons = [r['shape_attributes'] for r in a['regions']]
                    
                    width = a['file_attributes']['width']
                    height = a['file_attributes']['height']

                    # if width and height are not in annotation, load image and get height and width
                    # image_path = os.path.join(dataset_dir, a['filename'])
                    # image = skimage.io.imread(image_path)
                    # height, width = image.shape[:2]

                    image_path = os.path.join(dataset_dir, a['filename'])

                    label = a['label']

                    self.add_image(label,
                        image_id=a['filename'],  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons)
            else:
                continue

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a shurken or gun dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "shuriken" and image_info["source"] != "gun" :
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. 
        if image_info["source"] == "shuriken":
            #return array of ones
            return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        elif image_info["source"] == "gun":
             return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32) + 1

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shuriken":
            return info["path"]
        elif info["source"] == "gun":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = GDXrayDataset()
    dataset_train.load_dset(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = GDXrayDataset()
    dataset_val.load_dset(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=1,
               layers='heads')

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the noebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def eval_model(model, inference_config, weights_path):
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets/gdxray")
    dataset = GDXrayDataset()
    dataset.load_dset(DATASET_DIR, "val")
    dataset.prepare()

    #compute AP
    compute_ap_range(dataset, model, inference_config)

    #compute AP by itself
    compute_ap(dataset, model, inference_config)

def compute_ap(dataset, model, inference_config):
    image_ids = dataset.image_ids #np.random.choice(dataset.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image_name = dataset.image_info[image_id]['id'] 
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config,image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        
        # Run detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))

def compute_ap_range(dataset, model, inference_config):
    image_ids = dataset.image_ids #np.random.choice(dataset.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image_name = dataset.image_info[image_id]['id'] 
        
        print(image_name + " ap range results:")
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config,image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        
        # Run detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    
    print("mAP Range: ", np.mean(APs))

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect shurikens.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/shuriken/dataset/",
                        help='Directory of the Shuriken dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = GDXrayConfig()
    else:
        class InferenceConfig(GDXrayConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        #detect_and_color_splash(model, image_path=args.image, video_path=args.video)
        print("Removed this logic branch.  Use jupyter notebook inspect model")
    elif args.command == "eval":
        eval_model(model, config, weights_path)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
