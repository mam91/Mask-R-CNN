import time
"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

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
#from samples.shuriken_gun import shuriken_gun

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
SHURIKEN_WEIGHTS_PATH = "//mask_rcnn_shuriken_gun.h5"  
TRAIN_DIR = ""
STAGE_DIR = ""

############################################################
#  Configurations
############################################################


class ShurikenConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "shuriken_gun"

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

class ShurikenDataset(utils.Dataset):

    def load_shuriken(self, dataset_dir, subset):
        """Load a subset of the Shuriken dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("shuriken", 1, "shuriken")
        self.add_class("gun", 2, "gun")

        # Train or validation dataset?
        assert subset in ["train", "val", "staging"]
        dataset_dir = os.path.join(dataset_dir, subset)

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
                    
                    
                    #width = (a['file_attributes']['width'])
                    #height = (a['file_attributes']['height'])
                    image_path = os.path.join(dataset_dir, a['filename'])
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    image_path = os.path.join(dataset_dir, a['filename'])

                    if subset == 'staging':
                        label = 'shuriken'
                    else:
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
        else:
            super(self.__class__, self).image_reference(image_id)


annotations_added = 0
annotations_added_ssim = 0

def add_annotation(passed_ssim):
    global annotations_added
    annotations_added = annotations_added + 1

    if passed_ssim:
        global annotations_added_ssim
        annotations_added_ssim = annotations_added_ssim + 1

def get_annotation_counts():
    global annotations_added
    global annotations_added_ssim
    return annotations_added, annotations_added_ssim

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ShurikenDataset()
    dataset_train.load_shuriken(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShurikenDataset()
    dataset_val.load_shuriken(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=120,
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
    SHURIKEN_DIR = os.path.join(ROOT_DIR, "datasets/shuriken_gun")
    dataset = ShurikenDataset()
    dataset.load_shuriken(SHURIKEN_DIR, "val")
    
    STAGE_DIR = os.path.join(SHURIKEN_DIR, "staging/shuriken")

    # Must call before using the dataset
    dataset.prepare()
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Or, load the last model you trained
    #weights_path = model.find_last()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    #compute AP
    compute_ap_range(dataset, model, inference_config)

    #compute AP by itself
    compute_ap(dataset, model, inference_config)

    #compute average IoU
    compute_avg_iou(dataset, model, inference_config)
    

def detect_masks(model, inference_config):
    
    SHURIKEN_DIR = os.path.join(ROOT_DIR, "datasets/shuriken_gun")
    dataset = ShurikenDataset()
    dataset.load_shuriken(SHURIKEN_DIR, "val")
    
    STAGE_DIR = os.path.join(SHURIKEN_DIR, "staging")

    # Must call before using the dataset
    dataset.prepare()
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Or, load the last model you trained
    weights_path = model.find_last()

    # Load weights
    #print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    #compute AP
    #compute_ap(dataset, model, inference_config)
    print("Iterating through images in staging directory: " + STAGE_DIR)

    #for each image in staging directory
    staging_files = os.listdir(STAGE_DIR)
    print("Total staging files: " + str(len(staging_files)))
    current_file_num = 1
    #for staging_file in os.listdir(STAGE_DIR):
    for staging_file in staging_files:
        print("Processing file number: " + str(current_file_num))
        current_file_num = current_file_num + 1
        if not staging_file.endswith('.png'):
            print("Skipping non image file" + staging_file)
            continue

        if(doesFileExist(staging_file)):
            print("File already exists in training directory, skipping")
            continue

        print("Detecting instances in staging file: " + staging_file)

        image_name = staging_file
    
        #image = cv2.imread('./B0046_0041.png', cv2.IMREAD_COLOR)
        image = cv2.imread(STAGE_DIR + '/' + image_name, cv2.IMREAD_COLOR)

        # Run object detection
        results = model.detect([image], verbose=1)

        # Check if instance was detected
        ax = get_ax(1)
        r = results[0]
        #print(r)

        N = r['rois'].shape[0]
        if not N:
            print("No instance found")
            continue
        else:
            print("Instance found")

        masked_image, contours, image_label = visualize.display_masks_only(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")

        #Trim out only mask and unnecessary area
        y1, x1, y2, x2 = r['rois'][0]
        masked_image = masked_image[y1:y2, x1:x2, :]
        bbox_image = image[y1:y2, x1:x2, :]

        #Just write an image in working directory to be validated against 
        cv2.imwrite("eval_image.png", masked_image)

        #validation_success, ssim_success = validate_result(r, bbox_image, masked_image, image_label)
        if validate_result(r, bbox_image, masked_image, image_label):
            save_to_train(image, masked_image, contours, image_name, image_label)
    
    anno_count, anno_count_ssim = get_annotation_counts()
    print("Added " + str(anno_count) + " annotations.")
    print("Added " + str(anno_count_ssim) + " ssim annotations.")

def doesFileExist(filename):
    DATASET_ROOT = os.path.join(ROOT_DIR, "datasets/shuriken_gun")
    train_dir = os.path.join(DATASET_ROOT, "train")
    file_path = os.path.join(train_dir, filename)
    return os.path.isfile(file_path)


    
def validate_result(result, orig_image_bbox, mask_image, image_label):
    #if scores are not above threshold then move on
    print(result['scores'][0])
    if(result['scores'][0] < .95):
        print("Failed to detect above threshold")
        return False

    #perform contour matching OR shaprmask matching here
    passed_ssim = False
    contourImages = getContours(orig_image_bbox)
    print("Found " + str(len(contourImages)) + " contours to match")
    if(len(contourImages) > 0):
        if not (runSSim(contourImages, mask_image)):
            print("Failed SSIM matching")
            #return False
        else:
            passed_ssim = True

    #check against cnn of masks
    cnnScore, cnnLabel = evalMaskViaCnn()
    if(cnnLabel == image_label and cnnScore > 0.95):
        print("Cnn vadliation passed for label: " + image_label)
        add_annotation(passed_ssim)
        return True

    print("Validation failed")
    return False

def runSSim(contours, mask_image):   
    mask_image = cv2.copyMakeBorder(mask_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.imwrite("orig_mask.png", mask_image)
    orig_img = cv2.imread("orig_mask.png", 0)

    import random
    test_index = random.randint(0,50000)
    for i in range(len(contours)):
        c_image = contours[i]
        c_ssim = ssim(orig_img, c_image, data_range=c_image.max() - c_image.min())
        #print("SSIM: " + str(c_ssim))
        #cv2.imwrite("./testMasks/test" + str(test_index) + "-SSIM-" + str(c_ssim) + ".png", c_image)
        test_index = test_index + 1
        if (c_ssim > .85):
            return True
    return False

#C:\Users\mmill\Documents\GitHub\Education\MaskRCnn\datasets\shuriken_gun

def getContours(image):
    test_img = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))

    image_width = test_img.shape[1]
    image_height = test_img.shape[0]

    imgray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contourImages = []

    for k in range(len(contours)):
        X = []
        Y = []
        
        for i in range(len(contours[k])):
            X.append(int(round(contours[k][i][0][0])))
            Y.append(int(round(contours[k][i][0][1])))
        
        mask = np.zeros([image_height, image_width, 1], dtype=np.uint8)

        rr, cc = skimage.draw.polygon(Y, X)
        
        mask[rr, cc, 0] = 255
        
        mask = cv2.bitwise_not(mask)
        contourImages.append(mask)
        #plt_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    return contourImages

def compute_avg_iou(dataset, model, inference_config):

    SHURIKEN_DIR = os.path.join(ROOT_DIR, "datasets/shuriken_gun")
    staging_dataset = ShurikenDataset()
    staging_dataset.load_shuriken(SHURIKEN_DIR, "staging")
    staging_dataset.prepare()
    staging_ids = staging_dataset.image_ids

    image_ids = dataset.image_ids #np.random.choice(dataset.image_ids, 5)

    APs = []
    IoUs = []
    for image_id in image_ids:
        image_name = dataset.image_info[image_id]['id']       
        mask, class_ids = dataset.load_mask(image_id)

        for staging_id in staging_ids:
            staging_name = staging_dataset.image_info[staging_id]['id']
            if staging_name == image_name:
                mask_s, class_ids_s = staging_dataset.load_mask(staging_id)
                overlaps = utils.compute_overlaps_masks(mask, mask_s)
                for i in range(len(overlaps)):
                    for k in range(len(overlaps[i])):
                        if i == k:
                            #print(overlaps[i][k])
                            IoUs.append(overlaps[i][k])
    
    IoUsOnlyPred = [i for i in IoUs if i < 1.0]
    print("average IoU: ", np.mean(IoUs))
    print("average IoU (Preds only):", np.mean(IoUsOnlyPred))
    print("min IoU: ", str(min(IoUsOnlyPred)))
    print("max IoU: ", str(max(IoUsOnlyPred)))

def compute_ap(dataset, model, inference_config):
    image_ids = dataset.image_ids #np.random.choice(dataset.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image_name = dataset.image_info[image_id]['id'] 
        #print(image_name)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config,image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
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
        print(image_name)
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config,image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    
    print("mAP Range: ", np.mean(APs))


def evalMaskViaCnn():
    new_dir = "../../MaskClassification/"
    
    # Root directory of the project
    ROOT_DIR = os.path.abspath("../../")
    from MaskClassification.scripts import label_image

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)

    file_name = "eval_image.png"
    model_file = new_dir + "tf_files/retrained_graph.pb"
    label_file = new_dir + "tf_files/retrained_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Mul"
    output_layer = "final_result"

    graph = label_image.load_graph(model_file)
    
    t = label_image.read_tensor_from_image_file(file_name,
                                                input_height=input_height,
                                                input_width=input_width,
                                                input_mean=input_mean,
                                                input_std=input_std)
    

    #t = label_image.read_tensor_from_image_var(masked_image,input_height=input_height,input_width=input_width,input_mean=input_mean,input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
        end = time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = label_image.load_labels(label_file)

        print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
        template = "{} (score={:0.5f})"
        
        for i in top_k:
            print(template.format(labels[i], results[i]))
        #return results[i], labels[i]
        return results[len(results)-1], labels[len(labels)-1]


def isImageAllWhite(image):
    try:
        orig_img = cv2.imread("eval_image.png", 0)
        black_img = cv2.bitwise_not(orig_img)
        if cv2.countNonZero(black_img) == 0:
            return True
        else:
            return False
    except:
        return False

def save_to_train(image, mask_image, contours, image_name, image_label):
    DATASET_ROOT = os.path.join(ROOT_DIR, "datasets/shuriken_gun")
    train_dir = os.path.join(DATASET_ROOT, "train")
    mask_dir = os.path.join(ROOT_DIR, "MaskClassification/tf_files/xray_photos/shuriken")

    train_img_exists = False

    image_label = "shuriken"

    if(isImageAllWhite(mask_image)):
        print("Mask image is all white, skipping save.")
        return

    for train_file in os.listdir(train_dir):
        if train_file == image_name:
            train_img_exists = True

    if train_img_exists == False:
        #partition contours
        contour = contours[0]
        x_values = contour[:,0]
        y_values = contour[:,1]

        width = image.shape[0]
        height = image.shape[1]

        x_values = np.array(x_values).tolist()
        y_values = np.array(y_values).tolist()

        #create annotation
        annotation = {image_name: {"filename": image_name, "size": 234, "regions": [{"shape_attributes": { "name": "polyline", "all_points_x": y_values, "all_points_y": x_values}, "region_attributes": {}}], "file_attributes": {"width": width, "height": height}, "label": image_label}}

        annotation_file = image_name.split('.')[0]
        annotation_dump = train_dir + "/" + annotation_file + ".json"

        with open(annotation_dump, 'w') as outfile:
            json.dump(annotation, outfile)

        #move image and annotation to training directory
        print("Moving " + image_name + " to training dir and mask directory")
        cv2.imwrite(train_dir + "/" + image_name, image)
        cv2.imwrite(mask_dir + "/" + image_name, mask_image)
    else:
        print("Image " + image_name + " already exists in training dir")


#This is used to manually create the mask files for the hand annotated images
def export_data_masks():
    mask_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/shuriken_gun/train/'
    mask_dir = "C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/MaskClassification/tf_files/xray_photos/shuriken/"

    for mask_file in os.listdir(mask_path):
        if mask_file.endswith('.json'):
            annotation = json.load(open(mask_path + mask_file))
            file_name = list(annotation.keys())[0]
            
            info = annotation[file_name]
            file_attr = info['file_attributes']
            polygons = info['regions'][0]['shape_attributes']
            #print(mask_file)
            #image = cv2.imread(mask_path + file_name, cv2.IMREAD_COLOR)
            
            #if you get errors, swap widht and height because some of the annottions are wrong
            mask = np.zeros([file_attr["width"], file_attr["height"]], dtype=np.uint8) + 255
            #mask = np.zeros((height, width), dtype=np.uint8) + 255
            rr, cc = skimage.draw.polygon(polygons['all_points_y'], polygons['all_points_x'])
            mask[rr, cc] = 0
            mask = cv2.blur(mask, (4,4))

            max_y = int(round(max(polygons['all_points_y'])))
            min_y = int(round(min(polygons['all_points_y'])))
            max_x = int(round(max(polygons['all_points_x'])))
            min_x = int(round(min(polygons['all_points_x'])))

            mask = mask[min_y:max_y, min_x:max_x]
            cv2.imwrite(mask_dir + file_name, mask)

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
        config = ShurikenConfig()
    else:
        class InferenceConfig(ShurikenConfig):
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
        detect_masks(model, config)
    elif args.command == "eval":
        eval_model(model, config, weights_path)
    elif args.command == "gen_mask":
        export_data_masks()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
