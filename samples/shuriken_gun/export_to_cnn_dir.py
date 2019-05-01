import os
import sys
import json
import skimage
import cv2
import numpy as np

#This is used to manually create the mask files for the hand annotated images
def export_data_masks(dataset, directory, cnn_dir):
    mask_path = '../../datasets/' + dataset + '/' + directory + '/'

    for mask_file in os.listdir(mask_path):
        if mask_file.endswith('.json'):
            annotation = json.load(open(mask_path + mask_file))
            file_name = list(annotation.keys())[0]
            
            info = annotation[file_name]
            file_attr = info['file_attributes']
            polygons = info['regions'][0]['shape_attributes']
            print(mask_file)
            #image = cv2.imread(mask_path + file_name, cv2.IMREAD_COLOR)
            
            #if you get errors, swap widht and height because some of the annottions are wrong
            #mask = np.zeros([file_attr["width"], file_attr["height"]], dtype=np.uint8) + 255
            mask = np.zeros([file_attr["height"], file_attr["width"]], dtype=np.uint8) + 255
			#mask = np.zeros((height, width), dtype=np.uint8) + 255
            rr, cc = skimage.draw.polygon(polygons['all_points_y'], polygons['all_points_x'])
            mask[rr, cc] = 0
            mask = cv2.blur(mask, (4,4))

            max_y = int(round(max(polygons['all_points_y'])))
            min_y = int(round(min(polygons['all_points_y'])))
            max_x = int(round(max(polygons['all_points_x'])))
            min_x = int(round(min(polygons['all_points_x'])))

            mask = mask[min_y:max_y, min_x:max_x]
            cv2.imwrite('../../MaskClassification/tf_files/xray_photos/' + cnn_dir + '/' + file_name, mask)

export_data_masks('shuriken_gun', 'train', 'shuriken')