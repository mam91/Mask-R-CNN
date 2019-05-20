import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import shutil
import cv2

def strip_masks(directory):
	mask_path = 'C:/Users/mmill/Downloads/Mask_RCNN-master/datasets/shuriken_contours_plus/' + directory + '/'

	annotations = json.load(open(mask_path + 'via_region_data.json'))

	for mask in annotations:
		#print(annotations[mask])
		image_file = annotations[mask]['filename']
		mask_file = image_file[:-4] + '.json'

		mask_exists = False

		for data_file in os.listdir(mask_path):
			if mask_file == data_file:
				mask_exists = True
				break

		if not mask_exists:
			image_path = os.path.join(mask_path, image_file)
			image = skimage.io.imread(image_path)
			height, width = image.shape[:2]

			annotations[mask]['file_attributes']['height'] = height
			annotations[mask]['file_attributes']['width'] = width

			annotation = { image_file : annotations[mask] }
			with open(mask_path + mask_file, 'w') as outfile:  
				json.dump(annotation, outfile)

def delete_key(annotation, key):
	try:
		del annotation[key]
	except KeyError:
		pass

def add_labels(directory, label):
	mask_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/' + directory + '/'
	print(mask_path)
	for mask_file in os.listdir(mask_path):
		if mask_file.endswith('.json'):
			#load and add label
			annotation = json.load(open(mask_path + mask_file))

			file_name = list(annotation.keys())[0]
			print(mask_file)
			if('label' not in annotation[file_name]):
				print('No label found for: ' + mask_file)
				print('adding label ' + label)
				annotation[file_name]['label'] = label
				with open(mask_path + mask_file, 'w') as outfile:  
					json.dump(annotation, outfile)
			# else:
			# 	print('Deleting label key')
			# 	delete_key(annotation[file_name], 'label')
			# 	with open(mask_path + mask_file, 'w') as outfile:  
			# 		json.dump(annotation, outfile)

def doesFileExist(file_path):
    return os.path.isfile(file_path)

def split_dataset(dataset, percent_split):
	#dataset_path = 'C:/Users/mmill/Downloads/Mask_RCNN-master/datasets/' + dataset
	#for mask_file in os.listdir(mask_path):
	return True

def swap_height_and_width():
    mask_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/shuriken_gun/staging/'

    for mask_file in os.listdir(mask_path):
        if mask_file.endswith('.json'):
            annotation = json.load(open(mask_path + mask_file))
            file_name = list(annotation.keys())[0]
            info = annotation[file_name]
            polygons = info['regions'][0]['shape_attributes']
            all_points_y = polygons['all_points_x']
            all_points_x = polygons['all_points_y']
            polygons['all_points_x'] = all_points_y
            polygons['all_points_y'] = all_points_x
            annotation[file_name]['regions'][0]['shape_attributes'] = polygons
            with open(mask_path + mask_file, 'w') as outfile:  
                json.dump(annotation, outfile)
		
def move_training_data(root_dataset, subset_from, subset_to, move_percent):
	from_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/' + root_dataset + '/' + subset_from + '/'
	to_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/' + root_dataset + '/' + subset_to + '/'

	file_list = []
	for mask_file in os.listdir(from_path):
		if mask_file.endswith('.json'):
			file_list.append(mask_file)

	print("Total file count: " + str(len(file_list)))

	num_to_choose = int(round(len(file_list) * move_percent))

	print("Number to choose: " + str(num_to_choose))

	trimmed_file_list = np.random.choice(file_list, num_to_choose, replace=False)

	print("Trimmed file count: " + str(len(trimmed_file_list)))

	for mask_file in trimmed_file_list:
		annotation = json.load(open(from_path + mask_file))
		file_name = list(annotation.keys())[0]

		from_mask_path = from_path + mask_file
		from_img_path = from_path + file_name
		to_mask_path = to_path + mask_file
		to_img_path = to_path + file_name

		if not doesFileExist(from_img_path):
			print(file_name + " does not exist!")
			continue

		shutil.copyfile(from_mask_path, to_mask_path)
		shutil.copyfile(from_img_path, to_img_path)
	print("Copy complete")

def export_masks():
    mask_path = 'C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/datasets/shuriken_gun/train/'
    to_path = "C:/Users/mmill/Documents/GitHub/Education/MaskRCnn/MaskClassification/tf_files/xray_photos/shuriken/"

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
            mask = np.zeros([file_attr["height"], file_attr["width"]], dtype=np.uint8) + 255
            #mask = np.zeros((height, width), dtype=np.uint8) + 255
            rr, cc = skimage.draw.polygon(polygons['all_points_y'], polygons['all_points_x'])
            mask[rr, cc] = 0
            mask = cv2.blur(mask, (4,4))
                
            #print(polygons['all_points_y'])
                
            max_y = int(round(max(polygons['all_points_y'])))
            min_y = int(round(min(polygons['all_points_y'])))
            max_x = int(round(max(polygons['all_points_x'])))
            min_x = int(round(min(polygons['all_points_x'])))
                #print(max_y)
                
            mask = mask[min_y:max_y, min_x:max_x]
                
            cv2.imwrite(to_path + file_name, mask)
            #break
add_labels('shuriken_gun/train', 'gun')
#move_training_data('shuriken_gun', 'staging', 'train', 0.50)
#swap_height_and_width()
#export_masks()