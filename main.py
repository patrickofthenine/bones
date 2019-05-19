from imageai.Prediction import ImagePrediction
import os
import tensorflow as tf
import pathlib
import csv
import random
from PIL import Image
import datetime
AUTOTUNE = tf.data.experimental.AUTOTUNE

project_path 	= os.getcwd() 
training_model  = project_path + '/models/MURA-v1.1'
training_csv	= training_model + '/train_image_paths.csv'

#gets list of images
def get_all_images(path_to_images):
	print('...getting all images')
	counter=0
	list_of_images  = list()
	print('...opening training csv')
	with open(path_to_images, 'rt') as training_file: 
		print('...getting file paths')
		reader = csv.DictReader(training_file)
		for row in reader:
			if(counter<100):
				print('counter<100', counter)
				for image in row:
					if(row[image]):
						img_path = project_path + '/models/' + row[image]
						list_of_images.append(img_path)
						counter+=1

		random.shuffle(list_of_images)
	return list_of_images

def preprocess_image(image):
	print('preprocessing image', image)
	img = tf.image.decode_png(image)
	img_final = tf.image.resize(img, [192, 192])		
	#normalize 
	img_final /= 255.0
	print('returning processed image')
	return img_final

def create_tensors(images):
	print('...creating tensors')
	print('Total number of images: ', len(images))
	#create tensors
	counter = 0
	for image in images:
		if(counter <= 10):
			print(counter)
			try:
				preprocess_image(image)
			except:
				print('exception', image)	
			counter+=1
	print('Successfully Processed: ', counter)
	return images

def create_labels(count):
	print('...creating labels')
	labels = list()
	i = 0
	while(i<=count):
		labels.append('bones')
		i+=1
	print(len(labels), ' labels created')
	return labels

def build_tf_dataset(images):
	print('...building tf dataset', len(images))
	
	#slice array of paths into dataset of paths
	path_dataset = tf.data.Dataset.from_tensor_slices(images)

	#preprocess all images
	image_dataset = path_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
	print('image dataset created')
	
	#need a label for each item
	labels = list()
	for i in range(len(images)):
		labels.append('bone_human')
	label_dataset = tf.data.Dataset.from_tensor_slices(labels)
	print('label dataset created')

	print('zipping datasets')
	data = (image_dataset, label_dataset)
	dataset = tf.data.Dataset.zip(data) 
	
	print('...returning dataset')
	return dataset

def run_training(training_csv):
	print('Starting: ', datetime.datetime.now())
	start = datetime.datetime.now()
	
	#process the images
	images = get_all_images(training_csv)
	#create a dataset from processed images
	dataset = build_tf_dataset(images)
	#
	end   = datetime.datetime.now()
	print('Total Run Time', end - start)
	return 

run_training(training_csv)

