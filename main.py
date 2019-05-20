from imageai.Prediction import ImagePrediction
import os
import tensorflow as tf
import ssl
import pathlib
import csv
import random
from PIL import Image
import datetime
BATCH_SIZE = 32
DEV_IMG_COUNT = 3000
ssl._create_default_https_context = ssl._create_unverified_context
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

project_path 	= os.getcwd() 
training_model  = project_path + '/models/MURA-v1.1'
training_csv	= training_model + '/train_image_paths.csv'

#gets list of images
def get_all_images(path_to_images):
	print('...getting all images')
	counter=0
	list_of_images  = list()
	print('...opening training file csv')
	with open(path_to_images, 'rt') as training_file: 
		reader = csv.DictReader(training_file)
		for row in reader:
			if(counter<DEV_IMG_COUNT):
				for image in row:
					if(row[image]):
						img_path = project_path + '/models/' + row[image]
						list_of_images.append(img_path)
						counter+=1
	return list_of_images

def preprocess_image(image):
	print('...preprocessing')
	img = tf.image.decode_png(image)
	img_final = tf.image.resize(img, [192, 192])		
	#normalize 
	img_final /= 255.0
	print(img_final)
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

def build_tensorflow_dataset(images):
	print('...building tensorflow dataset from', len(images), 'images')
		
	#slice array of paths into dataset of paths
	path_dataset = tf.data.Dataset.from_tensor_slices(images)

	#preprocess all images
	image_dataset = path_dataset.map(preprocess_image, num_parallel_calls=BATCH_SIZE)
	print('...images created', image_dataset)
	
	#need a label for each item
	labels = list()
	for i in range(len(images)):
		labels.append('bone_human')
	label_dataset = tf.data.Dataset.from_tensor_slices(labels)
	print('...labels created', label_dataset)

	print('...creating dataset')
	data = (image_dataset, label_dataset)
	dataset = tf.data.Dataset.zip(data) 
	print('...returning dataset', dataset)
	return dataset

def get_keras_dataset(dataset, buffer_size):
	print('...getting keras dataset', dataset)
	data = dataset.shuffle(buffer_size=buffer_size) 
	data = data.repeat()
	data = data.batch(BATCH_SIZE)
	data = data.prefetch(buffer_size=buffer_size)
	keras_dataset = data.map(change_range) 
	return keras_dataset

def change_range(image,label):
	return 2*image-1, label

def run_training(training_csv):
	start = datetime.datetime.now()
	print('Starting: ', start)
	
	#get image paths from model csv
	images = get_all_images(training_csv)
	#create a dataset from processed images
	dataset = build_tensorflow_dataset(images)
	#get a keras dataset
	keras_dataset = get_keras_dataset(dataset, len(images))

	mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
	mobile_net.trainable = False
	image_batch, label_batch = next(iter(keras_dataset))
	feature_map_batch = mobile_net(image_batch)
	print(feature_map_batch.shape)
	#print(feature_map_batch.shape)

	model = tf.keras.Sequential()
	model.add(feature_map_batch)
	end   = datetime.datetime.now()
	print ('Ending: ', end)
	print('Total Run Time', end - start)
	return 

run_training(training_csv)
