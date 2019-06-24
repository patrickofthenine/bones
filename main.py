from imageai.Prediction import ImagePrediction
import os
import tensorflow as tf
import ssl
import pathlib
import csv
import random
from PIL import Image
import datetime
import pprint
import xml 
import glob
import pandas as pd
import xml.etree.ElementTree as ET

pp = pprint.PrettyPrinter(indent=4)
BATCH_SIZE = 32
DEV_IMG_COUNT = 10
ssl._create_default_https_context = ssl._create_unverified_context
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

#gets list of images
def get_all_images(path_to_images):
	counter=0
	list_of_images  = list()
	with open(path_to_images, 'rt') as training_file: 
		reader = csv.DictReader(training_file)
		for row in reader:
			#if(counter<DEV_IMG_COUNT):
			for image in row:
				if(row[image]):
					img_path = project_path + '/models/' + row[image]
					list_of_images.append(img_path)
					counter+=1
	return list_of_images

def preprocess_image(image):
	image_raw = tf.io.read_file(image)
	img_tensor = tf.image.decode_png(image_raw, channels=3)
	img_final = tf.image.resize(img_tensor, [224, 224])		
	#normalize 
	img_final /= 255.0
	#print(img_final.shape, img_final.numpy().min(), img_final.numpy().max())
	return img_final

def create_tensors(images):
	print('Total number of image sets: ', len(images))
	processed = list()
	#create tensors
	counter = 0

	for category, image_set in images.items():
		for image in image_set:
			try:
				processed.append(preprocess_image(image))
			except Exception as e:
				print(e)	
	print('Successfully Processed: ', len(processed), 'images')
	return processed

def build_tensorflow_dataset(images):
	print('...building tensorflow dataset from image sets', images.keys())
		
	#slice array of paths into dataset of paths
	tensors = create_tensors(images)
	image_dataset = tf.data.Dataset.from_tensor_slices(tensors)

	#need a label for each item
	labels = list()
	for i in range(len(tensors)):
		labels.append('bone_human')

	#label_dataset = tf.data.Dataset.from_tensor_slices(labels)
	#data = (image_dataset, label_dataset)
	dataset = tf.data.Dataset.zip(image_dataset) 
	return dataset

def create_labels(count):
	print('...creating labels')
	labels = list()
	i = 0
	while(i<=count):
		labels.append('bones')
		i+=1
	print(len(labels), ' labels created')
	return labels
	return dataset

def get_keras_dataset(dataset, buffer_size):
	for d in dataset:
		change_range(d)
	data = dataset.shuffle(buffer_size=buffer_size) 
	data = data.repeat()
	data = data.batch(BATCH_SIZE)
	data = data.prefetch(buffer_size=buffer_size)
	print(data)
	return data

def change_range(image):
	in_range_image = 2*image-1
	return in_range_image

def run_training_v1():
	start = datetime.datetime.now()
	print('Starting: ', start)
	images = create_image_hash()
	dataset = build_tensorflow_dataset(images)
	keras_dataset = get_keras_dataset(dataset, len(images))
	mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, alpha=1.0)
	mobile_net.trainable = False
	data_to_model = next(iter(keras_dataset))
	mobile_net(data_to_model)
	
	model = tf.keras.Sequential([
		mobile_net,
		tf.keras.layers.GlobalAveragePooling2D(),
		tf.keras.layers.Dense(len(images))
	])
	
	model.compile(optimizer=tf.train.AdamOptimizer(), 
		loss=tf.keras.losses.sparse_categorical_crossentropy,
		metrics=["accuracy"])

	model.summary()

	end   = datetime.datetime.now()
	print ('Ending: ', end)
	print('Total Run Time', end - start)
	return 

def create_image_hash():
	dirs = {}
	image_path = os.getcwd() + '/training_images/'
	for path, directories, files in os.walk(image_path, topdown=True):
		name = os.path.basename(os.path.normpath(path))
		for i, file in enumerate(files):
			files[i] = os.path.join(path, file)
		dirs[name] = files
	return dirs

#images = create_image_hash()
#pp.pprint(images)
#run_training()

#https://medium.com/coinmonks/tensorflow-object-detection-with-custom-objects-34a2710c6de5
def run_training():
	csv_cols = ['bone', 'other column']
	images = create_image_hash()
	outfile = os.path.join(os.getcwd(), 'output', 'images.csv')
	with open(outfile, 'w', newline='') as out:
		writer = csv.writer(out)
		for bone, paths in images.items():
			for path in paths:
				row = [bone, path]
				writer.writerow(row)
	print('done')
	return 

#run_training()
#from https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    image_xml_path = os.path.join(os.getcwd(), 'labels_output')
    xml_df = xml_to_csv(image_xml_path)
    xml_df.to_csv('skeletal_labels.csv', index=None)
    print('Successfully converted xml to csv.')

main()