from PIL import Image
import os
import tensorflow as tf
import np
from models.research.object_detection.utils import visualization_utils as vis_util
from models.research.object_detection.utils import label_map_util
from matplotlib import pyplot as plt

class BoneClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = 'training\\tuned_model\\frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)


    def get_classification(self, img):
        # Bounding Box Detection.
        try:
            with self.detection_graph.as_default():
                category_index = self.labelify()
                # Expand dimension since the model expects image to have shape [1, None, None, 3].
                img_expanded = np.expand_dims(img, axis=0)  
                (boxes, scores, classes, num) = self.sess.run(
                    [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                    feed_dict={self.image_tensor: img_expanded})
                
                vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                return img
        except Exception as e:
            print("Exception during classification", e)
        return


    def get_test_image_locations(self):
        test_image_path = 'test_images'
        images = os.listdir(test_image_path)
        image_paths = []
        for image in images:
            image = os.getcwd() + '\\test_images\\' + image
            image_paths.append(image)
        return image_paths


    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        desired_shape = (im_height, im_width, 3)
        np_array = np.array(image.getdata())
        try: 
            reshaped = np_array.reshape(desired_shape)
            reshaped_retyped = reshaped.astype(np.uint8)
            return reshaped_retyped
        except Exception as e:
            print('Exception loading image into numpy array', e)

    def labelify(self):
        NUM_CLASSES = 90
        label_path = os.path.join(os.getcwd(), 'training', 'data', 'object_detection.pbtxt')
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def open_images(self):
        IMAGE_SIZE = (12,8)
        image_paths = self.get_test_image_locations()
        for image_path in image_paths:
            img = Image.open(image_path)
            img_np = self.load_image_into_numpy_array(img)
            classification = self.get_classification(img_np)

            try:
                print('opening')
                plt_figure = plt.figure(figsize=IMAGE_SIZE)
                img_from_array = Image.fromarray(classification)
                img_from_array.show()
                
            except Exception as e:
                print('exception during plotting', e)





Bone = BoneClassifier()
Bone.open_images()


