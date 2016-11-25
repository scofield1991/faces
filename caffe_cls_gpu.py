import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from StringIO import StringIO
import cPickle as pickle
import collections
import skimage
import timeit
import sys
sys.path.append('/home/oleksandr/xgboost/xgboost/python-package/')
import xgboost as xgb
sys.path.append('/home/oleksandr/FaceRocognition')
from Util import smart_dummy, ensemble_classifier, gini, ROC, plot_feature_importance, print_res_table
from numba import jit
from scipy.spatial.distance import braycurtis
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/oleksandr/Caffe_GPU/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2


class CaffeVGGNet:

	def __init__(self, mode=0):
		
		#create net
		model_def = '/home/oleksandr/FaceRocognition/vgg_face_caffe/VGG_FACE_deploy.prototxt'
		model_weights = '/home/oleksandr/FaceRocognition/vgg_face_caffe/VGG_FACE.caffemodel'

		self.net = caffe.Net(model_def,      # defines the structure of the model
                	model_weights,  # contains the trained weights
                	caffe.TEST)     # use test mode (e.g., don't perform dropout)

		#get mean image
		mean_array = np.array([129.1863,104.7624,93.5940])

		# create transformer for the input called 'data'
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

		self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
		self.transformer.set_mean('data', mean_array)            # subtract the dataset-mean value in each channel
		self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
		self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

		# set the size of the input (we can skip this if we're happy
		#  with the default; we can also change it later, e.g., for different batch sizes)
		self.net.blobs['data'].reshape(1,        # batch size
                          		  3,         # 3-channel (BGR) images
                          	      224, 224)  # image size is 224x224

		mask_file_path = '/home/oleksandr/FaceRocognition/mask402.dump'

		#load mask
		with open(mask_file_path,'rb') as mask_file:  
			self.mask = pickle.load(mask_file)
		#mask_size = collections.Counter(self.mask)[True]

		#load xgboost model
		with open('/home/oleksandr/FaceDetection/Faces_vgg1_10_11_2016.dump','r') as model:
   			self.xgboost_model = pickle.load(model)



	def classify(self, img):

		#classify image
		# transform it and copy it into the net
		img_arr = np.asarray(Image.open(img))
		#image = caffe.io.load_image(img_path)
		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img_arr)

		# perform classification
		output = self.net.forward()

		# obtain the output probabilities
		output_prob = output['prob'][0]
			
		print output_prob

	def get_features(self, img):
		caffe.set_mode_gpu()

		image_64 = img.decode('base64')
		image_str = StringIO(image_64)
		im_arr = np.asarray(Image.open(image_str))
		img_float = skimage.img_as_float(im_arr).astype(np.float32)

		self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img_float)

		output = self.net.forward()


		#features_array = self.net.blobs['fc6'].data[self.mask].copy()

		features_array = self.net.blobs['pool5'].data.copy()
		features_array = features_array.reshape(25088)   #[self.mask]

		return features_array.astype(int).tolist()

	def make_prediction(self, img1, img2):
		image1_features = self.get_features(img1)
		image2_features = self.get_features(img2['base64'])

		br = braycurtis(image1_features, image2_features)

		#concatenate features of two images and reshape to two dimensional matrix
		features = np.hstack((image1_features, image2_features))
		br_features = np.hstack((features, br)).reshape(1, 805)
		proba = self.xgboost_model.predict_proba(br_features).tolist()[0]

		return {'name': img2['name'], 'different':proba[0], 'same':proba[1]}

		#return features.shape



