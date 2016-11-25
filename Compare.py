import cPickle as pickle
import numpy as np
import sys
import os 
import ast
import glob
import pandas as pd
from scipy.spatial.distance import braycurtis
sys.path.append('/home/oleksandr/xgboost/xgboost/python-package/')
import xgboost as xgb
sys.path.append('/home/oleksandr/FaceRocognition')
from Util import smart_dummy, ensemble_classifier, gini, ROC, plot_feature_importance, print_res_table


#load xgboost model
with open('/home/oleksandr/FaceDetection/Faces_vgg1_10_11_2016.dump','r') as model:
	xgboost_model = pickle.load(model)
	xgboost_model.clfs = xgboost_model.clfs[:10]

def load_file_names(path):
	files = []
	for filename in glob.iglob(path + '*'):
		files.append(filename)
	return files


def load_db_features(files):
	clients_array = []
	for file_name in files:
		print file_name
		text_file = open(file_name, "r")
		features_list = ast.literal_eval(text_file.read())
		features_arr = np.asarray(features_list)
		if type(clients_array) == np.ndarray:
			clients_array = np.vstack((clients_array, features_arr))
		else:
			clients_array = features_arr
    #answers.append(Compare.compare_features(main_features, add_features, key))

	return clients_array


def compare_features(feature_array_1, feature_array_2, key):

		br = braycurtis(feature_array_1, feature_array_2)
		#concatenate features of two images and reshape to two dimensional matrix
		features = np.hstack((feature_array_1, feature_array_2))
		br_features = np.hstack((features, br))
		proba = xgboost_model.predict_proba(br_features).tolist()[0]

		return {'name': key, 'different':proba[0], 'same':proba[1]}


def compare_features_batch(features):
		#concatenate features of two images and reshape to two dimensional matrix
		proba = xgboost_model.predict_proba(features).tolist()

		return proba

def braycurtis(arr_1, arr_2):
    return np.abs(arr_1 - arr_2, dtype=np.float64).sum(axis=1) / np.abs(arr_1 + arr_2).sum(axis=1)

def broadcast_array(arr, size):
	return np.tile(arr, (size_arr,1))


class FindPerson:

	def __init__(self, path_to_clients_features):

		file_names = load_file_names(path_to_clients_features)
		ids = [file_name[file_name.rfind('/')+1:] for file_name in file_names]

		self.ids = pd.DataFrame(ids)
		self.clients_features = load_db_features(file_names)

	def build_features(self, net, photo):
		size_arr = self.clients_features.shape[0]
		self.photo_features = net.get_features(photo)
		self.main_arr_broadcasted = broadcast_array(self.photo_features, size_arr)
		cls_arr = np.hstack((self.clients_features, self.main_arr_broadcasted))

		braycurtis_dist = braycurtis(self.clients_features, self.main_arr_broadcasted).reshape(1, size_arr)
		self.cls_arr_br = np.concatenate((cls_arr, braycurtis_dist.T),axis=1)
		
		
	def find_person_xgb(self):
		answers = compare_features_batch(self.cls_arr_br)
		df_answers = pd.DataFrame(answers)
		df_answers[0] = self.ids
		df_answers_sort = df_answers.sort(1, ascending=False)

		return df_answers_sort[:5].set_index(0).to_dict()[1]

	def find_person_distanse(self, first_step=False):
		br_distances = braycurtis(self.clients_features, self.main_arr_broadcasted)
		br_mask = br_distances < 0.5
		br_least_distances = br_distances[br_mask]
		ids = self.ids[0].where(br_mask).dropna()
		df_answers = pd.DataFrame(br_least_distances, index=ids)
		first_step_features = self.clients_features[br_mask]
		if first_step:
			return (df_answers, first_step_features) 

		df_answers_sort = df_answers.sort(0)

		return df_answers_sort[:5].to_dict()[0]

	def find_person_two_steps(self):
		first_step_answers, firs_step_features = self.find_person_distanse(first_step=True)
		size_arr = firs_step_features.shape[0]
		main_photo_features = broadcast_array(self.photo_features, size_arr)
		second_step_features = np.hstack((firs_step_features, main_photo_features))
		cls_arr_br = np.concatenate((second_step_features, first_step_answers.values),axis=1)
		first_step_answers[0] = compare_features_batch(cls_arr_br)

		return first_step_answers[:5].to_dict()[0]



	def make_prediction(self, img1, img2):
		image1_features = self.get_features(img1)
		image2_features = self.get_features(img2['base64'])

		br = braycurtis(image1_features, image2_features)

		#concatenate features of two images and reshape to two dimensional matrix
		features = np.hstack((image1_features, image2_features))
		br_features = np.hstack((features, br)).reshape(1, 805)
		proba = self.xgboost_model.predict_proba(br_features).tolist()[0]

		return {'name': img2['name'], 'different':proba[0], 'same':proba[1]}