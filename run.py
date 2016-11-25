#!/home/oleksandr/anaconda2/envs/flask/lib/python2.7
from app import app
from flask import Flask, request, jsonify
from caffe_cls_gpu import CaffeVGGNet
import timeit
import redis
import ast
import Compare
import numpy as np


app = Flask(__name__)


from app import views





@app.route('/get_photos', methods=['POST', 'GET'])
def index():
	if request.method == 'POST':
		try:
			#dopPhotos = request.json['dopPhotos']
			#mainPhoto = request.json['mainPhoto']
			
			dopPhotos = request.get_json()['dopPhotos']
			mainPhoto = request.get_json()['mainPhoto']

			predictions = []
			for photo in dopPhotos:
				predictions.append(
					vgg_net.make_prediction(mainPhoto, photo)
					)
			return jsonify(predictions)
			#return jsonify({'main': mainPhoto_features, 'dop': doPhoto_features})
		except KeyError:
			return jsonify({'error': 'KeyError'})


@app.route('/get_features', methods=['POST', 'GET'])
def get_features():
	if request.method == 'POST':
		try:
			#dopPhotos = request.get_json()['dopPhotos']
			mainPhoto = request.get_json()['mainPhoto']


			#strat_timer = timeit.default_timer()
			mainPhoto_features = vgg_net.get_features(mainPhoto)
			#elapsed = timeit.default_timer() - strat_timer
			#print ('time to make features 1st photo: ', elapsed)

			#pred = vgg_net.make_prediction(mainPhoto, dopPhotos[0]['base64'])
			#print predictions

			#return jsonify({'main': mainPhoto_features, 'dop': doPhoto_features})
			return jsonify({'main': mainPhoto_features})
		except KeyError:
			return jsonify({'error': 'KeyError'})


@app.route('/find_person_startapp', methods=['POST', 'GET'])
def find_person_startapp():
	if request.method == 'POST':
		try:
			mainPhoto = request.get_json()['mainPhoto']

			find_persons.build_features(vgg_net, mainPhoto)
			answer = find_persons.find_person()

			#return jsonify({'main': mainPhoto_features, 'dop': doPhoto_features})
			return jsonify({'main': answer})
		except KeyError:
			return jsonify({'error': 'KeyError'})



@app.route('/get_keys', methods=['POST', 'GET'])
def redis_keys():
	if request.method == 'POST':
		try:
			add_key = request.get_json()['listKey']
			main_key = request.get_json()['mainKey']
			print main_key
			classify_array = []
			answers = []
			
			redis_conn = redis.StrictRedis(host='10.62.130.236', port=6379)
			
			main_features = np.array(ast.literal_eval(redis_conn.get(main_key)))
			additional_keys = redis_conn.get(add_key)
			#print additional_keys
			list_keys = additional_keys.split(',')
			#print list_keys

			strat_timer = timeit.default_timer()
			for key in list_keys:
				add_features = np.array(ast.literal_eval(redis_conn.get(key)))
				all_features = np.hstack((main_features, add_features))
				if type(classify_array) == np.ndarray:
					classify_array = np.vstack((classify_array, all_features))
				else:
					classify_array = all_features
				#answers.append(Compare.compare_features(main_features, add_features, key))
			print classify_array.shape
			elapsed = timeit.default_timer() - strat_timer
			print ('time to get 1000 photos: ', elapsed)
			#print answers
			strat_timer = timeit.default_timer()
			answ = Compare.compare_features_batch(classify_array)

			elapsed = timeit.default_timer() - strat_timer
			print ('time to compare features 1000 photo: ', elapsed)
			

			return jsonify(answ)
			return jsonify(0)
		except KeyError:
			return jsonify({'error': 'KeyError'})


if __name__ == '__main__':
	vgg_net = CaffeVGGNet()
#	find_persons = Compare.FindPerson('/home/oleksandr/FaceDetection/flaskapp/clients/')
	app.run(host='0.0.0.0', port=8841, debug = False)
	
