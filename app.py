import numpy as np
from flask import Flask, render_template, send_file,flash, request,redirect
from flask_cors import CORS
import requests
import os
import urllib.request
from infer_util import init, img_inference
import matplotlib.pyplot as plt
import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import time
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import pandas

ALLOWED_EXTENSIONS = set(['bmp', 'png', 'jpg', 'jpeg', 'gif'])

global model
global session
model, session=init()
model._make_predict_function()
print('\n\n     MODEL INITIALIZED\n\n')
app=Flask(__name__)
CORS(app)
app.secret_key = "secret key"
# set the modified tf session as backend in keras

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = file.filename
			print(filename)
			filepath=os.path.join('./files', filename)
			file.save(filepath)
			print(filepath)
			# We want to execute the model here

			output, imgout=img_inference(model,session,filepath,thresh=.8)
			#---------------------------------------------
			print('---------------- Inference initiated -----------------------\n\n')
			
			cv2.imwrite('output.png',imgout)
			print(output)
			flash(f'Bounding box: {output}')
			return redirect('/')
		else:
			flash('Allowed file types are bmp, png, jpg, jpeg, gif')
			return redirect(request.url)


@app.route('/submit',methods=['POST'])
def submit():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = file.filename
			filepath=os.path.join('./files', filename)
			file.save(filepath)
			# We want to execute the model here
			output, imgout=img_inference(model,session,filepath,thresh=.8)			
			cv2.imwrite('output.jpg',imgout)
			print(output)
			flash(f'Bounding box: {output}')
			return send_file('output.jpg')
		else:
			flash('Allowed file types are bmp, png, jpg, jpeg, gif')
			return redirect(request.url)

if __name__=='__main__':
    print('Server starting')
    app.run(host='0.0.0.0',port=5000)