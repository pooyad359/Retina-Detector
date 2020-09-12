import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import pandas
import pytesseract
import re

global model
global labels_to_names
global session

THRES_SCORE = 0.5

def get_session():
#     config = tf.ConfigProto()
    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def img_inference(model,session,img_path,thresh=0.8):
    image = read_image_bgr(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    draw=image.copy()
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    print(f'Image resized to :{image.shape}\n')
    # process image
    start = time.time()
    with session.as_default():
        with session.graph.as_default():
            boxes, scores, labels = model.predict_on_batch(image[None,...])
    print("processing time: ", time.time() - start)
    
    # correct for image scale
    boxes /= scale
    try:
        output=boxes[0][0]
        xmin=np.int(output[0])
        ymin=np.int(output[1])
        xmax=np.int(output[2])
        ymax=np.int(output[3])
        imgout=draw[ymin:ymax,xmin:xmax,:]
        imgout=cv2.cvtColor(imgout,cv2.COLOR_RGB2GRAY)
        # imgout = cv2.medianBlur(imgout,5)
        # imgout = cv2.adaptiveThreshold(imgout,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,3)
    except:
        output=None
        imgout=draw
        imgout=cv2.cvtColor(imgout,cv2.COLOR_RGB2GRAY)
    imgout=auto_rotate(imgout)
    return output, imgout

def auto_rotate(img):
    try:
        osd=pytesseract.image_to_osd(img)
        deg=np.int(re.search('(?<=Rotate: )\d+', osd).group(0))
    except:
        deg=0
    
    if deg==90:
        imgout=cv2.transpose(img)
        imgout=cv2.flip(imgout,1)
    elif deg==270:
        imgout=cv2.transpose(img)
        imgout=cv2.flip(imgout,0)
    elif deg==180:
        imgout=cv2.flip(img, -1);
    else:
        imgout=img
    return imgout

def init():
    session=get_session()
    keras.backend.tensorflow_backend.set_session(session)
    CLASSES_FILE='./files/classes.csv'
    model_path='./files/R50_20_02_04_15_40.h5'

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)

    # load label to names mapping for visualization purposes
    labels_to_names = pandas.read_csv(CLASSES_FILE,header=None).T.loc[0].to_dict()
    return model, session

    