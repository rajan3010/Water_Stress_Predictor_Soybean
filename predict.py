# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from dataconfig import config
import glob
import csv

mean = np.array([123.68, 116.779, 103.939], dtype="float32")

def one_hot_encode(index):
    one_hot=np.zeros((len(config.CLASSES),),dtype='uint8')
    one_hot[index]=1
    return one_hot

def preprocess(image_path):
    
    image=cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    #image = image.astype("float32")
    #image -=mean
    return image
    
test_path=os.path.sep.join((config.BASE_PATH,config.TESTING))
filenames=sorted(glob.glob(os.path.sep.join((test_path,"*.jpg"))))
test_images=[preprocess(img) for img in filenames]

print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

pred_count=np.zeros((5,),dtype="uint8")
with open(config.PREDICTIONS_CSV_PATH, mode='w',newline='') as prediction_file:
    prediction_writer=csv.writer(prediction_file)
    for predimg in test_images:
        preds = model.predict(np.expand_dims(predimg, axis=0))[0]
        i = np.argmax(preds)
        one_hot_pred=one_hot_encode(i)
        pred_count[i]+=1
        prediction_writer.writerow(one_hot_pred)
    