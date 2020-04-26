import numpy as np
import pandas as pd
from dataconfig import config
import os
from sklearn.preprocessing import LabelEncoder
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf

train_pdf_path=os.path.sep.join([config.ORIG_DATASET, "TrainAnnotations.csv"])
#print(train_pdf)
data=pd.read_csv(train_pdf_path)

images =data['file_name']
labels =data['annotation']

labels=[str(index) for index in labels]
 #the data for training and the remaining 25% for testing
(training_images, validation_images, training_labels, validation_labels) = train_test_split(images, labels,test_size=0.25, random_state=42)
for split in (config.TRAINING,config.VALIDATION):
    print("[INFO] processing '{} split' ...".format(split))
    for label,image in zip(eval(split+'_labels'),eval(split+'_images')):
        image_path=os.path.sep.join([config.ORIG_DATASET, image])
        dir_path=os.path.sep.join([config.BASE_PATH,split,label])

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        p=os.path.sep.join([dir_path,image])
        shutil.copy2(image_path,p)
    #filename=os.path.sep.join([config.ORIG_DATASET,train_images[index]])