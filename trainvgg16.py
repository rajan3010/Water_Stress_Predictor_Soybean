import matplotlib
matplotlib.use('Agg')

from dataconfig import config
from dataconfig import util
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import pickle
import os
from sklearn.utils import class_weight

'''train_pdf_path=os.path.sep.join([config.ORIG_DATASET, "TrainAnnotations.csv"])
#print(train_pdf)
data=pd.read_csv(train_pdf_path)

images =data['file_name']
labels =data['annotation']

labels=[str(index) for index in labels]
 #the data for training and the remaining 25% for testing
(training_images, validation_images, training_labels, validation_labels) = train_test_split(images, labels,test_size=0.20, random_state=42)
for split in (config.TRAINING,config.VALIDATION):
    print("[INFO] processing '{} split' ...".format(split))
    for label,image in zip(eval(split+'_labels'),eval(split+'_images')):
        image_path=os.path.sep.join([config.ORIG_DATASET, image])
        dir_path=os.path.sep.join([config.BASE_PATH,split,label])

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        p=os.path.sep.join([dir_path,image])
        shutil.copy2(image_path,p)'''
        
    
#Construct a smaller model with lesser number of layers
baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
baseModel.summary()
#exit()
'''small_model = Sequential()
for layer in baseModel.layers[:-4]: # go through until last layer
    small_model.add(layer)'''
#small_model.summary()

# construct the head of the model that will be placed on top of the
# the base model
#headModel = small_model.output
headModel= baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
#headModel=BatchNormalization()(headModel)
headModel = Dropout(0.25)(headModel)
#headModel = Dense(512, activation="relu")(headModel)
#headModel = Dropout(0.25)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
#model.summary()
#exit()
def plot_training(H, N, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

#set the training paths and validation paths for referencing in the model
trainPath=os.path.sep.join([config.BASE_PATH, config.TRAINING])
validationPath=os.path.sep.join([config.BASE_PATH, config.VALIDATION])
testingPath=os.path.sep.join([config.BASE_PATH, config.TESTING])

training_len=util.get_no_of_images_dl(trainPath)
validation_len=util.get_no_of_images_dl(validationPath)
testing_len=util.get_no_of_images_dl(testingPath)
#initialize training data augmentation
trainAug= ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"

)

#Initialize validation data augmentation
valAug= ImageDataGenerator()

#Initialize test data augmentation
testAug= ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
#trainAug.mean = mean
#valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=config.BATCH_SIZE)
# initialize the validation generator
valGen = valAug.flow_from_directory(
    validationPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE)

testGen = testAug.flow_from_directory(
	testingPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

class_weights=class_weight.compute_class_weight('balanced', np.unique(trainGen.classes), trainGen.classes)
class_weights=dict(enumerate(class_weights))
#class_weights[2]=2.7
#class_weights[3]=2.5
#class_weights[0]=1.54
# initialize the testing generator
'''testGen = valAug.flow_from_directory(
    testingPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=config.BATCH_SIZE)'''

# load the VGG16 network, ensuring the head FC layer sets are left off

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
loss_weights=[1.0, 1.526, 4.239, 3.88, 2.58]
for layer in baseModel.layers:
    layer.trainable = False
#training the model after freezing the base model of VGG16
print("[INFO] compiling model for warmup")
opt=Adam(lr=1e-4)
model.compile(loss='binary_crossentropy',optimizer=opt, metrics=["accuracy"])

#train the head of the transfer learning model -[VGG16]
print("[INFO] training head... ")
H=model.fit_generator(
    trainGen,
    steps_per_epoch=training_len//config.BATCH_SIZE,
    validation_data=valGen,
    class_weight=class_weights,
    validation_steps=validation_len//config.BATCH_SIZE,
    epochs=40
)
plot_training(H, 40, config.WARMUP_PLOT_PATH)

trainGen.reset()
valGen.reset()


for layer in baseModel.layers[15:]:
    layer.trainable=True

#display which layers are trainable
for layer in baseModel.layers:
    print("{}.{}".format(layer,layer.trainable))

print("[INFO] re-compiling model....")
opt=SGD(lr=1e-4, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H=model.fit_generator(
    trainGen,
    steps_per_epoch=training_len//config.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=validation_len//config.BATCH_SIZE,
    class_weight=class_weights,
    epochs=15
)

plot_training(H,15,config.UNFROZEN_PLOT_PATH)

print("[INFO] evaluating after fine-tuning network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,steps=(testing_len // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
#plot_training(H, 20, config.UNFROZEN_PLOT_PATH)

#saving the model to disk
model.save(config.MODEL_PATH)

