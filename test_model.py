from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

def load_dataset(path):
    data = load_files(path)
    ship_files = np.array(data['filenames'])
    ship_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return ship_files, ship_targets

test_files, test_targets = load_dataset('shipImages/test')
ship_names = [item[20:-1] for item in sorted(glob("shipImages/train/*/"))]

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
test_tensors = paths_to_tensor(test_files).astype('float32')/255

###########################
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

# def extract_Resnet50(file_paths):
#     tensors = paths_to_tensor(file_paths).astype('float32')
#     preprocessed_input = preprocess_input_resnet50(tensors)
#     return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

# test_resnet50 = extract_Resnet50(test_files)
# print("Resnet50 shape", test_resnet50.shape[1:])

# from keras.layers.pooling import GlobalAveragePooling2D
# from keras.layers.merge import Concatenate
# from keras.layers import Input, Dense
# from keras.layers.core import Dropout, Activation
# from keras.callbacks import ModelCheckpoint
# from keras.layers.normalization import BatchNormalization
# from keras.models import Model

# def input_branch(input_shape=None):
    
#     size = int(input_shape[2] / 4)
    
#     branch_input = Input(shape=input_shape)
#     branch = GlobalAveragePooling2D()(branch_input)
#     branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
#     branch = BatchNormalization()(branch)
#     branch = Activation("relu")(branch)
#     return branch, branch_input

# resnet50_branch, resnet50_input = input_branch(input_shape=(1, 1, 2048))
# net = Dropout(0.3)(resnet50_branch)
# net = Dense(640, use_bias=False, kernel_initializer='uniform')(net)
# net = BatchNormalization()(net)
# net = Activation("relu")(net)
# net = Dropout(0.3)(net)
# net = Dense(133, kernel_initializer='uniform', activation="softmax")(net)

# model = Model(inputs=[resnet50_input], outputs=[net])
# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
# model.load_weights('ship_models/bestmodel.hdf5')

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

def extract_Resnet50(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

# ## Extract feature
test_resnet50 = extract_Resnet50(test_files)
print("Resnet50 shape", test_resnet50.shape[1:])

# ## Retrain the last layers for our data
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def input_branch(input_shape=None):
    
    size = int(input_shape[2] / 4)
    
    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input

resnet50_branch, resnet50_input = input_branch(input_shape=(1, 1, 2048))
net = Dropout(0.3)(resnet50_branch)
net = Dense(640, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.3)(net)
net = Dense(133, kernel_initializer='uniform', activation="softmax")(net)

model = Model(inputs=[resnet50_input], outputs=[net])
model.summary()

# ## Test the model
model.load_weights('ship_models/bestmodel.hdf5')

from sklearn.metrics import accuracy_score

predictions = model.predict([test_resnet50])
class_predictions = [np.argmax(prediction) for prediction in predictions]
class_true_labels = [np.argmax(true_label) for true_label in test_targets]
print('Test accuracy: %.4f%%' % (accuracy_score(class_true_labels, class_predictions) * 100))

import shutil
import pathlib
import cv2
import os

def save_test_results(test_files, true_path, false_path):
    # shutil.rmtree(true_path)
    # shutil.rmtree(false_path)
    pathlib.Path(true_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(false_path).mkdir(parents=True, exist_ok=True)
    class_encoding = {0: "Fishing", 1: "Cargo", 2: "Tanker"}
    for i, img in tqdm(enumerate(test_files)):
        try:
            imname = img.split('/')[-1]
            im = cv2.imread(img)
            cv2.putText(im, "Prediction: {} True: {}".format(class_encoding[class_predictions[i]], class_encoding[class_true_labels[i]]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(true_path, imname), im) if class_predictions[i]==class_true_labels[i] else cv2.imwrite(os.path.join(false_path, imname), im)
        except:
            pass

save_test_results(test_files, 'res_true', 'res_false')

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

class_names = ["Fishing", "Cargo", "Tanker"]
#class_names = np.unique(class_predictions)
cnf_matrix = confusion_matrix(class_true_labels, class_predictions)    
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
