
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')
import platform
python_v = platform.python_version()
print (python_v)


# In[2]:


import numpy as np
import wget
import zipfile
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input


# 
# ## Load data
# 

# In[3]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

def download_dataset(url, path='./'):
    print("  Downloading dataset from", url)
    return wget.download(url)

def unzip(filename, path='./'):
    with open(filename, 'rb') as f:
        z = zipfile.ZipFile(f)
        print("  Unzipping file", filename)
        for name in z.namelist():
            print("    Extracting file", name)
            z.extract(name,path)
        
def load_dataset(path):
    data = load_files(path)
    class_files = np.array(data['filenames'])
    class_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return class_files, class_targets

train_files, train_targets = load_dataset('shipImages/train')
valid_files, valid_targets = load_dataset('shipImages/valid')
test_files, test_targets = load_dataset('shipImages/test')

ship_names = [item[20:-1] for item in sorted(glob("shipImages/train/*/"))]

# Let's check the dataset
print('There are %d total ship categories.' % len(ship_names))
print('There are %s total ship images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training ship images.' % len(train_files))
print('There are %d validation ship images.' % len(valid_files))
print('There are %d test ship images.'% len(test_files))


# In[4]:



# ## Pre-process data

# In[5]:


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


# In[6]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# ## Fine-tune a pre-trained model (transfer learning)

# In[7]:


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

def extract_Resnet50(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)


# ## Extract feature

# In[8]:


train_resnet50 = extract_Resnet50(train_files)
valid_resnet50 = extract_Resnet50(valid_files)
test_resnet50 = extract_Resnet50(test_files)
print("Resnet50 shape", train_resnet50.shape[1:])


# ## Retrain the last layers for our data

# In[10]:


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


# ## Compile and fit the network

# In[12]:


model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/bestmodel.hdf5', 
                               verbose=1, save_best_only=True)
model.fit([train_resnet50], train_targets, 
          validation_data=([valid_resnet50], valid_targets),
          epochs=10, batch_size=4, callbacks=[checkpointer], verbose=1)


# ## Test the model

# In[13]:


model.load_weights('saved_models/bestmodel.hdf5')

from sklearn.metrics import accuracy_score

predictions = model.predict([test_resnet50])
type_predictions = [np.argmax(prediction) for prediction in predictions]
type_true_labels = [np.argmax(true_label) for true_label in test_targets]
print('Test accuracy: %.4f%%' % (accuracy_score(type_true_labels, type_predictions) * 100))

