
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


# ## Load data

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
    ship_files = np.array(data['filenames'])
    ship_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return ship_files, ship_targets

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


# ## Pre-process data

# In[4]:


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


# In[5]:


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


# ## Define the network

# In[6]:


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Activation, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', use_bias=False, input_shape=(224, 224, 3)))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dense(133, activation='softmax'))
model.summary()


# In[7]:


## Compile the network and fit to data


# In[8]:


from keras.callbacks import ModelCheckpoint  

EPOCHS = 10
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)
model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=EPOCHS, batch_size=32, callbacks=[checkpointer], verbose=1)


# ## Load the pre-trained weights (transfer learning)

# In[ ]:


model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# ## Test the model

# In[ ]:


# get index of predicted ship type for each image in test set
ship_type_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# report test accuracy
test_accuracy = 100*np.sum(np.array(ship_type_predictions)==np.argmax(test_targets, axis=1))/len(ship_type_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

