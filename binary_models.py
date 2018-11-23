
from __future__ import division
import os
import glob
import zipfile
import functools
from skimage import io
from skimage.segmentation import quickshift
import glob
from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.model_selection import train_test_split
#import matplotlib as mpl
#mpl.rcParams['axes.grid'] = False
#mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
#import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

import numpy as np
from skimage import color


# In[6]:


import tensorflow as tf
import os
import numpy as np
import json
from PIL import Image
#import matplotlib
#import matplotlib.pyplot as plt
import pandas as pd
#from matplotlib.collections import PatchCollection

from skimage import io
from skimage.segmentation import quickshift
import glob
#import seaborn as sns


# In[7]:


img_dir='conv_data/sat_match2'
label_dir='conv_data/gt_binary_masks/dark_green'

x_train_filenames = []
y_train_filenames = []
for index in range( 14):
  
  x_train_filenames.append(os.path.join(img_dir, "{}.png".format(index)))
  y_train_filenames.append(os.path.join(label_dir, "{}.png".format(index)))
  #print(index)



# In[8]:



sz=(1024,1024)
images_gt=[]
images_sat=[]

for i in range(0,14):
  img_num=i
  x_pathname = x_train_filenames[img_num]
  y_pathname = y_train_filenames[img_num]
  
 

  a=((io.imread(x_pathname)))[...,0:3]
  #a=cv2.resize(a,sz)
  images_sat.append(a)
  
  
  
  b= color.rgb2gray(io.imread(y_pathname)[...,0:3])
  #b=cv2.resize(b,sz)
  images_gt.append(b)
  print(i)
  


# In[9]:


images_gt=np.asarray(images_gt)
images_sat=np.asarray(images_sat)


# In[ ]:





# In[10]:



h,w=256,256

split_sat=[]
split_gt=[]
t=0
for i in range(0, 14):
 
 img_num=i
 x,y=0,0
 x_pathname = x_train_filenames[img_num]
 y_pathname = y_train_filenames[img_num]
 im_sat=images_sat[i]
 im_gt=images_gt[i]
 n=i
 print(i)
 
 while 1:
   x=0
   while 1:
     
     fname_lab=os.path.join(label_dir, "{}_{}_{}.png".format(n,x,y))
     fname_img=os.path.join(img_dir, "{}_{}_{}.png".format(n,x,y))
     a,b,c=im_sat[y:y+h, x:x+w].shape
     c,d=im_gt[y:y+h, x:x+w].shape
     #print(c)
     #print(d)
     
     
     if(a==256 and b==256 and c==256 and d==256):
       
       t=t+1
       
       #print(t)
       if(t==1):
         crop_img_x = im_sat[y:y+h, x:x+w]#.flatten()
         a=crop_img_x.reshape(1,h*w*3)
         #df_sat=pd.DataFrame(a)
         crop_img_y = im_gt[y:y+h, x:x+w]
         #b=crop_img_y.reshape(1,h*w*1)
         split_sat.append(crop_img_x)
         split_gt.append(crop_img_y)
         #df_gt=pd.DataFrame(b)
         #print("dddddddddd")
       else:
         crop_img_x = im_sat[y:y+h, x:x+w]#.flatten()
         a=crop_img_x.reshape(1,h*w*3)
         #temp=pd.DataFrame(a)
         #df_sat = pd.concat([df_sat, temp], axis=0, ignore_index=True)
         #plt.imsave(fname_img,crop_img_x)
         #print(crop_img_y.shape)
         split_sat.append(crop_img_x)

         crop_img_y = im_gt[y:y+h, x:x+w]#.flatten()
         #b=crop_img_y.reshape(1,h*w*1)
         #temp2=pd.DataFrame(b)
         #df_gt = pd.concat([df_gt, temp2], axis=0, ignore_index=True)
         #plt.imsave(fname_lab,crop_img_y)
         split_gt.append(crop_img_y)
     x=x+40
    # print(x)
     if(x>=600):
       break
   y=y+40
   if(y>=600):
     break
     
   
 


# In[11]:


split_gt=np.asarray(split_gt)
split_sat=np.asarray(split_sat)


# In[12]:


print(split_gt[1].shape)
a=split_gt[1]


# In[13]:



img_shape = (256, 256, 3)

batch_size = 64
epochs = 5


# In[14]:


X_train, X_test, y_train, y_test= train_test_split(split_sat, split_gt, test_size=0.3)


# In[32]:


im_shape = (256, 256, 3)
im_shape_label=(256,256,1)

X_train = X_train.reshape(X_train.shape[0], *im_shape)/255
y_train = y_train.reshape(y_train.shape[0], *im_shape_label)
X_test = X_test.reshape(X_test.shape[0], *im_shape)/255
y_test = y_test.reshape(y_test.shape[0], *im_shape_label)


# In[34]:


print(np.unique(y_train[1]))


# In[17]:


num_train_examples = len(X_train)
num_val_examples = len(X_test)
print(num_train_examples)


# In[18]:





import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D

from keras import backend as K

import keras
import h5py

from keras.layers.normalization import BatchNormalization


from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

import datetime
import os

import random
import threading
from tensorflow.python.keras import layers
from keras.models import model_from_json
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from keras import utils


# In[19]:


def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_uniform')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.advanced_activations.ELU()(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_uniform')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.advanced_activations.ELU()(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_uniform')(input_tensor)
  #print(input_tensor.shape)
  #print(decoder.shape)
  decoder = layers.concatenate([concat_tensor, decoder],axis=-1)
  #decoder = layers.BatchNormalization()(decoder)
  #decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_uniform')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  
  
  decoder = layers.advanced_activations.ELU()(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same',kernel_initializer='he_uniform')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.advanced_activations.ELU()(decoder)
  return decoder


# In[20]:


inputs = layers.Input(shape=img_shape)

# 256
#print(inputs.shape)
encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 128
#print(encoder0_pool.shape)
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 64

encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 32

encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 16

encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
# 8

center = conv_block(encoder4_pool, 1024)
# center

decoder4 = decoder_block(center, encoder4, 512)


decoder3 = decoder_block(decoder4, encoder3, 256)
# 32

decoder2 = decoder_block(decoder3, encoder2, 128)
# 64

decoder1 = decoder_block(decoder2, encoder1, 64)
# 128

decoder0 = decoder_block(decoder1, encoder0, 32)


outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(decoder0)





# In[21]:


model = models.Model(inputs=[inputs], outputs=[outputs])


# In[24]:


save_model_path = 'model/weights_lightgreen.hdf5'
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)


# In[42]:




import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D

from keras import backend as K

import keras
import h5py

from keras.layers.normalization import BatchNormalization


from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import binary_crossentropy
from tensorflow.python.keras import layers


import datetime
import os

import random
import threading

from keras.models import model_from_json

img_rows = 256
img_cols = 256


smooth = 1e-12

num_channels = 3
num_mask_channels = 1


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

'''
def get_unet0():
    inputs = Input(img_shape)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
  
    a=UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.concatenate([a, conv4], axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization(mode=0, axis=1)(crop9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model
  
  '''


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, img_rows, img_cols ,num_channels))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols, num_channels))
    X_height = X.shape[1]
    X_width = X.shape[2]

    for i in range(batch_size):
        #random_width = random.randint(0, X_width - img_cols - 1)
        #random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image]#, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = np.array(X[random_image])#, :, random_height: random_height + img_rows, random_width: random_width + img_cols])
    return X_batch, y_batch


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            y_batch[i] = yb

        yield X_batch, y_batch[:, :, 16:16 + img_rows - 32, 16:16 + img_cols - 32]

'''
def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model



data_path = '../data'
now = datetime.datetime.now()

print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))


print('[{}] Reading train...'.format(str(datetime.datetime.now())))
f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

#X_train = f['train']

#y_train = np.array(f['train_mask'])[:, 5]
#y_train = np.expand_dims(y_train, 1)
print(y_train.shape)

train_ids = np.array(f['train_ids'])

'''

datagen = ImageDataGenerator(
        )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    #history=model.fit_generator(datagen.flow(x_train, y_train,
                       

batch_size = 2
nb_epoch = 50

history = History()
callbacks = [
    history,
]
#a=batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True)
suffix = 'crops_3_'
model.compile(optimizer=tf.contrib.opt.NadamOptimizer(learning_rate=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
model.fit(X_train, y_train,
                    
                    steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                    epochs=nb_epoch,
                    verbose=1,
                    #samples_per_epoch=batch_size * 400,
                    use_multiprocessing=True,
                    callbacks=[cp],
                     workers=8
                    )

#save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
#save_history(history, suffix)
#a=batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True)

#save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
#save_history(history, suffix)
#f.close()


# In[40]:


tf.test.is_gpu_available(
    cuda_only=True,
    min_cuda_compute_capability=None
)


# In[43]:


print(tf.__version__)


# In[ ]:


suffix = 'crops_4_'
model.compile(optimizer=tf.contrib.opt.NadamOptimizer(learning_rate=1e-4), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
    batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
    steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
    epochs=nb_epoch,
    verbose=1,
    use_multiprocessing=True,
    #samples_per_epoch=batch_size * 400,
    callbacks=[cp]
    )


# In[ ]:




