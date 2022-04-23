#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders as sf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers


# In[2]:


train_dir=r'C:\Users\swati gour\Desktop\CNN-Final\train'
val_dir=r'C:\Users\swati gour\Desktop\CNN-Final\val'


# In[3]:


train_datagen=ImageDataGenerator(rescale=1./255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)   #mirror image
validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                 target_size=(200,200),
                                                 batch_size=5,
                                                 color_mode='rgb')
validation_generator=validation_datagen.flow_from_directory(val_dir,
                                                 target_size=(200,200),
                                                 batch_size=5,
                                                 color_mode='rgb',
                                                           shuffle=True)


# In[4]:


model1=models.Sequential()
# 1st layer
model1.add(layers.Conv2D(32,(3,3),activation='relu',
                       input_shape=(200,200,3)))
model1.add(layers.MaxPooling2D((2,2)))

# 2nd layer
model1.add(layers.Conv2D(64,(3,3),activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))
        
# 3rd layer
model1.add(layers.Conv2D(128,(3,3),activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))

# 4th layer
model1.add(layers.Conv2D(256,(3,3),activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))


# 5th layer
model1.add(layers.Flatten())
model1.add(layers.Dropout(0.5))

#6th layer
model1.add(layers.Dense(512,activation='relu'))

# 7th layer
model1.add(layers.Dense(256,activation='relu'))
# output layer
model1.add(layers.Dense(12,activation='softmax'))


# In[5]:


model1.summary()


# In[6]:


model1.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adam(learning_rate=1e-4),
             metrics=['accuracy'])


# In[7]:


history=model1.fit(train_generator,
                           epochs=100,
                           validation_data=validation_generator)


# In[8]:


model1.save('capstone1.h5')


# In[ ]:


labels=['ACNE','ATOPIC DERMATITIS','BASAL CELL CARCINOMA','BENIGN KERATOSIS','CLEAR SKIN','ECZEMA',
       'MELANOCYSTIC NEVI','MELANOMA','PSORIASIS','SEBORRHEIC KERATOSES','FUNGAL INFECTION','WARTS']
from tensorflow.keras.preprocessing import image
#file_name='MODEL1.h5'
#model=models.load_model(file_name)
img=image.load_img(r"C:\Users\swati gour\Desktop\CNN-Final\val\PSORIASIS\0_15.jpg",color_mode='rgb',target_size=(200,200,3))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])

val=model1.predict(images)
res=val.flatten()
pred = np.where(res == np.amax(res))
index=pred[0].tolist()
print("List of Indices of maximum element :",pred[0],labels[index[0]])


# In[14]:


from tensorflow.keras import models
labels=['ACNE','ATOPIC DERMATITIS','BASAL CELL CARCINOMA','BENIGN KERATOSIS','CLEAR SKIN','ECZEMA',
       'MELANOCYSTIC NEVI','MELANOMA','PSORIASIS','SEBORRHEIC KERATOSES','FUNGAL INFECTION','WARTS']
from tensorflow.keras.preprocessing import image
file_name=r'capstone1.h5'
model=models.load_model(filepath=file_name)
img=image.load_img(r"C:\Users\swati gour\Desktop\CNN-Final\train\ECZEMA\0_28.jpg",color_mode='rgb',target_size=(200,200,3))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])

val=model.predict(images)
res=val.flatten()
pred = np.where(res == np.amax(res))
index=pred[0].tolist()
print("List of Indices of maximum element :",pred[0],labels[index[0]])

