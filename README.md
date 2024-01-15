# 🌟 Image Classification - Cats 😺 vs Dogs 🐶 Problem 🌟



## Table of Contents

1. Problem overview
2. Data processing\
   2.1. Import Libraries\
   2.2. Preprocessing\
        2.2.1. Extract and generate image\
        2.2.2. EarlyStopping setting
3. Models Training\
   3.1. Convolutional Network\
   3.2. Convolutional Network with multiple different epoch\
   3.3. Stack model with transfer learning and convolutional network\
   3.4. Transfer learning Xception model
4. Evaluate the model\
   4.1. Generate prediction\
   4.2. Confusion Matrix
5. Conclusion



### 1. Problem overview 


For an effective approach combining both Deep Neural Networks (DNNs) and Transfer Learning in classifying images of cats and dogs, you can start with a pre-trained DNN model, such as ResNet, VGG, or Inception, which has already learned rich feature representations from a large and diverse dataset. Then, you adapt this model to our specific task (classifying cats and dogs) by fine-tuning some of its layers with our dataset of cat and dog images. This method utilizes the advanced feature extraction capabilities of DNNs and the efficiency of Transfer Learning, enabling our model to achieve high accuracy with less training data and time.

Will Cukierski. (2016). Dogs vs. Cats Redux: Kernels Edition. Kaggle. https://kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition


## 2. Data processing
   ### 2.1. Import Libraries

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential


from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


   ## 2.2. Preprocessing
   ### 2.2.1. Extract and generate image

```python
import zipfile
train_zip='../input/dogs-vs-cats-redux-kernels-edition/train.zip'
zip_ref=zipfile.ZipFile(train_zip,'r').extractall('./')

test_zip = '../input/dogs-vs-cats-redux-kernels-edition/test.zip'
zip_ref=zipfile.ZipFile(test_zip,'r').extractall('./')

import os
train_filenames = os.listdir('./train')
test_filenames = os.listdir('./test')

# Create DataFrame with ImageDataGenerator
train = pd.DataFrame(columns=['path', 'label'])
train['path'] = train_filenames
train['label'] = train['path'].str[0:3]

train.label.value_counts().plot.bar() # Balanced Data

width, height = 150, 150
trainDatagen = train_datagen.flow_from_dataframe(train, directory = './train', x_col='path', y_col='label', classes=['cat', 'dog' ],
                                           target_size=(width,height), class_mode = 'categorical', batch_size = 16,
                                           subset='training')

valDatagen = train_datagen.flow_from_dataframe(train, directory = './train', x_col='path', y_col='label', classes=['cat','dog'],
                                           target_size=(width,height), class_mode = 'categorical', batch_size = 16,
                                           subset='validation')


x, y = trainDatagen.next()
x.shape, y.shape

# Display the training data

plt.figure(figsize=(15,15))
for i in range(9):
    img, label = trainDatagen.next()
    plt.subplot(331+i)
    plt.imshow(img[0])
plt.show()

# Test data

test = pd.DataFrame(columns=['path'])
test['path'] = test_filenames
test.head()

test_datagen = ImageDataGenerator(rescale=1/255.0)
width, height = 150, 150
testDatagen = test_datagen.flow_from_dataframe(test, directory = './test', x_col='path', class_mode= None,
                                           target_size=(width,height), batch_size = 16, shuffle=False)

```


## 3. Models Training & Prediction
### 3.1. Convolutional Network

```python
model = models.Sequential()
model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
 
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
    
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
 
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(2,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer=optimizers.Adam(learning_rate=1e-4),metrics=['acc'])
 
model.summary()

history = model.fit(trainDatagen, steps_per_epoch = len(trainDatagen), epochs=10, validation_data = valDatagen, validation_steps=len(valDatagen), shuffle=True)

predictions = model.predict(testDatagen, batch_size=32, verbose =1)
predictions

submission = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submission['label'] = predictions[:,0]
submission.to_csv('submission_cnn_epoch10.csv', index=False)
submission.head()
```

```python
history = model.fit(trainDatagen, steps_per_epoch = len(trainDatagen), epochs=15, validation_data = valDatagen, validation_steps=len(valDatagen), shuffle=True)

predictions1 = model.predict(testDatagen, batch_size=32, verbose =1)
predictions1

submission1 = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submission1['label'] = predictions1[:,0]
submission1.to_csv('submission_ep15_1.csv', index=False)
submission1.head()
```

