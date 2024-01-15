# 🌟 Image Classification - Cats 😺 vs Dogs 🐶 Problem 🌟



## Table of Contents

1. Problem overview
2. Data processing\
   2.1. Import Libraries\
   2.2. Preprocessing\
        2.2.1. Extract and generate image\
        2.2.2. EarlyStopping setting
3. Models\
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


   ## 2.2. Preprocessing
   ### 2.2.1. Extract and generate image\

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

plt.figure(figsize=(15,15))
for i in range(9):
    img, label = trainDatagen.next()
    plt.subplot(331+i)
    plt.imshow(img[0])
plt.show()


## Test data

test = pd.DataFrame(columns=['path'])
test['path'] = test_filenames
test.head()


test_datagen = ImageDataGenerator(rescale=1/255.0)
width, height = 150, 150
testDatagen = test_datagen.flow_from_dataframe(test, directory = './test', x_col='path', class_mode= None,
                                           target_size=(width,height), batch_size = 16, shuffle=False)

```




