# 🌟 Image Classification - Cats 😺 vs Dogs 🐶 Problem 🌟



## Table of Contents

1. Problem overview
2. Data processing\
   2.1. Import Libraries\
   2.2. Preprocessing\
        2.2.1. Extract and generate image\
        2.2.2. EarlyStopping setting
3. Models Training & Prediction\
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
```
<img width="663" alt="Screenshot 2024-01-20 at 00 16 42" src="https://github.com/trungle14/Image_Classification_Deep_Neural_Network/assets/143222481/ec5d5021-06a8-400e-aed5-c6cb4984827d">

```python
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

## 3.2. Transfer learning Xception model

```python
images_size = 150
batch_size = 32

base_model = Xception(weights='imagenet', include_top=False, input_shape=(images_size, images_size, 3))
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    
    layers.Flatten(),
    
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2,activation='softmax'),
])

model.summary()

learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,  # Initial learning rate for training
    decay_steps=1000,            # Number of steps before decaying the learning rate
    decay_rate=0.5,              # Rate at which the learning rate decreases
)
optimizer = optimizers.Adam(learning_rate=learning_rate_schedule)
model.compile(optimizer=optimizer,
             loss="categorical_crossentropy",
              metrics=['accuracy']
             )
from tensorflow.keras.callbacks import LearningRateScheduler
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)
learning_rate_reduce = ReduceLROnPlateau(
    monitor='val_acc',   # Metric to monitor for changes (usually validation accuracy)
    patience=5,          # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1,           # Verbosity mode (0: silent, 1: update messages)
    factor=0.5,          # Factor by which the learning rate will be reduced (e.g., 0.5 means halving)
    min_lr=0.00001       # Lower bound for the learning rate (it won't go below this value)
)
lr_callback = LearningRateScheduler(learning_rate_schedule)
callback=[ lr_callback , learning_rate_reduce ,early_stopping ]




# ExponentialDecay for the learning rate
learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.5,
)

# Use this learning rate schedule in the optimizer
optimizer = optimizers.Adam(learning_rate=learning_rate_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Callbacks (without LearningRateScheduler)
callbacks = [
    EarlyStopping(
        min_delta=0.001,
        patience=5,
        restore_best_weights=True,
    ),
    # Optionally include ReduceLROnPlateau if you want to use it instead of ExponentialDecay
]

# Fit the model
history = model.fit(
    trainDatagen,
    steps_per_epoch=trainDatagen.samples // batch_size,
    epochs=20,
    validation_data=valDatagen,
    validation_steps=valDatagen.samples // batch_size,
    callbacks=callbacks
)



predictions_xcep = model.predict(testDatagen, batch_size=32, verbose =1)
predictions_xcep


submission1 = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submission1['label'] = predictions_xcep[:,0]
submission1.to_csv('submission_xception.csv', index=False)
submission1.head()
```


## 3.3. Stack model with transfer learning and convolutional network


```python
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate


# Load VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Custom CNN Model
custom_cnn = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),])

# Feature extraction
# Add GlobalAveragePooling2D to both models
vgg_output = GlobalAveragePooling2D()(vgg16.output)
custom_cnn_output = GlobalAveragePooling2D()(custom_cnn.output)

# Concatenate features
combined = layers.concatenate([vgg_output, custom_cnn_output])


# Additional layers
x = layers.Flatten()(combined)
x = layers.Dense(1024, activation='relu')(x)
#-- output = layers.Dense(10, activation='softmax')(x)  # Adjust number of units based on your problem
output = layers.Dense(2, activation='sigmoid')(x)


# Combined model
model = Model(inputs=[vgg16.input, custom_cnn.input], outputs=output)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# Fit the model


def generate_dual_input(generator):
    for (inputs, labels) in generator:
        yield [inputs, inputs], labels

# Create dual-input generators
train_dual_gen = generate_dual_input(trainDatagen)
val_dual_gen = generate_dual_input(valDatagen)


history = model.fit(
    train_dual_gen,
    steps_per_epoch=len(trainDatagen),
    validation_data=val_dual_gen,
    validation_steps=len(valDatagen),
    epochs=10  # Adjust as needed
)

def generate_dual_input_test(generator):
    for inputs in generator:
        yield [inputs, inputs]


test_dual_gen = generate_dual_input_test(testDatagen)
predictions_stacked = model.predict(test_dual_gen, steps=len(testDatagen), verbose=1)

submission1 = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submission1['label'] = predictions_stacked[:,0]
submission1.to_csv('submission_stacked.csv', index=False)
submission1.head()
```


```pytohon
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()
```
<img width="333" alt="Screenshot 2024-01-20 at 15 16 45" src="https://github.com/trungle14/Image_Classification_Deep_Neural_Network/assets/143222481/826f7f67-c6af-4e2f-870c-93d78d5d3648">


#### Since we do not have actual label of test dataset on kaggle in order to calculate performance metric like accuracy or precision. In this case, I would like to check it manually by displaying picture along with the predicted label.

<img width="320" alt="Screenshot 2024-01-20 at 15 18 15" src="https://github.com/trungle14/Image_Classification_Deep_Neural_Network/assets/143222481/478ac336-409b-4d50-a13c-84da1a0ccc57">

The results show that out of 21 samples out of 12500, we can get 14/21 correct at the cut-off of 80%.
Notice that these results could be relatively sensitive with the chosen cut-off. More importantly, with the small number of sample in the test dataset we can prove that our model does a great job, although the prediction is not really high but always better than random guess.

## Conclusion

I already tried 4 different models and gained different results accordingly.

Deep Convolutional network along with batch normalization : 2.4 Kaggle score
Enhance Deep Convolutional network with higher epoch: 5.4 Kaggle score
Stacked model with Transfer learning - VGG16 and Convolutional network: 1.1 Kaggle score
Transfer learning - Xception model: 10.8 Kaggle score
We can see that the Stacked model outperformed the others which is totally make sense when we can combine 2 different kind of models to increase the performacne, due to limitation of time, I only do the stacked model with only 2 models, if I had more time I believe stacked model among transfer learning like ResNet may be able to improve the performance




