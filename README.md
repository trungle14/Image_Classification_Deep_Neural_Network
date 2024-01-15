# 🌟 Image Classification - Cats 😺 vs Dogs Problem 🐶 🌟



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

@misc{dogs-vs-cats-redux-kernels-edition,
    author = {Will Cukierski},
    title = {Dogs vs. Cats Redux: Kernels Edition},
    publisher = {Kaggle},
    year = {2016},
    url = {https://kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition}
}

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
```
