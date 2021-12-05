import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout 
from tensorflow.keras.optimizers import Adam

from cnnvis import Visualizer

from keras.models import load_model
 

m = load_model('../models/model_dl_balanced_2conv1dense.h5')
 

visualizer = Visualizer(model=m, image_shape=(28, 28, 1), batch_size=1)
 


from matplotlib import pyplot as plt
import numpy as np


kernels = visualizer.get_kernels('conv2d') 
print(np.mean(kernels[:, :, :, 1],axis=-1))
plt.matshow(np.mean(kernels[:, :, :, 1], axis=-1),cmap='gray')
plt.show() 
 
kernels = visualizer.get_kernels('conv2d_1')
print(np.mean(kernels[:, :, :, 1],axis=-1))
plt.matshow(np.mean(kernels[:, :, :, 1], axis=-1),cmap='gray')
plt.show() 