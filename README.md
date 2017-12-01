# deep-learning-test

testing out keras on MNIST data, locally and on Cooley  
  
---Content---
  
deep-digits-local.py  
explore a simple convnet on your laptop  
  
keras-test.py  
code for running a convnet on Cooley and save the learning curve  
  
deep-digits-history.ipynb  
notebook for plotting learning curves  
  
digit_augmentation_exploration.ipynb  
notebook for exploring data augmentation on your laptop  
  
keras_with_augmentation.py  
code for running a convnet with data augmentation on Cooley and save the learning curve.  
  
  
---How to make sure you are running on the GPU on Cooley---
  
Make sure you install tensorflow, tensorflow-gpu, and keras, using pip or conda.
  
To see if your tensorflow installation sees the GPU, type this in a python shell:
  
from tensorflow.python.client import device_lib
device_lib.list_local_devices()  