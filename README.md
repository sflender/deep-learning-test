# deep-learning-test

various python codes to test out Keras locally and on Cooley.  
  
##Content

- deep-digits-local.py : explore a simple convnet on your laptop

- keras-test.py : code for running a convnet on Cooley and save the learning curve

- deep-digits-history.ipynb : notebook for plotting learning curves

- digit_augmentation_exploration.ipynb : notebook for exploring data augmentation on your laptop

- keras_with_augmentation.py : code for running a convnet with data augmentation on Cooley and save the learning curve.  
  
- parallel-keras-test.py : use both Cooley GPU's
  
## Installing tensorflow and Keras on Cooley

I followed instructipon from https://gist.github.com/wscullin/70409948a5a812e0e874339a8a1a256c with the difference that I used the pre-build wheel at /soft/libraries/unsupported/tensorflow-whl-1.3.0/  

My soft environment is set up like this:
+mvapich2
+gcc-4.9.3
+cuda-7.5.18
+git-2.10.0
+java-1.8.0.60
LD_LIBRARY_PATH+=/soft/libraries/unsupported/cudnn-7.5.1/cuda/lib64
@default
  
First create a new conda environment:
`conda create -n "test_env" python=2.7 anaconda`
  
activate the environment: 
`source activate test_env`

pip install of the tensorflow wheel:  
`pip install /soft/libraries/unsupported/tensorflow-whl-1.3.0/tensorflow-1.3.0-cp27-cp27mu-linux_x86_64.whl` 
  
also install keras to run the exaple code:  
`pip install keras` 
  
now get an interactive node:
`qsub -I -A datascience -t 00:30:00 -n 1 -q debug`

activate the environment:
`source activate test_env`

To see if your tensorflow installation sees both of the GPUs on one Cooley node, type this into a python shell: 
  
`from tensorflow.python.client import device_lib`
`device_lib.list_local_devices()` 

now you can run the example:  
`python keras-test.py`

