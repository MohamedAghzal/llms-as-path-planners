In order to run the training. You need to first convert your data into the appropriate format. To do this, you can run the following

``python convert.py $path_to_training_data $setting``

Where:

- **$path_to_training_data** is a relative path to the json file containing the training data
- **$setting** ``single`` for single-goal and ``multi`` for multi-goal. 

Afterwards navigate to the directory of the model you would like to train (current supported ones are ``T5`` and ``BART``) and run the following:

``./script.sh`` 

This assumes that you are running the training on ``SLURM`` with 2 ``A100.40gb`` GPUs. You can customize the file ``script.sh`` to better fit your platform/needs. 
