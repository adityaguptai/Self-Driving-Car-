# Self Driving Car (End to End CNN/Dave-2)
![alt img](https://cdn-images-1.medium.com/max/868/0*7dReqQXElneHBWUr.jpg)
Please so refer the [Self Driving Car Notebook](./Self_Driving_Car_Notebook.ipynb) for complete Information and Visualization

A TensorFlow/Keras implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes.

### How to Use
Download Dataset by Sully Chen: [https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view]
Size: 25 minutes = 25{min} x 60{1 min = 60 sec} x 30{fps} = 45,000 images ~ 2.3 GB

Note: You can run without training using the pretrained model if short of compute resources

Use `python3 train.py` to train the model

Use `python3 run.py` to run the model on a live webcam feed

Use `python3 run_dataset.py` to run the model on the dataset

To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ into your web browser.

### Credits & Inspired By
(1) https://github.com/SullyChen/Autopilot-TensorFlow<br>
(2) Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]<br>
(3) Nvidia blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ <br>
(4) https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/