# **Behavioral Cloning** 
   Track test driving
## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./media/NvidiaArch.png "Nvidia Architecture"



---
# Files Submitted

* Track-Driving-Training.ipynb the jupyter notebook containing the code for the model
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode(edited from original)
* model.h5 containing a trained convolution neural network 
* run1.mp4 a recording of the model driving on the first track


##  Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around track 1 by executing 

python drive.py model.h5


##  Pipeline and training

The Track-Driving-Training.ipynb. This file is the pipeline I used for training and validating the model.
The drive.py file has been edited from its original code to also include preprocessing of images(lines 50-65 and 81-83). This is the same preprocessing that follows 
Nvidia's recomendation of changing the image size  and converting it to YUV 

# Model Architecture and Training Strategy

##  Architecture

The model used is the suggested Nvidia architecture.
* Input size is 3@66x200
* Convolutional Layer: 36 feature maps 5x5 Kernal - Output:36@14x47
* Convolutional Layer: 48 feature maps 5x5 Kernal - Output:48@5x22
* Convolutional Layer: 64 feature maps 3x3 Kernal - Output:64@3x20
* Convolutional Layer: 64 feature maps 3x3 Kernal - Output:64@1x18
* Flatten
* Fully Connected Layer x4

Here is a visualization of the model:

![alt text][image1]

##  Parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 193).

##  Training data

Training data used was provided by Udacity.




#### Resources:
Training set - Udacity

Simulator - Udacity

 https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project https://github.com/o-ali/BehavioralCloning https://github.com/ksakmann/CarND-BehavioralCloning
