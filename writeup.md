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

[image1]: ./NvidiaArch.png "Nvidia Architecture"
[image2]: ./ReducedData.png "Data Histogram"
[image3]: ./center_2018_01_15_20_58_50_532.jpg "training"
[image4]: ./originaldata.png "original data"
[image5]: ./angle.png "greater than0.33"
[image6]: ./angleshuffle.png "angle rotated"
---
# Files Submitted

* Track-Driving-Training.ipynb the jupyter notebook containing the code for the model
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode(edited from original)
* model.h5 containing a trained convolution neural network 
* https://youtu.be/TrXmzi4mkAA a recording of the model driving on the first track (too big to upload)


##  Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around track 1 by executing 

python drive.py model.h5


##  Pipeline and training

The Track-Driving-Training.ipynb. This file is the pipeline I used for training and validating the model.
The drive.py file has been edited from its original code to also include preprocessing of images(lines 50-65 and 81-83). This is the same preprocessing that follows 
Nvidia's recomendation of changing the image size  and converting it to YUV.

 *Preprocess image function (distortion, color space conversion)

 1. Crop the image to keep pixels 50-140 on the y axis while retaining the x axis and the 3(bgr) layers
 2. Resize the image to minimize the required memory and make the training faster, the target resize is the recommended amount from the Nvidia documentation
 3. Apply a small gaussian blur to reduce noise
 4. Convert the image to YUV color space to better the contrast for the learning process. As the Nvidia doc suggests, the YUV will allow the model to learn view the contrasting terrains/edges better
 
  ![alt text][image4]    ![alt text][image3]
  
  Then use the  generator method to create extra training data by applying changes to the current images and adding them as new data.
  The first change is to make sure the images match our pre processing from the drive.py so we call pre_process() on each of them.
  Next we shuffle the data and if the angle size is greater or less than .33 we create a mirrored image and apply the opposite(to the original) angle and append that data to the data set. We then shuffle the data again before yielding the batch size back
  
  Since the data created  and shuffled by the generator provides enough changes we use the initial data set for both training and validation sets
  
  we then use the data to train the model.
 
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

the model used is the Keras model with Nvidia cnn 
we implemented the Normalization layer with Keras lamda, and ended up with three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers - and as described in the paper text - including converting from RGB to YUV color space, and 2x2 striding on the 5x5 convolutional layers. 
Here is a visualization of the model:

![alt text][image1]

 I also added a pooling layer to the last two to  the last two convolution layers, it will progressively reduce the spatial size of the representation to reduce the amount of parameter computation, which in turn will help with any over fitting problems

      model.add(Convolution2D(64,3,3,activation="relu"))
      model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
      model.add(Convolution2D(64,3,3,activation="relu"))
      model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

##  Parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 193).

##  Training data

Training data used was provided by Udacity.



## Creating  the Training Set & Training Process

The data set initially contained 24108 images and 8036 angle recordings of track 1 from various runs and after several attempts to train the model on the data without modifications 
I realized that it contains a lot of "useless" angle values, specifically a lot(almost half) of entries where the angle is 0 which resulted in a model with a bias towards not making adjustments
and not steering as much as needed. To work around this issue I cycled through the dataset and reduced its size by removing rows of data where the same angle is repeated more than the average angles of the dataset.
The result of the data deletion can be seen in this histogram:

![alt text][image2]


This resulted in a much smaller starting training set but was overcome with the creation of extra data by mirroring the remaining images and angle values through the generator function "generate_training_data".
The validation set was created from the reduced training set through the generator function too. The generator function shuffles the data to avoid over fitting and create a reliable validation set.

![alt text][image5]    ![alt text][image6]


#### Resources:
Training set - Udacity

Simulator - Udacity

 https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project https://github.com/o-ali/BehavioralCloning https://github.com/ksakmann/CarND-BehavioralCloning
