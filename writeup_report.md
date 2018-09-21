# **Behavioral Cloning**

## Writeup Template
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./out_images/sample_Original.jpg "Original image"
[image2]: ./out_images/sample_Cropped.jpg "Cropped image"
[image3]: ./out_images/sample_Resized.jpg "Resized image"
[image4]: ./out_images/sample_GaussianNoised.jpg "Gaussian Noised image"
[image5]: ./out_images/sample_HistogramEqualized.jpg "Histogram Equalized image"
[image6]: ./out_images/sample_GaussianBlurred.jpg "Gaussian Blur image"
[image7]: ./out_images/sample_Normalized.jpg "Normalized image"
[image8]: ./out_images/left_recovery.jpg "Left Recovery image"
[image9]: ./out_images/right_recovery.jpg "Right Recovery Image"
[hist1]: ./out_images/angles_hist.jpg "Steering angles histogram"
[hist2]: ./out_images/angles_hist_normal.jpg "Steering angles histogram"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* process_images.py for image processing and data augmemntation process
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture similar with nVidia network.

Nvidia network architecture can be found here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

....

#### 2. Attempts to reduce overfitting in the model

Because I already pre-processed the data I decided not to add the dropou layers to the model to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate = 0.001 (model.py 88 ).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used:
* 3 full circles records for center of the road driving
* 2 full circles records for center of the road driving in reverse direction
* records with recovering driving from the left and right sides of the road
* records with driving in complicated places on track (  turns of the road )



For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to check, that simulator is works. I write simple and primitive ( and fast! ) network primitive_network ( model.py line 55-60 ) and get result simulator works, but car pulled off the track.

Then I copy nVidia network ( model.py line 63-81 ) and try it. This model has a hudge number of parameters, as result I spent a lot of time while I train a model. So I decide to reduce count of fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

After that I make some experiments with different activation functions for convolution layers, batch sizes. Finally, I decide to use Relu activation and batch size equals 128

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-81) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Lambda         		| 160x200x3 Input RGB image   					|
| Convolution2D			| 5х5 convolution, 2x2 pooling, ReLu activation	|
| Convolution2D			| 5х5 convolution, 2x2 pooling, ReLu activation	|
| Convolution2D			| 5х5 convolution, 2x2 pooling, ReLu activation	|
| Convolution2D			| 3х3 convolution, ReLu activation	            |
| Convolution2D			| 3х3 convolution, ReLu activation	            |
| Flatten				|      									        |
| Fully connected		| outputs 100									|
| Fully connected		| outputs 50									|
| Fully connected		| outputs 10									|
| Fully connected		| outputs 1										|


#### 3. Creation of the Training Set & Training Process


To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

Then I recorded two laps on track using center lane driving in reverse direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back on track:

![alt text][image8]
![alt text][image9]

After that, for each image from center, left, right camera in collected dataset I perform following augmemntation:
* Crop image from 160x320 to 105x320 ( crop trunk from bottom and sky from top )
![alt text][image2]
* Resize image from 105x320 to 66x200
![alt text][image3]
* Add Gaussian noise with probabiluty 0.1
![alt text][image4]
* Equalize image histogram
![alt text][image5]
* Add Gaussian blur
![alt text][image6]
* Normalize image
![alt text][image7]

For each center image I save steering angle without correction. For each left image I save steering angle with correction 0.2. For each right image I save steering angle with correction -0.2

Also, for each image I perform horizontal flip operation. For each flipped images I save inverted steering angle.

This procedure allows me to increase amount of picture six times without any additional training


After the collection process, I had 102414 number of data points.


Because track include long sections without turns and any curves, the captured data has many images with steering angles equals zero.
Using function `process_all_images` in file `process_images.py` I collects all races in one dataset. After that I try to check histogram of steering angles.

![alt text][hist1]



Then I try to reduce amount of steering angles close to zero. For this purpose I once again make steering angles with 27 number of beans. I calculate value `avg_angles_per_bin` equal average numbers of images in bean. For each bean I calulate number of steering angles and probability to drop image and streering angle from dataset wich will be:
* equal 1.0 - if number of images in bean less than 1/2 of `avg_angles_per_bin`
* 1.0/(hist[i]/avg_angles_per_bin * 0.5) - if numbers of images in bean more than 1/2 of `avg_angles_per_bin`


So, as result I have following histogram:
![alt text][hist2]

So, finally I had 21350 number of data points.

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation loss dynamics and simulation tests. I used an adam optimizer with learing rate = 0.001.
