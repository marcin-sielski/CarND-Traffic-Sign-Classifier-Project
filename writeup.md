# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization_of_dataset.png "Visualization of datasets"
[image2]: ./images/visualization_of_processed_images.png "Visualization of processed images"
[image3]: ./images/new_dataset.png "New dataset"
[image4]: ./images/predictions.png "Predictions"
[image5]: ./images/top_5_predictions.png "Top 5 Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/marcin-sielski/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python built-in library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below we can find distribution of images belonging to each classes for training, validation and test set. Distributions are similar for all sets however from the bars it looks like some of the classes are overrepresented and some are underrepresented which might have negative impact on training/recognition process.

![alt text][image1]

The picture above shows also randomly selected image from each dataset and its label (class).

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it will be easier for classifier to learn them.

As a last step, I normalized the image data because I wanted to stabilize learning process and reduce number of training epochs required to train classifier. I "centered" the data by substracting mean and divided it by stddev.

Image below shows an effect of preprocessing the dataset.

![alt text][image2]

I decided to not generate additional data because I was satisfied with the result of training and verification of the model on the test dataset. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based on [LeNet-5](https://engmrk.com/lenet-5-a-classic-cnn-architecture/) architecture and consisted of the following layers:

| Layer         		|     Description	        					| 
|----------------------:|:----------------------------------------------|
| __CNN__               |                                               | 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16     				|
| __Classifier__        |                                               |
| Fully connected		| Flatten, outputs 400        					|
| Fully connected       | Outputs 120                                   |
| RELU					|												|
| Droput                | keep probablility 0.55                        |
| Fully connected       | Outputs 84                                    |
| RELU					|												|
| Droput                | keep probablility 0.55                        |
| Fully connected       | Outputs 43                                    |
| Softmax				|           									| 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
* an Adam Optimizer,
* 150 epochs,
* 128 batch size,
* 0.001 learning rate with inverse time decay.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.978 
* test set accuracy of 0.955

I initially seleted well known LeNet-5 CNN model because its architecture is fairly simple and easy to learn and I was able to get pretty good results (slightly above 0.93) on validation set by just adjusting hyperparameters.

I wanted to improve the result anyway and I manage to achieve that by mainly adding dropout (reguralization) on classfier layers (~0.96). Introducing learning rate with inverse time decay constantly increased accuracy on validation set well above 0.97. I experienced undefitting when keep probabilty for dropout was setup below 0.5.

Convolution layers are main building blocks of CNN that uses filters/kernels on the input data to produce feature maps. CNN defines features (e.g. horizontal lines, vertical lines, curves etc.) only during learning process. A certain cobination of features in a certain area might idicate larger, more complex features. Classifier layers maps discovered features in the images to appropriate classes (labels).  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]

All the images I selected were very difficult to classify by the network whitout upfront preprocessing mainly beacuse of its different sizes. I used torchvision functions from pytorch framework to preprocess the images. I applied Resize, CenterCrop, Grayscale and Normalization transforms (results depicted above). Looking at results of transformation I considered "Slippery road" image to be difficult to be recognized by the network mainly beacuse the sign itself was very small and complicated. Additionaly "Slippery road" sign was underrepresented in the original dataset which might have resulted in poor recognition performance.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image4]
   							|
As expected the model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.5%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

Here are the results:
![alt text][image5]

Model was 100% sure about correct classification for all signs except "Slippery road" sign.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


