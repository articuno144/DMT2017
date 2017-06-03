### Introduction and background

Gesture recognition is one of the main part of our projects. Our *armband* collects the MMG and accelerometer signals from the users and uses these inputs to interpret the commands from the user. The user commands include two aspects: directions and gestures. Directional commands are interpreted by the NU interface with the accelerometers, gyroscopes and magnetometers, while the gesture commands are classified with the mechanomyogram signals (MMG, muscle vibration signals). Gesture interpretations are carried out with a <u>pretrained</u> Convolutional Neural Network (CNN) <u>calibrated for each user</u>. Constructed in python, the CNN <u>receives data</u> transmitted from the NU interface, classifies the gestures, and either pass the information back to the NU interface for simulation or uses them directly for quadcopter control. 

#### Backgrounds

##### Neural Networks

Neural networks are biologically inspired universal approximators for functions that may have large numbers of inputs and outputs. Supervised learning is a branch of machine learning problems, where the desired outputs are fed with the inputs to the machine learning algorithms, so that the underlying function that maps the outputs to the inputs could be approximated. In this project, our neural network learns to map the MMG signals to the gesture/noise classifications.

##### Fully connected layers and non-linear activation functions

The most basic neural network is constructed with the fully connected layers joined by activation functions. Fig ___ shows one unit of the fully connected layer and its biological analogy. In these layers, each neuron unit in the neural network is non-linearly activated weighted sum of the neuron layer. The weights w are the trainable parameters that approximates the desired patterns. The activation function f induces non-linearity in the neural network, providing it the ability to approximate non-linear functions. 

![neural_model](C:\DMT2017\main\report\img\neural_model.png)

The neurons between adjacent layers are fully connected as shown in Fig ___. This is a rather simplistic structure  in that the network output is independent of the input and output in the previous time steps, and that the adjacency of the input data points is not explicitly considered.![mlp](C:\DMT2017\main\report\img\mlp.png)

##### Convolutional Neural Networks

CNN is a type of neural network widely used in modern computer vision tasks that excels at identifying local features. CNN treats the adjacent inputs as local patterns as oppose to unrelated data points. CNNs are charactorised by convolutional layers, where neurons are arranged in 3 dimensions: width, height and depth, and each layer of neuron in the depth direction is a learnable filter, or feature extractor. For instance, in image classifications, these neural groups may be excited by local features like edges or brightness gradient in a certain direction, converting the images to a map of neuron excitation. A few fully connected layers are then trained to classify the image contents based on the neuron excitation map.

![depthcol](C:\DMT2017\main\report\img\depthcol.jpeg)

Figure __ shows an example of the convolutional layer. The input image is of dimension 32x32x3, and the example neurons are arranged in groups of 5x5x3 to form filters, so that each filter examines a 5x5 square in the input image. There are 5 groups, and by sweeping the filters on the input image, a result excitation map of dimension 28x28x5 is produced, assuming stride is 1 and no padding is used.

##### Pooling layers

The pooling layer is another key component in convolutional neural networks. ++++++++++

##### Dropout

+++++++++++

### Neural network structure

##### rationale

The challenge in this project is that gestures must be identified among the noise signals of other muscle movements, which could make up over 90 percent of the signals received. Specifically, the users would inevitably move their arms to control the drone directions. These muscle movements should be rejected by the gesture recognition algorithm. 

Earlier in our project, the classic signal processing techniques like filtering and applying thresholds was considered. For instance,the MMG signals caused by moving arms to control the drone direction is usually smaller than those caused by gestures, so we could potentially reject the squared and smoothed MMG signals below a certain magnitude to ignore the noises, and only investigate in the values above the threshold to classify the gestures. However, these methods require manual adjustments for each user. For instance, some people may require a larger threshold value as they have more vigorous muscle activities. In addition, some gestures may have weaker or more abrupt signals than others, and these gestures may be accidentally rejected. Neural networks, however, is not constrained by such a threshold value, as it examines the shape as well as the magnitude of the input to differentiate between gestures and noise.

Various types of neural network structures including Long-short term memory (LSTM) and multilayer perceptron (MLP) are tested, and the CNN is chosen because it gives the best result, and is able to produce these results in real time.

##### drawbacks

The most significant disadvantage of neural networks are that they work like black boxes. It is hard to locate the exact issues when the networks do not work as desired due to their intrinsic random initialisation process. However, several visualisation techniques have been utilised to interpret how the neural networks work.

Another drawback is that the neural network we used requires more computational power than many other signal processing techniques, and the computation is carried out at every time step. Although we have shown in our project that real time processing is feasible, there is still a large incentive for us to simplify the neural network.

#### Structure

A schematic of the CNN structure is shown in fig__. We are attaching 3 MMG sensors on the armband, producing three streams of data at each time step. 50 time steps of the most recent data streams are kept, corresponding to 1 second, so that effectively, a running window of 1 second of muscle vibration is fed to the neural network at each new time step. The weights in the first convolutional layer are of dimension 1x3x3x32, with each unit covering a local connectivity of height 1, width 3, and depth 3. This transforms the original 1-by-50-by-3 input to an neuron activation map of depth of 32. The weights in the second layer are of dimension 1x5x32x64. The final neuron activation map is reshaped to a vector, which is then processed with two successive fully connected layers, producing the final result. The classification result is expressed as a 9 dimensional one-hot vector, with 8 dimensions corresponding to the gesture and 1 corresponding to other muscle movements. For instance, if the neural network classifies the input as gesture 3, the output would be [0,0,1,0,0,0,0,0,0]’, and if the input is classified as noise, the output should return [0,0,0,0,0,0,0,0,1]’. 

#### Pretraining and calibration

Early experiments have shown that different users may have highly distinct muscle activities, and hence mechanomyogram (MMG) signals for the same gesture _____ insert picture__________. Thus, a desirable gesture recognition algorithm must be able to adapt to each user.

As mentioned in the background section, the two convolutional layers acts as the filters or feature extractors, and the fully connected layers converts the activation map of these filters to the result classification. Pretraining of the neural network is based on the assumption that although the users may have different muscle movements for the same gesture in the time scale of seconds, the local MMG input features should be similar in the time scale of milliseconds. Hence the trained convolutional layers can be directly applied to the new users, and calibration is only required to be performed on the two fully connected layers. This techniques reduces both computational time and required gesture examples for calibration from the new user.

```python
second_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
    cost, var_list=[weights['wd2'], weights['out'], biases['bd2'], biases['out']])
...
sess.run(second_optimizer, feed_dict={
         x: batch_x, y: batch_y, keep_prob: 0.5, learning_rate: 0.0001})
sess.run(second_optimizer, feed_dict={
         x: noise_x, y: noise_y, keep_prob: 0.5, learning_rate: 0.0005})
sess.run(second_optimizer, feed_dict={
         x: new_noise, y: new_noised, keep_prob: 0.5, learning_rate: 0.00005})
```

The calibration is carried out as shown with gradient descent.

##### Design iterations

###### Length of the running window

Originally, we our running window of MMG input has the length of 300, corresponding to 0.6s of MMG signals. We have found that the classification accuracy is unaffected if the MMG sampling rate is reduced to 50 times per second. Hence the neural network only takes the input of shape 50x3, saving the computational power significantly. On the other hand, keeping a longer window of data (increased to 1s from 0.6s) has slightly improved our classification accuracy.

###### Usage of the accelerometer data

Intuitively, the accelerometer data should help differentiate between gestures like thumb-up and thumb-down. However, it turned out that the result from 10 trials shows no significant difference between using the accelerometer readings and not using them. We hence decided to discard this branch of the network.

### Data Collection

The pretraining of the neural network is carried out with data collected from 27 participants. Each participant was requested to preform 8 gestures for 15 times each, followed by 30 seconds of arm movements without gestures. These MMG signals are saved as .csv files and processed in Matlab. Once the raw data are processed, we identified where the gestures took place, and labelled the period from 50 time steps before that point as the gesture input. As for noise (non-gesture muscle movements), the entire duration is labelled as noise and fed to the neural network. When users control quadcopters with the armbands, we predict that gesture should take up only about 5% percent of the total usage time. Hence, for isometric training, we used this percentage to distribute the training data (5% gesture vs 95% noise).

For new users, a calibration procedure is carried out before using the armband. Figure __ shows the calibration form. Once the button 'calibration' is clicked, the user would be prompted to perform the gestures for five times each when the indication light is on. The light is turned on for one second for each gesture trial, and there is a one-second interval between the gesture trials. The data recorded when the light is on (a running window ending at the instance when the light is just switched off) is labeled as input corresponding to the gesture, and the duration of the light being switched off (0.2s after light is switched off until the light is turned on) is fed to the neural network labeled as noise. This is further illustrated in Figure ___.

### Data Transmission

Data transmission between the C# form application and the CNN constructed with TensorFlow on python in this project is carried out with named pipes. A pipe server is created in the C# form and is fed with a byte array of the MMG signals as well as the roll, pitch and yaw angle of the IMU. The python program reads the byte array information posted in the pipe by the C# form, passes it through the neural network and obtains the gesture classification. Depending on the task, the gesture results are further used as shown in Figure ___.

### Test and visualization

confusion matrix, label significant values.

Parts of the input can be systematically occluded to determine the part of the input that is most crucial in classification. Concretely, the input with a small patch set to 0 can be fed to the neural network, and the result is compared between the original input and the patched input. If the classification result has changed, it is implied that the covered region has been responsible for the gesture classification. This patch is iterated across all time steps in the input signal to compare the importance of the input local regions. The results shown in Fig__ has clearly shown that the regions critical for gesture classification corresponds fairly well to our intuitions. 

Further tests of the gesture classification and directional control can be tested in the simulation form in section ___.