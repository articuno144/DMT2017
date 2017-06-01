# DMT2017
This repository is contains the work done for the DMT project 30 (Drones). The C# forms are not included.

**Dependencies:**

 - pandas
 - tensorflow
 - numpy
 - matplotlib (to visualize the neural network)
 - cv2
 - imutils

This project aims to create a human machine interface (HRI) for drone control. Specifically, an armband will be created to recognize the user's gesture and direction command inputs. The results obtained by processing these inputs would then be used to control drones in a C# simulation and a real drone. Our project includes a convolutional net for gesture recognition constructed with tensorflow under `/CNN/`, drone locating system with three cameras, and the drone control PID control loop.

### Gesture Recognition
Main work for gesture recognition is in the folder `/CNN/`. As the name suggests, gestures are recognized with a convolutional neural network. We used 2 convolutional layers for both accelerometer and MMG (mechanomyogram) data, and added two fully connected layers after that. From experiments, different users could have drastically distinct muscle signal outputs for the same gesture. Hence the neural network needs to be tuned for each user. The network weights are pre-trained with signals from 23 participants, and the actual users would be requested to record 5 signal samples for each gesture to be recognized to further tune the two fully connected layers (takes about 20 seconds to tune).

![CNN_structure](https://github.com/articuno144/DMT2017/blob/master/photos/CNN_structure.png)

<!--insert the confusion matrix for gesture recognition accuracy-->

### Drone Locating System with Webcams

This project is intended to allow the control of a drone swarm with the armband. Accurate knowledge on the drone locations are critical in swarm control, as this information is required to prevent drone collisions. The drones that we use does not have onboard cameras, so external locating system is required to detect the drone location. 

![coord_1](https://github.com/articuno144/DMT2017/blob/master/photos/coord_1.png)

![coord_2](https://github.com/articuno144/DMT2017/blob/master/photos/coord_2.png)

Three methods have been tried out to locate the drones.

##### Movement detection

This method compares the webcam readings with the first frame that it captures. It applies a Gaussian blur on the gray scale each frame and calculates the absolute difference between the real-time frame with the first frame. A threshold is then taken to locate the moving objects. 

A disadvantage with this method is that it is sensitive to ambient light, rendering it unsuitable for the exhibition on 13th June. Furthermore, it is intrinsically difficult to differentiate between two moving drones with this method as each webcam is not able to tell which drone is which. As a result, each pair of webcams would give $\frac{2^2}{2!} = 2$ possible location. We have 3 webcams, giving $3C2 = 3$ pairs, and comparisons are required to reduce the possible locations.

This part is adapted from: <!--Ask David-->

##### Colored ball tracking

By attaching colored balls on the drones, we are able to track each of the drones individually. Here, we are applying color filters on each frame, so that only the balls are identified by the webcams. This method is also a bit more robust to light changes. Hence it will be used at the demo.

This part is adapted from: <!--Ask Xinyang-->

Both of the webcam systems have a delay of about 0.15s due to the hardware. This limits us to non-aggressive, linearized control of the drone.

##### Kinect

Kinect detects the depths as well as the drones' locations on the frame. It is robust and simple, and it has a lower delay compared to the webcams. However we ordered it late so this part is still under development.

### PID control for the drones

Currently, the integral controller is not yet implemented.

The controller takes in the desired location, and calculates the error in location and velocity based on the drone locating system. It runs in a separate thread from the locating system and gesture recognition system, and sends the drone command signal 100 times per second.

![PID](https://github.com/articuno144/DMT2017/blob/master/photos/PID.png)

[This video](https://www.youtube.com/watch?v=IBKpXXAWtTo&feature=youtu.be) illustrates the location control working with the drone locating system (movement detection).
