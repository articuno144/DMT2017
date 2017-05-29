# DMT2017
This repository is contains the work done for the DMT project 30 (Drones). The C# forms are not included.

**Dependencies:**

 - pandas
 - tensorflow
 - numpy
 - matplotlib
 - cv2

This project aims to create a human machine interface (HRI) for drone control. Specifically, an armband will be created to recognize the user's gesture and direction command inputs. The results obtained by processing these inputs would then be used to control drones in a C# simulation and a real drone. Our project includes a convolutional net for gesture recognition constructed with tensorflow under /CNN/, drone locating system with three cameras, and the drone control PID control loop.

### Gesture Recognition
Main work for gesture recognition is in the folder /CNN/. As the name suggests, gestures are recognized with a convolutional neural network. We used 2 convolutional layers for both accelerometer and MMG (mechanomyogram) data, and added two fully connected layers after that. From experiments, different users could have drastically distinct muscle signal outputs for the same gesture. Hence the neural network needs to be tuned for each user. The network weights are pre-trained with signals from 23 participants, and the actual users would be requested to record 5 signal samples for each gesture to be recognized to further tune the two fully connected layers (takes about 20 seconds to tune).

![CNN_structure](C:\DMT2017\photos\CNN_structure.png)

<!--insert the confusion matrix for gesture recognition accuracy-->

### Drone Locating System with Webcams

This project is intended to allow the control of a drone swarm with the armband. Accurate knowledge on the drone locations are critical in swarm control, as this information is required to prevent drone collisions. The drones that we use does not have onboard cameras, so external locating system is required to detect the drone location. 

<!--insert camera set up and screen shots-->

![coord_1](C:\DMT2017\photos\coord_1.png)

![coord_2](C:\DMT2017\photos\coord_2.png)

### PID control for the drones

Currently, the integral controller is not yet implemented.

The controller takes in the desired location, and calculates the error in location and velocity based on the drone locating system. It runs in a separate thread from the locating system and gesture recognition system, and sends the drone command signal 100 times per second.

<!--insert close loop digram here-->

The video below illustrates the location control working with the drone locating system.

https://www.youtube.com/watch?v=IBKpXXAWtTo&feature=youtu.be