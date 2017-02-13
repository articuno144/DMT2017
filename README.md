# DMT2017
This repository is contains the work done for the DMT project 30 (Drones). This directory is for circulating the data processing codes and the data only. The main C# form will not be uploaded here.

**Dependencies:**

 - pandas
 - tensorflow
 - numpy
 - matplotlib

This project aims to create an human machine interface (HRI) for drone control. Specifically, an armband will be created to recognise the user's gesture and direction command inputs. These information would then be used to control drones in a C# simulation form and possibly real drones in May. Our project current include a convolutional net for gesture recognition constructed with tensorflow under /CNN/, simple camera steaming under /camera_test/, and named pipe prototypes under /pipe_test/.

### Gesture Recognition
Main work for gesture recognition is done in the folder /CNN/. As the name suggests its done with a convolutional neural network. We used 2 convolutional layers for both accelerometer and MMG data, and added two fully connected layers after that.
