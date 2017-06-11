## Changes to be made on the slides

NN - 1st page: NN only, show MMG signal example

add flow charts

## Presentation Script

#### Gesture Recognition

To achieve gesture control of the drones, we need to interpret the inputs from the users with the armband. The main challenge here is to distinguish the gestures from the other muscle movements, like tilting arms for the directional control of the quads, and when a gesture is detected, we need to tell what gesture has been performed. This challenge can be viewed as a classification problem, and we approached this problem with neural networks.

the neural network is a biologically inspired function approximator, and here it approximates the unknown function that maps our input, which is the mechanomyogram signals, or the MMG signals for short, to the output gesture classification. It is noted that the non-gesture movements, which we refer to as noises, including tilting arm etc, are also a category, so that the neural network should classify the input signal as noise over 90% of the time.

Convolutional neural networks, or ConvNets for short, is a type of the neural network that is good at extracting local features. We used two convolutional layers in the network. The first convolutional layer would extract the features correspond to the hills, valleys, or rising edges etc. in the MMG plot, and the second convolutional layer would extract more abstract features. These features are converted to a neural activation map, that indicates which features is recognized in what location along the plot. The function that produces the classification output based on the activation map is then approximated with the fully connected neural layers. The armband can be calibrated for each new user. This correspond to training the last two fully connected layers to learn the new users' mapping from the feature activations to the gesture classifications.

The accuracy of the calibrated neural network classification is shown here. It is noted that some gestures have achieved particularly good results, like clenching fist, snapping fingers, and double clenching fist. These gestures will be used for the demonstration tomorrow. 

#### Drone Control

For the control of the actual quadcopters, we are applying a feedback loop on the drone location. A PID controller is applied to guide the drones to the target locations based on their actual location and velocity obtained from the camera function. Each drone is declared in python as a drone class object, allowing us to infer the drone locating for a short period if the camera location feed is slightly unstable. The drone library allows us to input the desired roll and pitch angle directly to the drones, allowing us to simplify the system to a second order system, as the x, y coordinates of the drone is proportional to the pitch and roll angles integrated twice with respect to time respectively. The block diagram of the control loop is shown here.
$$
-A\ddot{x}\approx \phi = K_px+K_d\dot{x}\\
-B\ddot{y}\approx \theta = K_py+K_d\dot{y}\\
$$

#### Program Architecture

Putting things together, the armband sets the target locations for the drones. The gestures would allow us to switch the control modes to control a single drone, two drones as a group, or lock the drones in their current locations. 

However, as each of the parts run at a different frequency, multi-threading is implemented to assemble the parts. The neural network predicts the user gesture 50 times per second in the main python thread, and the camera runs in another thread which loops at 60FPS. Each of the drone runs in its own thread, which calculates the commands 100 times per second. For a demonstration of the results, please come to our booth tomorrow.