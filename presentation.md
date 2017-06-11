## Presentation Script

#### Gesture Recognition

To achieve gesture control of the drones, we need to interpret the inputs from the users with the armband. The main challenge here is to distinguish the gestures from the other muscle movements, like tilting arms for the directional control of the quads, and when a gesture is detected, we need to tell what gesture has been performed. This challenge can be viewed as a classification problem, and we approached this problem with neural networks.

the neural network is a biologically inspired function approximator, and here it approximates the unknown function that maps our input, which is the mechanomyogram signals, or the MMG signals for short, to the output gesture classification. It is noted that the non-gesture movements, which we refer to as noises, including tilting arm etc, are also a category, so that the neural network should classify the input signal as noise over 90% of the time.

Convolutional neural networks, or ConvNets for short, is a type of the neural network that is good at extracting local features. We used two convolutional layers in the network. The first convolutional layer would extract the features correspond to the hills, valleys, or rising edges etc. in the MMG plot, and the second convolutional layer would extract more abstract features. These features are converted to a neural activation map, that indicates which features is recognized in what location along the plot. The function that produces the classification output based on the activation map is then approximated with the fully connected neural layers. The armband can be calibrated for each new user. This correspond to training the last two fully connected layers to learn the new users' mapping from the feature activations to the gesture classifications.

The accuracy of the calibrated neural network classification is shown here. It is noted that some gestures have achieved particularly good results, like clenching fist, snapping fingers, and double clenching fist. These gestures will be used for the demonstration tomorrow. 

#### Program Architecture

The drone control receives the target locations from the armband. Various gestures allows us to switch between controlling one drone, two drones as a group, or lock the drones in their current locations. 

Some tools we used in the program assembly have greatly improved the performance. Firstly, as different parts of  our program runs at different frequency, multi-threading has allowed us to run several loops simultaneously, so that the frequency that the drone receives command is not limited by the other loops.

The C# application form is used to handle communication from the IMU. To pass data between the C# application form and our main script written in python, a named pipe server is created in C#. Through the pipe, C# sends the MMG signals, and receives the gesture classification.

Each actual drone is declared as a drone class object. This has simplified the communication with the drones and allowed us to declare location and velocity as class variables, so that the drone location read from the webcams can be smoothed.