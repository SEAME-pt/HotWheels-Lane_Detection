# HotWheels - Lane Detection 🛣️ ⛙
## Introduction
The first step in the development of an autonomous driving vehicle is to identify the lanes where it needs to circulate. There are many different approaches to achieve this but the most interesting and challenging one was to create our own model from scratch, and that's exactly what we did.  
  
The development of a lane detection model is no easy task and because we lacked the knowledge to jump right into action we had to do a lot of research first. The methods to use, the datasets, the frameworks, the libraries ... all of this required an extensive research in order to form a clear path for the team to follow and execute. (All of this research can be found in this repository in the *docs* folder)
  
### Deep learning
After completing the research stage the team concluded that a **deep learning** approach would be the best way to achieve our goal. In a simple note, deep learning is a subfield of machine learning that uses algorithms inspired by the structure and function of the human brain (neural networks) to learn from large amounts of data. The next step would then be to create and train a neural network model to identify roads/lanes in images provided by the camera in the car.

## Development
### Convolutional Neural Network (CNN)
A CNN is a type of deep learning model, mostly used for image processing and computer vision tasks, that can automatically and adaptively learn spatial hierarchies of features from input images. Unlike traditional neural networks, CNNs use convolutional layers to reduce the number of parameters and preserve spatial relationships in the data.  
  
This process is very complex and is divided into the multiple stages described below.  
  
***1. Convolution layer***  
- Applies filters (kernels) that scan the image.  
- Each filter detects a specific feature (edges, corners, etc).  
- Outputs a feature map (or activation map).  
  
  
***2. Activation function***  
- Usually ReLU (Rectified Linear Unit): replaces negative values with zero.  
- Introduces non-linearity, allowing the network to learn complex patterns.  
  
  
***3. Pooling layer***  
- Reduces the size of the feature map to make the network faster and less sensitive to exact locations.  
- Common method: Max pooling — takes the highest value in a small window (e.g., 2x2)  
  
  
***4. More convolutions and pooling***  
- The network stacks multiple layers to learn higher-level features.  
    -> Early layers: detect edges or textures.  
    -> Deeper layers: recognize parts of objects (like eyes, wheels, etc).  
  
  
***5. Flattening***  
- Converts the 2D feature maps into a 1D vector to pass into the next stage.  
  
  
***6. Fully connected layer***  
- A standard neural network layer that takes the flattened data and produces a final prediction.  
  
  
PS: The step 4 is were most models differ because the number of layers and their configuration varies with the complexity of the input images and the results obtained. The layers in this step are in constant change until the final predictions are satisfactory.  
  
  
### CULane dataset
Even though the final objective was to feed live frames to the model from the car camera, the training/testing process of the model required THOUSANDS of labeled images which led us to use an external dataset for this purpose. Being so we chose CULane because it contains images from multiple scenarios with different complexity, lighting, visual noise, etc, which was perfect to train our model to be accurate in all cases.  
  
This dataset contained +90k labeled images which were used in the training process of our model and some more unlabeled images that were used in the testing/validation part. However, after converting the model to work on the Jetson and testing with live data, the results were not the same. Not even close actually. The model wasn't able to predict anything in a real scenario because a lot of variables were different from the training, such as lighting, camera colors, camera angle, color of the lanes, etc. Being so we were forced to try a different approach and create our custom dataset with images from the lab, the CARLA simulator, the CULane dataset and from other datasets such as TUSimple and CurveLanes.  
  
***Fine tuning*** is a technique in machine learning where a pre-trained model is adapted to a specific task by continuing its training on a smaller, task-specific dataset. Instead of training a model from scratch, fine-tuning leverages the knowledge already learned by the model, requiring less data and computation. It is commonly used in areas like natural language processing and computer vision to improve performance on specialized tasks.
  

## Results
After a very exaustive process of multiple retrainings, layer reconfiguration, data augmentation, camera calibration and fine tunning ... we were finally able to get accurate results that corresponded to our expectations.  
  
![results1](https://github.com/user-attachments/assets/0ca6ed88-e3a3-4544-8313-da59d3a5160f)

![results2](https://github.com/user-attachments/assets/11421ae6-9f0d-4ddd-96be-bf5b473675bb)

![results3](https://github.com/user-attachments/assets/bdc819a9-b8de-4c53-9ec5-7e268bedf647)

___