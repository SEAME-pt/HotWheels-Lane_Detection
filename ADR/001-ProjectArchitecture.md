# Project Architecture

## Context
Having finished the Cluster Instrument model, we had to start thinking about the development of the next modules that require a lot more knowledge, processing power and software architecture. After some research we came to the conclusion that the usage and implementation of AI algorithms would be very positive for us and from there we just had to choose what methods served our purposes better. 
  
## Decision
For the modules related to image processing (Lane Detection, Object Detection and Road Segmentation) we chose to apply the method of ***supervised learning*** that relies on a labeled dataset to train its model. This method finds patterns between the input image and the expected output label of it by using parameters which are adjusted based on a loss calculation function that determines how far the model's prediction was from the real value.  
  
On the other hand when it comes to controlling the car, this method is not the most appropriate one because the expected output is not clear, structured or well-defined. Being so we needed another method for controll operations (steering, throttle and braking) and opted for the ***reinforcement learning***. The logic used by this method is based on a reward system where there's an agent capable of doing actions and an environment that responds to these actions with a feedback. The algorithm learns which actions to do or not based on the positive or negative feedback.  
  
Each module requires the usage and implementation of a new dataset and algorithm. Even though we only did a small research on that topic these are the options that we considered so far:  
  
**Lane detection**: SCNN(algorithm) and CULane(dataset)  
  
**Object detection**: YOLOv5(algorithm) and COCO(dataset)  
  
**Road segmentation**: DeepLabV3(algorithm) and Cityscapes(dataset)  
  
## Consequences
We believe that this strucuture is very similar to the ones currently used in the automotive industry and will best mimic a real-life application.  
It also allows for a dynamic solution and greater adaptability for different environments and scenarios. 