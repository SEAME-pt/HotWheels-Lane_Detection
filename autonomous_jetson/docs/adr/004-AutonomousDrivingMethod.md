# Autonomous Driving Method
## Context
Being able to detect lanes correctly is very important in the process of developing a self driving car. However, the real challenge is the decision making that comes after those detections. There are many different ways to achieve an autonomous driving model, one of them being reinforcement learning.  
  
In a first instance the team thought that this method would be the way to go. However this approach relies only in reacting to the environment instead of thinking ahead of time which is a better way of thinking. Besides that the training process requires hundres of thousands of trial and error runs which consumes a lot of time and resources that we didn't have.  
  
In most cases reinforcement learning is a good choice but due to our limited timelines we opted for searching different approaches that would require less time and resources.
  
## Decision
Considering all those variables we decided to switch to a modular method. This approach combines our already developed detection models with a planned/predictive approach, which was actually better and much faster to implement, by using Polynomial Fitting to plan a route based on geometry extraction from the input frame and rely on MPC to follow this route.  
  
By doing this we'd be able to predict future states and optimize the trajectory while considering aspects like: lane curvature, obstacle positions and vehicle dynamics. MPC would receive the path and penalize proximity to obstacles, deviation from lane center and sharp steering or acceleration, outputting optimal steering and throttle over a short time period. Constraints like "don't collide with bounding boxes" and "stay inside the lane boundaries" would also be defined.  
  
*Research: [Reinforcement Learning](https://github.com/SEAME-pt/HotWheels-Lane_Detection/blob/main/docs/research/007-RL.md) and [Motion Control Alternatives](https://github.com/SEAME-pt/HotWheels-Lane_Detection/blob/main/docs/research/010-MotionControlAlternatives.md)*

___