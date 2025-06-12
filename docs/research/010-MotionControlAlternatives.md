# Motion control alternatives
## Introduction
Since we already have the lane detection and the object detection models, the recommended approach would be a modular implementation that combines: perception (lane/object detection), planning (path planning) and control (PID, MCP).  
  
This module has a more complex integration but it's easier to test and control since every sub-module can be improved separately.

## Perception
The perception part is mostly complete, there are just some final adjustments to be made to finish this chapter. Our lane/object detections models are already integrated in the car and working as expected.  
  
*For more info: [Lane Detection](https://github.com/SEAME-pt/HotWheels-Lane_Detection/blob/main/Lane-Detection.md) and [Object Detection](https://github.com/SEAME-pt/HotWheels-Lane_Detection/blob/main/Object-Detection.md)*

## Path planning
In order to implement a path planning approach we first needed to extract useful information from the lane detection mask to estimate the vehicle's lateral position relative to the lane (for example wether it's centered or drifting towards one side). There were some interesting methods that achieved this by extracting the lane geometry from the input image: **Contour Detection** and **Polynomial Fitting**.  
  
- **Contour Detection** extracts the outer boundaries of each lane marking as a set of ordered points. This is especially useful for separating left and right lanes when the mask includes both, and it gives us structured access to each lane’s shape. For instance, applying cv::findContours would yield two contours, each representing the shape of a lane marking. These can be further analyzed to extract their endpoints, direction vectors, or centroids.
  
- **Polynomial Fitting** takes the raw lane pixel coordinates or the contour points and fits a smooth curve (e.g., a line or parabola). This is valuable to represent each lane as a continuous mathematical function — for example, fitting a 2nd-degree polynomial to the left and right lane pixels would give us a function y = ax² + bx + c for each lane. From this, we can interpolate lane positions at specific image heights, estimate curvature, and derive a clean lane center trajectory.  
  
## Control
To play the control role there are two main options: **PID** or **MPC**. They're very different from eachother and have distinct requirements.

### PID (Proportional Integral Derivative)
PID is a simple and widely used control algorithm that adjusts a system’s output based on error (the difference between a target and current state). It's made up of three components:

**P (Proportional)**: Reacts to the current error. *(Example: "If I'm 0.5m off-center, steer back toward the center proportionally")*

**I (Integral)**: Reacts to the accumulation of past errors. *(Helps eliminate steady-state error (e.g., consistent small drift)*

**D (Derivative)**: Reacts to the rate of change of error. *(Helps smooth the response and prevent overshooting)*

![Image](https://github.com/user-attachments/assets/d8c98c27-0f99-4e80-bcc5-236f98246837)

NOTE: This method is simple to implement but not ideal for constraint handling like obstacles and sharper curves.

### MPC (Model Predictive Control)
MPC is an advanced control strategy that uses a model of the system to predict future behavior and optimize control inputs over a short horizon.  
  
How it works:
1. **Predict** how the system will behave over the next T seconds using a dynamic model.  
2. **Optimize** a cost function (e.g., stay on lane center, smooth control, avoid obstacles) subject to constraints (e.g., steering limits).  
3. **Apply only the first control input**, then repeat the process at the next timestep (receding horizon).

![Image](https://github.com/user-attachments/assets/fb87c05d-1a1e-477f-8d8a-2fcbeeeb27c2)

___