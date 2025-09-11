# HotWheels - Motion Control 🕹️ 🚗
## Introduction 📘
The final objective of the project was to develop a motion control algorithm that would allow the JetRacer to drive by itself in the track present in the lab.  
  
While driving around this "city" the robot would have to react to some objects and adjust it's behaviour accordingly, applying decisions to slow down, stop or to continue driving. To achieve that we decided to base the process of decision-making on the output of the previously developed detection models (lane and object) that would correctly identify said objects.

## Development 🛠️
In order to develop the autonomous driving feature we had to find a way to create a path for the robot to follow, leading to our first step: *polynomial fitting*. By using the output binary mask of our previously developed lane detection model we could calculate the optimal trajectory to follow based on the side lanes, extracting the central points of the left and right lanes and finding the average distance between them. When connected, these average points would form the best route for the car.  
  
With a clear path to serve as a guide we proceeded to the next step that was developing the algorithm to adjust the direction and throttle of the robot according to the predicted route. By extracting the angle formed on the guideline we were able to adjust the steering of the wheels and turn the car when necessary.  
  
Besides that, to control the speed we manipulate the throttle value sent to the motors. If the car is stationary a bigger value is used in order to overcome the inertia, after that a constant smaller throttle is applied. When the current speed is higher than the defined limit the throttle keeps reducing until the speed reaches the desired value.  
  
After achieving a stable autonomous control of the car the final step was to implement the slow down behaviour and the emergency breaking with the ultrasound sensor. This sensor was connected to the arduino and after some code adaptation we were able to send it's values to the main app aswell. By defining a safe distance of 20cm the car would send a stopping signal to the motors if an object was detected inside that safety area. When the object is no longer being detected the robot resumes it's path.  
  
Regarding the slow down behaviour, when some specific traffic signs were detected (crosswalk, danger, yellow light, etc) the car would reduce the current throttle to achieve a safer speed.

## Results 📊
Even though it's definitely not perfect, the team was finally satisfied with the results of the autonomous control, having developed a robot capable of driving through a track without human intervention, maintaining itself inside the driveable lanes and reacting to the environment around it. Below are some videos of the final resutls demonstrating these behaviours.  
  

### 1 - First PID integration 🎮
https://github.com/user-attachments/assets/bcf525d6-47f7-412e-91ff-15908c063fc1


### 2 - Emergency Breaking 🛑
https://github.com/user-attachments/assets/1d08eb9a-09f0-4ef4-bde9-25ccc430146a


### 3 - Full lap on the track 🏎️
https://github.com/user-attachments/assets/d8e4957b-69d1-42ab-bc0c-0162824e71ce


___