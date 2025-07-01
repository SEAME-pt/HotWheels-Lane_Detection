# Middleware change
## Context
The current middleware that we're using (ZeroC) is functional but it proved to be inefficient due to its design, which required a thread to poll for updates every 100 ms. This approach risked overloading the CPU with frequent requests, especially under high demand. Considering that we still have to integrate the lane and the object detection model in the car we really needed to change our middleware approach to ensure a fast and low latency communication between apps.
  
We decided to try a ZeroMQ implementation and the results were very satisfactory. 
  
## Decision
Unlike ZeroC, ZeroMQ uses a publish/subscribe model that allows the publisher to broadcast updated values to all subscribers in real time, eliminating the need for constant polling, working with minimal latency when propagating value changes. Additionally, implementing this design in the system was more straightforward compared to ZeroC, which required a more complex and less efficient setup.  
  
Being so, from now on, we will implement ZeroMQ in all the apps (car controls and car cluster) to guarantee an optimal communication network.
  
  
*Research: [ZeroC](https://github.com/SEAME-pt/HotWheels-Lane_Detection/blob/main/docs/research/004-ZeroC.md) and [ZeroMQ](https://github.com/SEAME-pt/HotWheels-Lane_Detection/blob/main/docs/research/005-ZeroMQ.md)*

___