# CARLA simulator
## Introduction
CARLA is an open-source simulator for autonomous driving research, developed from the ground up to support development, training and validation of autonomous urban driving systems. It is grounded on Unreal Engine to run the simulation and uses the ASAM OpenDRIVE standard to define roads and urban settings. Control over the simulation is granted through an API handled in Python and C++ that is constantly growing as the project does.  
  
## The simulator
It consists of a scalable client-server architecture. The server is responsible for everything related with the simulation itself: sensor rendering, computation of physics, updates on the world-state and its actors and much more.  
  
## Capabilities of CARLA

- ***Traffic manager***  
A built-in system that takes control of the vehicles besides the one used for learning. It acts as a conductor provided by CARLA to recreate urban-like environments with realistic behaviours.  
    
- ***Sensors***  
Vehicles rely on them to dispense information of their surroundings. In CARLA they are a specific kind of actor attached the vehicle and the data they receive can be retrieved and stored to ease the process. Currently the project supports different types of these, from cameras to radars, lidar and many more.  
  
- ***Recorder***  
This feature is used to reenact a simulation step by step for every actor in the world. It grants access to any moment in the timeline anywhere in the world, making for a great tracing tool.  
  
- ***ROS bridge and Autoware implementation***  
As a matter of universalization, the CARLA project ties knots and works for the integration of the simulator within other learning environments.  
  
- ***Open assets***  
CARLA facilitates different maps for urban settings with control over weather conditions and a blueprint library with a wide set of actors to be used. However, these elements can be customized and new can be generated following simple guidelines.  
  
- ***Scenario runner***  
In order to ease the learning process for vehicles, CARLA provides a series of routes describing different situations to iterate on. These also set the basis for the CARLA challenge, open for everybody to test their solutions and make it to the leaderboard.  

___