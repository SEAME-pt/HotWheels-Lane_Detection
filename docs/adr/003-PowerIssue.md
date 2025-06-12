# Power Issue
## Context
Since the beginning of the project that we struggled with power spikes that would happen while using the servo motor. These spikes caused the board to shutdown every time and were compromising our progress significantly. For a lot of time we believed that it was a software issue and would just avoid using the servo motor at all. 
  
## Solution
After really digging deep in the problem we realised it was a deficiency in the expansion board (component that feeds the energy from the batteries to the Jetson Nano) and switched for a new one. After testing with the new component no problems were reported so we thought that we had solved the issue.  
  
However, after 1 week or so, it happened again. During a driving test the board shutdown after changing directions (action performed by the servo motor). Having in mind that the component was new and had been working well so far we concluded that it couldn't be another hardware deficiency. Being so we opted for an alternative plan and implemented a tension regulator in our circuit to better distribute the energy throughout the system.  
  
It burned. Not because the solution was bad or there were issues with the equipment. It burned due to human error and poorly made connections. Unfortunately we did not order a second piece in case the first one had issues (rookie mistake) and really needed to implement a solution fast. Being so we decided to divide the energy consumption in our system. The main components would be fed by the expansion board while the extra ones would use an external source (powerbank).  
  
So far this solution seems to have worked and we can now control the car and use all the hardware with no apparent issues.
  
## Solution Update 1 - Drained powerbank
We came back to the lab after a week of vacation and the car was not working. We immediatly checked the powerbank and realised it was out of power. No big deal, just charge it right? If only it was that simple... The powerbank gave no signs of life. Nothing happened when we connected it to the charger or pressed the power button.  
  
It turns out that when we left the lab the week before we left everything on and plugged which caused the drainage of the powerbank. Even after a wild attempt of short circuting it we were not able to resurrect it. Having ruined yet another solution we had to come up with another plan. We decided to make use of some batteries that we had and use them to play the same role as the powerbank. However, batteries are not as easy to recharge and we required a special hardware component to do so. After acquiring that component, welding the batteries together and assembling everything in the car this solution worked.  
  
Hopefully this will be our permanent solution but we already have a backup plan in mind in case we have troubles again.
  
## Solution Update 2 - Not enough power
Surprise, surprise, we had issues again. This time the problem was a bit simpler though, the only issue was that the arduino was not receiving enough energy (it needs 5V and was only getting 3V) so we changed the wires in a way that the board would connect directly to it. Below is a simple diagram to demonstrate how the energy distribution is being made right now.  
  
![energy-distribution](https://github.com/user-attachments/assets/54c33530-8afa-43f3-921d-13ae52e7688f)

___