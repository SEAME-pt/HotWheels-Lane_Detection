# Autonomous Driving Method

## Context
Being able to detect lanes correctly is very important in the process of developing a self driving car. However, the real challenge is the decision making that comes after those detections. In order to implement an autonomous driving algorithm with reinforcement learning, the team had to do a small research to determine the best approach to use in our project. To achieve this we gathered the 4 most used methods and created a pros/cons list to help us decide.

### Results
#### 1. Proximal Policy Optimization (PPO)

✅ PROS:  
- **Stable and robust:** Works well in real-world robotics and autonomous driving.  
- **Policy-based:** Handles continuous action spaces smoothly.  
- **Efficient:** Uses a clipped objective function to prevent large policy updates, leading to stable learning.  
- **Parallelizable:** Works well in distributed training environments.  
  
❌ CONS:  
- **Less sample efficient:** Requires more interactions with the environment compared to off-policy methods.  
- **Still needs hyperparameter tuning:** Finding the right clipping parameter is crucial for stability.  
- Can be slower to converge than TD3 or SAC.
  
#### 2. Soft Actor-Critic (SAC)

✅ PROS:  
- **Explores more effectively:** Uses entropy regularization, making it good at balancing exploration and exploitation.  
- **Stable and robust learning:** Works well for continuous action spaces.  
- **Off-policy:** More sample-efficient than PPO.  
- Performs well in complex, high-dimensional tasks.  
  
❌ CONS:  
- **Computationally expensive:** Requires learning two Q-functions and an actor.  
- More complex implementation than PPO or TD3.  
- Entropy tuning can be tricky, affecting performance.  


#### 3. Twin Delayed Deep Deterministic Policy Gradient (TD3)

✅ PROS:  
- **Reduces Q-function overestimation:** Uses two Q-networks to improve accuracy.  
- **Better stability than DDPG:** By delaying policy updates, it prevents excessive variance.  
- **Efficient in sample usage:** Unlike PPO, it is off-policy, making training faster.  
- Works well in continuous action spaces.  
  
❌ CONS:  
- **Hard to tune hyperparameters:** Learning rate, target policy smoothing, and delay factors need fine-tuning.  
- **Still deterministic:** Unlike SAC, it doesn't naturally encourage exploration.  
- More complex than PPO.  

#### 4. Deep Q-Network (DQN)

✅ PROS:  
- **Simple to implement:** Works well with discrete action spaces.  
- **Sample efficient:** Uses experience replay and target networks.  
- Good for learning basic driving behaviors.  
- Less computationally expensive compared to PPO, SAC, or TD3.  
  
❌ CONS:  
- **Only works with discrete actions:** Not suitable for smooth control in autonomous driving.  
- **Struggles with continuous environments:** Needs discretization, which can lead to suboptimal behavior.  
- **Prone to overestimation bias:** Can make learning unstable.  
- Less effective for high-dimensional tasks.  
  
## Decision
After taking all of the above aspects into consideration we decided that the ***Proximal Policy Optimization (PPO)*** would be our way to go regarding the method to develop our autonomous driving algorithm. Because our objective is to use camera frames as input to the model we concluded that this approach was the one that would perform better once fully developed due to it's stability and adaptability to different environments.
  
## Consequences
We hope that this decision simplifies the development of the model and helps us reach our expected goals.
