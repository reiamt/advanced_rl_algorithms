## Reinforcement Learning Course: Intro to Advanced Actor Critic Methods

### 1. Actor Critic (AC)

the actor-critic method describes a method to approximate the value function to then derive an optimal policy
the ac method consists of two DNNs:
    1. used to approximate the agent's policy directly (actor)
    2. used to approximate the value function (critic)
both DNNs work together: the actor selects actions and the critic evaluates the states and the result is compared to the rewards from the environment
belongs to Temporal Difference learning (since learning happpens in episodes)

algo overview:
- init actor critic network
- repeat for large number of episodes:
- reset the environment, score, terminal flag
- while state is not terminal:
    - select action according to actor network
    - take action and receive reward and new state
- plot scores over time for evidence of learning

### 2. Deep Deterministic Policy Gradients (DDPG)

- type of actor-critic method (two DNNs)
- critic evaluates state and action pairs
- actor decides what to do based on current state (or any other we pass through)
    - network outputs action values, no probabilities here
    - noise for explore-exploit dilemma

### 3. Twin Delayed Deep Deterministic Policy Gradients (TD3) 
extension of 2. DDPG

- in Q learning, there is bias from the max over actions
- no max ni update rule for AC methods, so whats responsible for it?
- overestimation (state is higher evaluated than it actually is thus resulting in a suboptimal solution) also comes from approximation errors
- neural networks are a source of approximation error
- AC methods are bootstrpped so errors accumulate

- double Q learning uses two Q networks
- can use clipped double Q which underestimates
- delay policy updates to give Q time to converge
- use target networks for actor and (both) critics (6 in total)
- use soft updates to the target network

