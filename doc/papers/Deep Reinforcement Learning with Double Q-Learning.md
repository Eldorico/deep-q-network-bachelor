# Deep Reinforcement Learning with Double Q-Learning

#### Abstract

- DQN (Q-Larning with a neural network) algorithms thends to overestimate action values under certain conditions
- They propose a specific adaptation to the DQN Algorithm and whow that the resulting algorithm not only reduces the overestimations but also leads to much better performance on several games. 

#### Introduction

- Theses overestimations occur when actions values are inaccurate, irrespective of the source of approxiamtion error. (Of course, imprecise values estimates are the norm during learning, which indiactes that overestimations may be much more common thant previously appreciated.)
- Overoptimistic values are not necessariliy a problem since it is a well known exploration technique. 

#### Background

- $Q_\pi(s,a) = [ R_1 + \gamma R_2 + ... | S_0 = s, A_0=a, \pi]$ . Optimal value is then $Q_*(s,a) = max_{\pi}Q_{\pi}(s,a) $ An optimal policy is easyliy derived by selectinf the highest valued action in each state.


#### Deep Q Networks

- A deep Q network (DQN) is a multi-layered neural network that for a givent state $s$, outputs a vector of actions values $Q(s,a, \theta)$ where $\theta$ are the parameters of the network.  For an $n$-dimensional state space and an action space containing $m$ actions, the neural network is a function from $\mathbf{R}^n$  to $\mathbf{R} ^m$ .
- Experience Replay
- Target Nework. 
- Target used by DQN is then $Y_t^{DQN} \equiv R_{t+1} + \gamma max_aQ(S_{t+1}, a; \theta_t^-) $ 
- Both experience replay and target network dramatically improve the performance. (Mnih et al., 2015)

#### Double Q-learning

- The max operator in standar Q-learning and DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic values estimates.  

- To prevent this, we can decouple the selection from the evaluation. We can rewrite the target $Q$ as: $Y_t^Q= R_{t+1} + \gamma Q(S_t+1,   argmax_aQ(S_t+1, a; \theta_t); \theta_t)$

  ​