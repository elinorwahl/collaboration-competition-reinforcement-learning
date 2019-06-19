# Reinforcement Learning Project: Collaboration and Competition

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

## Introduction

In this project, we teach two reinforcement learning agents to skillfully play a game against each other in the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. In this environment, two agents control rackets to bounce a ball over a net. If an agent succeeds in hitting the ball over the net, it receives a reward of +0.1; if it lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  The goal of each agent is to keep the ball in play.

![Trained Agent][image1]

The observation space in this environment consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must reach an average score of +0.5 over 100 consecutive episodes. This average score is calculated by adding up the rewards that each agent received (without discounting), to get a score for each agent. We then take the maximum of these 2 scores to get the overall score for each episode.

## Methods

The algorithm being used for this task is a deep deterministic policy gradients (DDPG) algorithm, which is explained [here]([DDPG](https://arxiv.org/pdf/1509.02971.pdf). This algorithm makes use of an Actor network, which estimates the best action for a given state, and a Critic network, which uses the actor's output to estimate the value of an action in a given state. At each time step, the critic's output is the used to update the actor's weights, and the critic is updated with the gradients from the temporal-difference error signal.

However, an important distinction in this case is that two agents are in play, and both are competing with each other while attempting to maximize their performance. This means that there must be some combination of shared and independent learning, to enable them to effectively play against each other.

Therefore, in `maddpg_agent.py`:

- The `Agent.OUNoise()` method, which adds Ornstein-Uhlenbeck noise to the action selection process, takes the number of agents as well as the action size as input.

```
self.noise = OUNoise((num_agents, action_size), random_seed)
```

- The `Agent.step()` method records the experiences and rewards of each of the agents in replay memory separately.

```
for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
```

- The `Agent.act()` method uses a separate action selection process for each agent.

```
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for agent_num, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_num, :] = action
        self.actor_local.train()
        if add_noise:
            actions += self.epsilon * self.noise.sample()
        return np.clip(actions, -1, 1)
```

- This method also contains a very important element of my implementation: the addition of an epsilon value to gradually decrease the the Ornstein-Uhlenbeck noise sampling, and encourage both agents to select more and more actions likely to yield high expected rewards as they learn. The epsilon value is gradually decayed in the `Agent.reset()` method.

```
    def reset(self):
        self.epsilon = max(0.01, self.epsilon - self.epsilon_decay) # decay noise
        self.noise.reset()
```

- In the `Agent.learn()` method, gradient clipping is added to the critic network's loss function, which bounds the upper limits of the gradients close to 1, and prevents the the weight update gradients from exploding.

```
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
```

- In the `Agent.OUNoise.sample()` method, the `np.random.standard_normal()` function is used, to make the Ornstein-Uhlenbeck noise process follow a Gaussian distribution instead of a truly random one.

```
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
```

And in `model.py`, both the Actor and the Critic models have batch normalization applied to their first hidden layers. This requires decompressing the state input from a 1D tensor.

```
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
```

These are the chosen hyperparameters of the agents:

```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-2              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
EPSILON = 1.0           # noise decay
```

The final network architecture consists of an Actor and a Critic network each with two hidden layers of size `256, 128` and batch normalization in their first hidden layers; a `tanh` output activation function for the Actor, and a `relu` output activation function for the Critic.

These are the results of training the two agents:

![Graph of MADDPG agent performance](/images/maddpg-results.png)

Impressively, the two agents managed to reach an average score of +0.5 points within 227 training episodes, and to steadily improve their performance throughout the training process.

## Further Study

I would like to experiment with altering the code to make each of the two agents perform the learning step separately, as well as selecting actions and recording rewards separately. Raising the batch size, tau and learning rates of the agents had a huge impact on performance, and there might be even better values to try.

Raising the target score or allowing the training to continue for longer may provide a better sense of how well the agents can maintain strong performance over time.

Finally, it would be interesting to apply [proximal policy optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) to this same task.

## Usage 

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file.