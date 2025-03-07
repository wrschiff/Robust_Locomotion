import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
import scipy
import stable_baselines3.ppo

############ DISCOUNTED SUM IMPLEMENTATION ###############
def discounted_sum(discount_factor, arr):
   """
   CREDIT: OpenAI spinning up implementation of PPO. Calculates cumalative sum:
      \ sum { \gamma^{t} * R_t }
   Don't know how it works. Can be found here: 
      https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
   We use it to calculate expected return, and GAE Advantage
   """
   return scipy.signal.lfilter([1], [1, float(-discount_factor)], arr[::-1], axis=0)[::-1]


###################### POLICY CLASS ######################

class Policy(nn.Module):
   """
   A stochastic policy parameterized by a neural network. Outputs the mean of
   a gaussian for each action, with the STD initialized to e^(-0.5)
   """
   def __init__(self, num_features, num_actions):
      super().__init__()
      self.log_std = torch.nn.Parameter(-0.5*torch.ones(num_actions))
      self.layers = nn.Sequential(
         nn.Linear(in_features=num_features, out_features=64),
         nn.Tanh(),
         nn.Linear(in_features=64, out_features=64),
         nn.Tanh(),
         nn.Linear(in_features=64, out_features=num_actions)
      )
   
   def _get_distribution(self, x):
      """
      Batch of states are passed into network. mus: [batch_size x num_actions]
      Normals: [batch_size]
      """
      mus = self.layers(x)
      return Normal(mus, torch.exp(self.log_std))
   
   def _get_log_probs(self, distr: Normal, actions) -> torch.Tensor:
      """
      Returns the log probability of some action being chosen.
      Note that exp( log(pi'(a)) - log(pi(a)) ) = pi'(a) / pi(a), but is more numerically stable.
      Also: log( pi(a_1) * pi(a_2) * ... * pi(a_n) ) = log(pi(a_1)) + log(pi(a_2)) + ... for N dimensions
      of action vector.
      """
      return distr.log_prob(actions).sum(axis=-1)
   
###################### VALUE FUNCTION CLASS ######################   

class Critic(nn.Module):
   def __init__(self, num_features):
      super().__init__()
      self.layers = nn.Sequential(
         nn.Linear(in_features=num_features, out_features=64),
         nn.Tanh(),
         nn.Linear(in_features=64, out_features=64),
         nn.Tanh(),
         nn.Linear(in_features=64, out_features=1)
      )
   
   def forward(self, batch):
      # values will be [batch_size x 1] -> one prediction for each batch
      values = self.layers(batch)
      # HOWEVER, when scaling the probabilities, we just want a tensor of [batch_size].
      # This ensures scalars are multiplied out to each example in batch.
      return values.squeeze(-1)


###################### REPLAY BUFFER ######################
class TrajectoryBuffer():
   def __init__(self, state_space, action_space, size=4000):
      """
      For every trajectory, in order to compute actor loss, we need the following:
         :param states = to calculate current distribution over states
         :param actions = to calculate current log prob of action over distribution
         :param prev_logprobs = constant denom in ratio term
         :param values = value estimation of states, used to calculate advantage
         :param advantages = previously calculated GAE advantages as constant scalar to gradient
         :param rewards = tracking reward to estimate returns
         :param returns = separate array to calculate returns, train value estimator on this

      To keep track of trajectories, we have:
         :param trajectory_start = index of start of trajectory in buffer. Trajectories are
         contiguous in the array
         :param idx = index in buffer for current step in trajectory.
         :param epoch_size = num timesteps after which we & perform update
      
      For GAE Advantage and discounted return, we have:
         :param lam = factor in cumlative sum for GAE
         :param gamma = discount for future returns
      """
      if (np.isscalar(state_space)):
         # Pass the shape as a tuple. Note that if we have to get nparray shape or tensor shape, also returns a tuple
         self.states = np.zeros((size, state_space), dtype=np.float32)
      else:
         # If state space is an iterable (dim1, dim2, ...), this will unpack it.
         self.states = np.zeros((size, *state_space), dtype=np.float32)
      if (np.isscalar(action_space)):
         self.actions = np.zeros((size, action_space), dtype=np.float32) 
      else:
         self.actions = np.zeros((size, *action_space), dtype=np.float32)

      self.prev_logprobs    = np.zeros(size, dtype=np.float32)
      self.vals             = np.zeros(size, dtype=np.float32)
      self.advantage        = np.zeros(size, dtype=np.float32)
      self.rewards          = np.zeros(size, dtype=np.float32)
      self.returns          = np.zeros(size, dtype=np.float32)

      self.trajectory_start, self.idx = 0, 0
      self.epoch_size = size
      self.lam = 0.95
      self.gamma = 0.99

   def clear(self):
      """
      Clears the buffer, so we can start building trajectories without worrying about
      complicating indices.
      """
      self.states           = np.zeros_like(self.states)
      self.actions          = np.zeros_like(self.actions)
      self.prev_logprobs    = np.zeros_like(self.prev_logprobs)
      self.vals             = np.zeros_like(self.vals)
      self.advantage        = np.zeros_like(self.advantage)
      self.rewards          = np.zeros_like(self.rewards)
      self.returns          = np.zeros_like(self.returns)

   def add_experience(self, s, a: torch.Tensor, prev_prob, vals: torch.Tensor, reward):
      """
      At each time step, we add the experience. Value of an action is estimated at this point.
      GAE Advantage will be calculated later using these values, and will scale the gradient.
      """
      self.states[self.idx]        = s
      self.actions[self.idx]       = a
      self.prev_logprobs[self.idx] = prev_prob
      self.vals[self.idx]          = vals
      self.rewards[self.idx]       = reward
      self.idx += 1
      # Note: we do NOT add returns or advantages, which are calculated at the end of trajectory

   def calculate_advantages(self, final_V=0):
      """
      At the end of a trajectory, go back and compute GAE estimated advantages (for actor loss)
      AND true discounted return (for critic loss)
      """
      # Note: currently NOT using final_V, as I assume all trajectories end from terminal state
      sliced = slice(self.trajectory_start, self.idx)
      #print(f"TRAJECTORY START: {self.trajectory_start}")
      ##### CALCULATE GAE ######
      # (r_1, r_2, ... r_h)
      trajectory_rewards = np.append(self.rewards[sliced], final_V)
      # (V_1, V_2, ... V_h)
      values = np.append(self.vals[sliced], final_V)
      #print(f"TRAJECTORY REWARDS:\n{trajectory_rewards}\n\nVALUES:\n{values}")
      # r_t + gamma * V_{t+1} - V_t
      deltas = trajectory_rewards[:-1] + self.gamma * values[1:] - values[:-1]
      #print(f"DELTAS:\n{deltas}")
      self.advantage[sliced] = discounted_sum(self.gamma * self.lam, deltas)
      #print(f"ADVANTAGES: {self.advantage}")

      ##### CALCULATE DISCOUNTED RETURN AT EACH TIME STEP ########
      trajectory_returns = discounted_sum(self.gamma, trajectory_rewards)[:-1]
      # We don't keep the last one?
      self.returns[sliced] = trajectory_returns

      self.trajectory_start = self.idx
   
   def get_trajectories(self):      
      """
      Once we fill up the entire buffer, return it all as one big batch to do training.
      Buffer will reset via zeroing indices -- which will overwrite previous values.
      """
      assert self.idx == self.epoch_size
      self.idx, self.trajectory_start = 0, 0
      # Normalize the advantage values.
      adv_mean = self.advantage.mean()
      adv_std = self.advantage.std()
      self.advantage = (self.advantage - adv_mean) / (adv_std + 1e-8)
      return dict(states        = torch.as_tensor(self.states, dtype=torch.float32), 
                  acts          = torch.as_tensor(self.actions, dtype=torch.float32), 
                  prev_logprobs = torch.as_tensor(self.prev_logprobs, dtype=torch.float32),
                  advs          = torch.as_tensor(self.advantage, dtype=torch.float32), 
                  rets          = torch.as_tensor(self.returns, dtype=torch.float32)
               )

      
   

###################### PPO CLASS ######################

class ActorCritic():
   def __init__(self, state_dim, action_dim, buffer_size):
      # Networks
      self.pi = Policy(state_dim, action_dim)
      self.v = Critic(state_dim)
      # Buffer
      self.buffer = TrajectoryBuffer(state_dim, action_dim, size=buffer_size)
      self.pi_losses = []
      self.v_losses = []
      # Hyperparameters
      self.ratio_clip = 0.2
      self.pi_optim = optim.Adam(self.pi.parameters(), lr=2e-4)
      self.critic_opt = optim.Adam(self.v.parameters(), lr=5e-4)
      self.pi_gradsteps = 60
      self.v_gradsteps = 60

   def step(self, states):
      """
      We don't need any gradients when we perform a step. Trajectories currently collected
      are done under the "old" policy and probs are used simply as the denom in the ratio term.
      """
      with torch.no_grad():
         # Get pi distribution over state(s)
         distrs = self.pi._get_distribution(states)
         # Sample action(s) from pi
         actions = distrs.sample()
         # Record log probs for later (training pi network)
         log_probs = self.pi._get_log_probs(distrs, actions)
         # Current estimate of value function in order to train it
         values = self.v(states)
      return actions.numpy(), log_probs.numpy(), values.numpy()


   def compute_actor_loss(self, data: dict) -> torch.Tensor:
      """
      Assume data is a dict, with all states, actions, rewards, in a trajectory.
      Goal is to calculate gradient step and take gradient step for one trajectory.
      """
      # Pull out states, actions, advantage values from a trajectory
      states, actions, advantage, old_probs = data['states'], data['acts'], data['advs'], data['prev_logprobs']

      # 1. Calculate r = [ pi'(a|s) ] / [pi(a|s)]
      distrs = self.pi._get_distribution(states)
      #print(distrs.batch_shape, actions.shape)
      primed_log_probs = self.pi._get_log_probs(distrs, actions)
      #print(f'ACTIONS: {actions}\n\n')
      #print(f'CURRENT LOG PROBS OF GIVEN ACTION (ones): {primed_log_probs}\n\n')
      #print(f'OLD LOG PROBS OF GIVEN ACTION (ones) - should be the same under same policy\n{old_probs}\n\n')
      # exp( log(pi'(a|s)) - log(pi(a|s)) )
      ratio = torch.exp(primed_log_probs - old_probs)
      #print(f"RATIO: {ratio}")
      # 2. rclip(r) * A
      clipped_ratio = torch.clamp(ratio, 1-self.ratio_clip, 1+self.ratio_clip)
      # 3. If the advantage is negative, we don't clip. 
      # Note that we perform gradient ascent (but actually we just take negative and do gradient descent)
      surrogate_loss = -torch.min(clipped_ratio*advantage, ratio*advantage)
      return surrogate_loss.mean()
      


   def compute_critic_loss(self, data: dict) -> torch.Tensor:
      """
      Critic Loss is very simple: (V(s) - returns(s))^2.
      Critic tries to exactly predict the returns for the rest of the 
      """
      states, returns = data['states'], data['rets']
      loss = torch.pow((self.v(states) - returns), 2).mean()
      return loss

   def act(self, state):
      """
      Calls ActorCritic.step(), and just throws away the log probs and values.
      Used for inference.
      """
      return self.step(torch.as_tensor(state, dtype=torch.float32))[0]


   def gradient_step(self):
      trajectories = self.buffer.get_trajectories()
      # For tracking purposes: what was the actor/critic loss to begin with?
      self.pi_losses.append(self.compute_actor_loss(trajectories))
      self.v_losses.append(self.compute_critic_loss(trajectories))

      # GRADIENT STEPS ON CRITIC
      for _ in range(self.v_gradsteps):
         self.critic_opt.zero_grad()
         critic_loss = self.compute_critic_loss(trajectories)
         critic_loss.backward()
         self.critic_opt.step()

      #GRADIENT STEP ON ACTOR
      for _ in range(self.pi_gradsteps):
         self.pi_optim.zero_grad()
         actor_loss = self.compute_actor_loss(trajectories)
         actor_loss.backward()
         self.pi_optim.step()
   
   def train(self, env, epochs, steps_per_epoch):
      for i in range(epochs):
         state, _ = env.reset()
         trajectory_rewards = 0
         for _ in range(steps_per_epoch):
            action, log_prob, value = self.step(torch.as_tensor(state, dtype=torch.float32))
            state_p, reward, done, _,  _ = env.step(action)
            self.buffer.add_experience(state, action, log_prob, value, reward)
            trajectory_rewards += reward

            if done:
               self.buffer.calculate_advantages(final_V=0)
               print(f"trajectory reward: {trajectory_rewards}")
               trajectory_rewards = 0
               state, _ = env.reset()

         # When we exit an epoch -- either we perfectly ended a trajectory (unlikely) or
         # we were in the middle. If so, need to calculate the advantages.
         if not done:
            final_state_v = self.v(torch.as_tensor(state_p, dtype=torch.float32)).detach()
            self.buffer.calculate_advantages(final_state_v)
         
         self.gradient_step()
         print(f"ACTOR LOSSES FOR EPOCH {i}: {self.pi_losses[-1]}\n")
         print(f"CRITIC LOSSES FOR EPOCH {i}: {self.v_losses[-1]}\n\n")






# ppo = ActorCritic(10,10)
# act, log_prob, val = ppo.step(torch.rand(10,10))
# buffer_stuff = []
# for _ in range(5):
#    for _ in range(10):
#       with torch.no_grad():
#          state = np.ones(10)
#          action = torch.ones(10)
#          distr = ppo.pi._get_distribution(torch.as_tensor(state, dtype=torch.float32))
#          prev_probs = ppo.pi._get_log_probs(distr, action)
#          adv = ppo.v(torch.as_tensor(state, dtype=torch.float32))
#          ppo.buffer.add_experience(state,action,prev_probs,adv,1)

#    ppo.buffer.calculate_advantages()
#    current_trajectores = ppo.buffer.get_trajectories()
#    critic_loss = ppo.compute_critic_loss(current_trajectores)
#    actor_loss = ppo.compute_actor_loss(current_trajectores)
#    print(actor_loss)





# #### TEST 2 ####
# test_data = torch.rand(64, 10)
# check = Policy(10, 10)
# check2 = Critic(10)
# distributions = check._get_distribution(test_data)
# actions = distributions.sample()
# print(actions)








##################### RANDOM STUFF ############################
# checks = torch.rand(3)
# others = torch.rand(3)
# checks.requires_grad = True
# others.requires_grad = True
# bless = others * checks
# #bless.requires_grad = True
# num = bless[0] + bless[1] + bless[2]
# #num.requires_grad = True
# num.backward()
# print(checks, others)
# print(others._grad, checks._grad)
# #bless.backward()
# #print(others._grad, checks._grad)

