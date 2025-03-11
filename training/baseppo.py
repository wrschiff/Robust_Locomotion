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

def init_policy_network(mod):
   """
   Takes a torch module. Checks if it is a linear layer (we don't initialize activation functions)\n
   For the last layer, initializes to 100x smaller weights than the other layers.
   Uses `xavier_uniform_` distribution.
   """
   if isinstance(mod, nn.Linear):
      # We added this attribute for the last layer
      if hasattr(mod, "is_policy_output"):
         # 100x smaller than otherewise (1)
         nn.init.xavier_uniform_(mod.weight, 0.01) 
      else:
         nn.init.xavier_uniform_(mod.weight)
      
      if mod.bias is not None:
         nn.init.zeros_(mod.bias)


###################### POLICY CLASS ######################

class Policy(nn.Module):
   """
   A stochastic policy parameterized by a neural network. Outputs the mean of
   a gaussian for each action, with the STD initialized to e^(-0.5)
   """
   def __init__(self, num_features: int, num_actions: int, action_range: int):
      """
         :param num_features = size of observation space / num input features into network\n
         :param num_actions = size of actions space / num output features from network\n
         :param action range = absolute value of continuous action range
      """
      super().__init__()
      self.action_range = action_range
      self.log_std = torch.nn.Parameter(-0.5*torch.ones(num_actions))
      self.layers = nn.Sequential(
         nn.Linear(in_features=num_features, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=num_actions)
      )
      # Define an attribute of the last layer that signifies it as policy output
      # We will initialize this last layer with 100x smaller weights (https://arxiv.org/pdf/2006.05990)
      self.layers[-1].is_policy_output = True
      self.apply(init_policy_network)
   
   def _get_distribution(self, x):
      """
      Batch of states are passed into network. mus: [batch_size x num_actions]
      Normals: [batch_size]
      """
      mus = self.layers(x)
      return Normal(mus, torch.exp(self.log_std))
   
   def _get_log_probs(self, distr: Normal,raw_actions) -> torch.Tensor:
      """
      Returns the log probability of some action being chosen.
      Note that exp( log(pi'(a)) - log(pi(a)) ) = pi'(a) / pi(a), but is more numerically stable.
      Also: log( pi(a_1) * pi(a_2) * ... * pi(a_n) ) = log(pi(a_1)) + log(pi(a_2)) + ... for N dimensions
      of action vector.
      """
      ### NEW: u = tanh(x), where x is the raw action.
      #logP(u) = logP(x) - logtanh'(x)
      log_act = distr.log_prob(raw_actions)
      # Since we want the log prob of the true action, we need to add this correction
      correction = torch.log(self.action_range * (1 - torch.tanh(raw_actions)**2) + 1e-6) # Small constant for numerical stability
      #return (log_act - correction).sum(axis=-1)
      ## OLD VERSION, we know this works:
      return distr.log_prob(raw_actions).sum(axis=-1)
   
###################### VALUE FUNCTION CLASS ######################   

class Critic(nn.Module):
   def __init__(self, num_features):
      super().__init__()
      self.layers = nn.Sequential(
         nn.Linear(in_features=num_features, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=128),
         nn.Tanh(),
         nn.Linear(in_features=128, out_features=1)
      )
   
   def forward(self, batch):
      # values will be [batch_size x 1] -> one prediction for each batch
      values = self.layers(batch)
      # HOWEVER, when scaling the probabilities, we just want a tensor of [batch_size].
      # This ensures scalars are multiplied out to each example in batch.
      return values.squeeze(-1)


################# CUSTOM TRAJECTORY DATASET #################
class TrajectoryDataset():
   def __init__(self, trajectories: dict):
      self.trajectories = trajectories
   
   def __getitem__(self, index):
      experience = dict()
      for key, value in self.trajectories.items():
         experience[key] = value[index]
      return experience
   
   def __len__(self):
      return self.trajectories["advs"].size(0)

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

   def add_experience(self, s, a, prev_prob, vals, reward):
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
      sliced = slice(self.trajectory_start, self.idx)
      #print(f"TRAJECTORY START: {self.trajectory_start}")
      ##### CALCULATE GAE ######
      # (r_1, r_2, ... r_h)
      trajectory_rewards = np.append(self.rewards[sliced], final_V)
      # (V_1, V_2, ... V_h)
      values = np.append(self.vals[sliced], final_V)

      # r_t + gamma * V_{t+1} - V_t
      deltas = trajectory_rewards[:-1] + self.gamma * values[1:] - values[:-1]
      self.advantage[sliced] = discounted_sum(self.gamma * self.lam, deltas)


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
      self.idx = 0
      self.trajectory_start = 0
      # Normalize the advantage values.
      adv_mean = self.advantage.mean()
      adv_std = self.advantage.std()
      self.advantage = (self.advantage - adv_mean) / (adv_std + 1e-8)
      # Normalize the return values -- DOESNT WORK
      # ret_mean = self.returns.mean()
      # ret_std = self.returns.std()
      # self.returns = (self.returns - ret_mean) / (ret_std + 1e-8)
      return dict(states        = torch.as_tensor(self.states, dtype=torch.float32), 
                  acts          = torch.as_tensor(self.actions, dtype=torch.float32), 
                  prev_logprobs = torch.as_tensor(self.prev_logprobs, dtype=torch.float32),
                  advs          = torch.as_tensor(self.advantage, dtype=torch.float32), 
                  rets          = torch.as_tensor(self.returns, dtype=torch.float32)
               )

   def get_trajectories_as_DataLoader(self, batch_size):
      """
      The purpose of this function is to do mini-batch stochastic gradient descent (SGD)
      instead of full-batch GD. Dictionary of trajectories is wrapped in custom dataset class.
      That is then wrapped in `torch.utils.data.DataLoader`. Allows variable batch sizes and shuffling.
      
         :param batch_size = mini batch size.
      """
      full_buffer_trajs = TrajectoryDataset(self.get_trajectories())
      return torch.utils.data.DataLoader(full_buffer_trajs, batch_size=batch_size, shuffle=True)

   

###################### PPO CLASS ######################

class ActorCritic():
   def __init__(self, state_dim, action_dim, buffer_size, action_range, verbose=False):
      # Networks
      self.pi = Policy(state_dim, action_dim, action_range)
      self.v = Critic(state_dim)
      # Buffer
      self.buffer = TrajectoryBuffer(state_dim, action_dim, size=buffer_size)
      self.buffer_size = buffer_size
      # Tracking
      self.pi_losses = []
      self.v_losses = []
      self.reward_tracking = []
      self.kl_divergence = []
      # Hyperparameters
      self.ratio_clip = 0.2
      self.pi_optim = optim.Adam(self.pi.parameters(), lr=1e-4)
      self.critic_opt = optim.Adam(self.v.parameters(), lr=3e-4)
      #self.critic_opt = optim.AdamW(self.v.parameters(), lr=2e-4, weight_decay=0.01)
      self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_opt, 0.95)
      self.pi_gradsteps = 5
      self.v_gradsteps = 5
      self.minib_size = 64
      self.reward_scalar = 1
      self.kl_coeff = 0.1
      # So we know how to apply the tanh to restrict our actions
      self.action_range = action_range
      #Other
      self.verbose = verbose
      # Set up observation tracking to normalize observations
      self.init_obs_tracking(state_dim)
   
   def init_obs_tracking(self, state_dim):
      """
      Initialize a dictionary `self.inputs` to keep a running average 
      of the standard deviation and mean of all the observations we've seen.

         :param state_dim = state dimension, number of means / stds to keep track of
      """
      self.inputs = dict(
         m2=np.zeros(state_dim),
         means=np.zeros(state_dim),
         step=0
      )
   
   def normalize_observation(self, state) -> torch.Tensor:
      """
      Takes a state, and normalizes it to have a mean of 0 and STD of 1,
      according to a running average that it also updates.
      Uses Welford's algorithm to keep track of the running mean / STD.

         :param state = state to be normalized
      """
      assert state.shape == self.inputs["means"].shape
      

      # Calculate Mean
      self.inputs["step"] += 1
      delta1 = state - self.inputs["means"]
      self.inputs["means"] += delta1 / self.inputs["step"]

      # Calcualte STD
      delta2 = state - self.inputs["means"]
      self.inputs["m2"] += delta1 * delta2
      var = self.inputs["m2"] / (self.inputs["step"] - 1) if self.inputs["step"] > 1 else 0.0
      #std = np.maximum(np.sqrt(var), 1e-6)
      std = np.sqrt(var)

      # Return the normalized observation
      #normalized_state = (state - self.inputs["means"]) / std
      normalized_state = (state - self.inputs["means"]) / (std + 1e-8)
      return torch.as_tensor(normalized_state, dtype=torch.float32)


   def step(self, states):
      """
      We don't need any gradients when we perform a step. Trajectories currently collected
      are done under the "old" policy and probs are used simply as the denom in the ratio term.
      """
      with torch.no_grad():
         # Get pi distribution over state(s)
         distrs = self.pi._get_distribution(states)
         # Sample action(s) from pi
         raw_actions = distrs.sample()
         ##### ADDED, MIGHT NOT WORK #####
         actions = F.tanh(raw_actions)*self.action_range
         #################################
         # Record log probs for later (training pi network)
         log_probs = self.pi._get_log_probs(distrs, raw_actions)
         # Current estimate of value function in order to train policy
         values = self.v(states)
      return raw_actions.numpy(), log_probs.numpy(), values.numpy(), actions.numpy()


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
      # H(p) = - log(p(x))
      approx_kl_divergence = old_probs - primed_log_probs
      kl_loss = approx_kl_divergence * self.kl_coeff
      #print(f"RATIO: {ratio}")
      # 2. rclip(r) * A
      clipped_ratio = torch.clamp(ratio, 1-self.ratio_clip, 1+self.ratio_clip)
      # 3. If the advantage is negative, we don't clip. 
      # Note that we perform gradient ascent (but actually we just take negative and do gradient descent)
      surrogate_loss = -torch.min(clipped_ratio*advantage, ratio*advantage)
      #surrogate_loss = -torch.min(clipped_ratio*advantage, ratio*advantage) + kl_loss
      return surrogate_loss.mean(), approx_kl_divergence.mean().item()
      


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
      state = self.normalize_observation(state)
      return self.step(torch.as_tensor(state, dtype=torch.float32))[0]


   def gradient_step(self):
      """
      After completely filling the buffer, this function does mini-batch SGD over the entire
      buffer, for both the actor and the critic, `self.pi_gradsteps` and `self.v_gradsteps` times,
      respectively.
      """
      # These trajectories are wrapped by the `torch.utils.data.Dataloader` class
      trajectories = self.buffer.get_trajectories_as_DataLoader(batch_size=self.minib_size)

      # N full passes of buffer on critic
      for _ in range(self.v_gradsteps):
         # Track average loss over buffer and keep for logging purpose
         loss = []
         # mini batch SGD over entire buffer
         for minib_traj in trajectories:
            self.critic_opt.zero_grad()
            critic_loss = self.compute_critic_loss(minib_traj)
            critic_loss.backward()
            self.critic_opt.step()
            loss.append(critic_loss.item())
         
         self.v_losses.append(np.mean(loss))

      #N full passes of buffer on actor
      for s in range(self.pi_gradsteps):
         # Track average loss & kl divergence over buffer and keep for logging purpose
         loss = []
         kl_diverge = []
         #mini batch SGD over entire buffer
         for minib_traj in trajectories:
            self.pi_optim.zero_grad()
            actor_loss, dkl = self.compute_actor_loss(minib_traj)
            actor_loss.backward()
            self.pi_optim.step()
            # Tracking
            loss.append(actor_loss.item())
            kl_diverge.append(dkl)
         
         # Logging
         self.pi_losses.append(np.mean(loss))
         self.kl_divergence.append(np.mean(kl_diverge))
         if self.kl_divergence[-1] > 0.015:
            print(f"HIGH AVERAGE DIVERGENCE ON {s}: {self.kl_divergence[-1]}")
   
   def train(self, env, epochs):
      for i in range(epochs):
         state, _ = env.reset()
         # Track the trajectory rewards over each epoch
         trajectory_rewards = []
         trajectory_reward = 0.0
         #Scheduled learning rate for critic
         if i > 200:
            self.critic_scheduler.step()

         for _ in range(self.buffer_size):
            # Normalize the observation
            #print(f"BEFORE {state}\n")
            state = self.normalize_observation(state)
            #print(f"AFTER {state}\n\n")

            raw_action, log_prob, value, true_action = self.step(torch.as_tensor(state, dtype=torch.float32))
            # Sample action, value, calculate log prob
            #raw_action, log_prob, value, true_action = self.step(state)
            #print("ACTIONS", raw_action, true_action)
            state_p, reward, done, _,  _ = env.step(raw_action)
            #print("REWARD", reward)

            scaled_reward = reward * self.reward_scalar

            self.buffer.add_experience(state, raw_action, log_prob, value, scaled_reward)

            trajectory_reward += scaled_reward

            if done:
               # At each trajectory, we look back over the entire trajectory and calculate advantages, returns.
               self.buffer.calculate_advantages(final_V=0)

               # Store trajectory reward and reset
               if self.verbose: print(f"trajectory reward: {trajectory_reward}")
               trajectory_rewards.append(trajectory_reward)
               trajectory_reward = 0.0
               state, _ = env.reset()
            else:
               state = state_p

         # When we exit an epoch -- either we perfectly ended a trajectory (unlikely) or
         # we were in the middle. If so, need to calculate the advantages.
         if not done:
            state_pnorm = self.normalize_observation(state_p)
            #final_state_v = self.v(torch.as_tensor(state_p, dtype=torch.float32)).detach()
            final_state_v = self.v(state_pnorm).detach()
            self.buffer.calculate_advantages(final_state_v)
         self.gradient_step()
         # Average the trajectory rewards across the epoch, store, reset list
         self.reward_tracking.append(np.mean(trajectory_rewards))
         print(f"TRAJECTORY REWARDS FOR EPOCH {i}: {trajectory_rewards}\n")
         print(f"ACTOR LOSSES FOR EPOCH {i}. BEFORE: {self.pi_losses[-5]}\tAFTER: {self.pi_losses[-1]}\n")
         print(f"CRITIC LOSSES FOR EPOCH {i}. BEFORE: {self.v_losses[-5]}\tAFTER: {self.v_losses[-1]}\n\n")



# ppo = ActorCritic(10, 10, 2048)
# randoms = np.asarray([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
# print(ppo.normalize_observation(randoms))
# print(f"\nMEANS: {ppo.inputs["means"]}\n\nM2: {ppo.inputs["m2"]}\n\n")
# randoms2 = np.asarray([1,1,1,1,1,10,10,10,10,10])
# print(ppo.normalize_observation(randoms2))
# print(f"\nMEANS: {ppo.inputs["means"]}\n\nM2: {ppo.inputs["m2"]}\n\n")
############### TEST DATASET ####################
# import gymnasium as gym
# env = gym.make('Ant-v5', terminate_when_unhealthy=True)
# ppo = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], buffer_size=2048)
# ppo.train(env, epochs=1, steps_per_epoch=2048)
# traj_dict = ppo.buffer.get_trajectories()
# wrapped_dataset = TrajectoryDataset(traj_dict)
# # print(wrapped_dataset[0])
# # print(wrapped_dataset[2047])
# # print(wrapped_dataset[-1])
# # print(len(wrapped_dataset))
# wrapped_loader = torch.utils.data.DataLoader(wrapped_dataset, batch_size=64, shuffle=True)
# small_batch = next(iter(wrapped_loader))
# print(small_batch["states"].shape)
# print(small_batch["acts"].shape)
# print(small_batch["advs"].shape)
# print(small_batch["rets"].shape)
# print(small_batch["prev_logprobs"].shape)

# ppo = ActorCritic(10, 10, buffer_size=2048)
# print(ppo.pi.layers[0].weight)
# print(ppo.pi.layers[2].weight)
# print(ppo.pi.layers[4].weight)

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

