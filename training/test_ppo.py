import baseppo
import gymnasium as gym
import torch
import numpy as np
import time


env = gym.make('Ant-v5', terminate_when_unhealthy=True)
# print(env.observation_space.shape[0])
# state, _ = env.reset()
# state, reward, = env.step(np.random.rand(env.action_space.shape))
# print(env.action_space.shape)
#print(state)
# print(reward)
# print(done)
ppo = baseppo.ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], buffer_size=4000)
ppo.train(env=env, epochs=25, steps_per_epoch=4000)
# torch.save(ppo.pi.state_dict(), "rl_models/pi")
# torch.save(ppo.v.state_dict(), "rl_models/v")
# ppo.v.load_state_dict(torch.load("rl_models/v"))
# ppo.pi.load_state_dict(torch.load("rl_models/pi"))

# import stable_baselines3.ppo as stable_ppo
# baseline_ppo = stable_ppo.PPO(policy="MlpPolicy", env=env)
# baseline_ppo.learn(50000)



env = gym.make('Ant-v5', render_mode="human", terminate_when_unhealthy=True)
rewards = []
with torch.no_grad():
   while True:
      state, _ = env.reset()
      rewards.append(0)
      done = False
      while not done:
         action = ppo.act(state)
         #action, _ = baseline_ppo.predict(state, deterministic=True)
         state_p, reward, done, _, _ = env.step(action)
         state = state_p
         rewards[-1] += reward
         #time.sleep(0.05)
      print(f"Trajectory ended. Overall reward: {rewards[-1]}")
   
