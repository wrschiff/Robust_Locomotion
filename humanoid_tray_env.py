#@title Humanoid Env
import jax
from jax import numpy as jp
import numpy as np
from matplotlib import pyplot as plt
import os

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model

class Humanoid(PipelineEnv):

   def __init__(
      self,
      forward_reward_weight=1.25,
      ctrl_cost_weight=0.1,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(1.0, 2.0),
      ########### WEIGHT FOR BALANCING TRAY ############
      balance_tray_reward_weight=4.0,
      terminate_when_boxfall=True,
      ##################################################
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,
   ):
   #
      mj_model = mujoco.MjModel.from_xml_path(os.getcwd() + '/basic_humanoid.xml')
      mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
      mj_model.opt.iterations = 6
      mj_model.opt.ls_iterations = 6

      sys = mjcf.load_model(mj_model)

      physics_steps_per_control_step = 5
      kwargs['n_frames'] = kwargs.get(
         'n_frames', physics_steps_per_control_step)
      kwargs['backend'] = 'mjx'

      super().__init__(sys, **kwargs)

      self._forward_reward_weight = forward_reward_weight
      self._ctrl_cost_weight = ctrl_cost_weight
      self._healthy_reward = healthy_reward
      self._terminate_when_unhealthy = terminate_when_unhealthy
      self._healthy_z_range = healthy_z_range
      self._reset_noise_scale = reset_noise_scale
      ########### WEIGHT FOR BALANCING TRAY ############
      self._balance_tray_reward_weight = balance_tray_reward_weight
      self._terminate_when_boxfall = terminate_when_boxfall
      ##################################################
      self._exclude_current_positions_from_observation = (
         exclude_current_positions_from_observation
      )

      ########### INDICES FOR BOX AND TRAY IN XPOS ARRAY ############
      self.tray_x_id = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_XBODY, "tray")
      self.box_x_id = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_XBODY, "box")
      ###############################################################

   def reset(self, rng: jp.ndarray) -> State:
      """Resets the environment to an initial state."""
      rng, rng1, rng2 = jax.random.split(rng, 3)

      low, hi = -self._reset_noise_scale, self._reset_noise_scale
      qpos = self.sys.qpos0 + jax.random.uniform(
         rng1, (self.sys.nq,), minval=low, maxval=hi
      )
      qvel = jax.random.uniform(
         rng2, (self.sys.nv,), minval=low, maxval=hi
      )

      data = self.pipeline_init(qpos, qvel)

      obs = self._get_obs(data, jp.zeros(self.sys.nu))
      reward, done, zero = jp.zeros(3)
      metrics = {
         'forward_reward': zero,
         'reward_linvel': zero,
         'reward_quadctrl': zero,
         'reward_alive': zero,
         'x_position': zero,
         'y_position': zero,
         'distance_from_origin': zero,
         'x_velocity': zero,
         'y_velocity': zero,
      }
      return State(data, obs, reward, done, metrics)

   def step(self, state: State, action: jp.ndarray) -> State:
      """Runs one timestep of the environment's dynamics."""
      data0 = state.pipeline_state
      data = self.pipeline_step(data0, action)

      com_before = data0.subtree_com[1]
      com_after = data.subtree_com[1]
      velocity = (com_after - com_before) / self.dt
      forward_reward = self._forward_reward_weight * velocity[0]

      min_z, max_z = self._healthy_z_range
      is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
      is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
      if self._terminate_when_unhealthy:
         healthy_reward = self._healthy_reward
      else:
         healthy_reward = self._healthy_reward * is_healthy

      ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

      ############## CALCULATE BOX-TRAY REWARD ##################
      euclid_dist_tb = jp.linalg.norm(data.x.pos[self.tray_x_id] - data.x.pos[self.box_x_id])
      balance_cost = euclid_dist_tb * self._balance_tray_reward_weight
      ###########################################################

      obs = self._get_obs(data, action)
      ############## ADD TO OVERALL REWARD ##################
      reward = forward_reward + healthy_reward - ctrl_cost - balance_cost
      #######################################################

      print(f'CTRL COST BEFORE SCALAR (as benchmark): {ctrl_cost}')
      print(f'EUCLID DISTANCE: {euclid_dist_tb} \t\tSCALED REWARD: {balance_cost}\t\tTOTAL REWARD: {reward}')

      ########## ADDING TERMINATION CONSTRAINT IF BOX FALLS OFF TRAY ############
      is_balanced = data.x.pos[self.tray_x_id][2] < data.x.pos[self.box_x_id][2]
      done = 0.0
      if (self._terminate_when_unhealthy):
         done = 1.0 - is_healthy
      if (self._terminate_when_boxfall and done == 0.0):
         done = 1.0 - is_balanced
      ###########################################################################
      # PREVIOUS METHOD: done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

      print(f'TRAY HEIGHT: {data.x.pos[self.tray_x_id][2]}\tBOX HEIGHT: {data.x.pos[self.box_x_id][2]}\tDONE:{done}')
      
      state.metrics.update(
         forward_reward=forward_reward,
         reward_linvel=forward_reward,
         reward_quadctrl=-ctrl_cost,
         reward_alive=healthy_reward,
         x_position=com_after[0],
         y_position=com_after[1],
         distance_from_origin=jp.linalg.norm(com_after),
         x_velocity=velocity[0],
         y_velocity=velocity[1],
      )

      return state.replace(
         pipeline_state=data, obs=obs, reward=reward, done=done
      )

   def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
   ) -> jp.ndarray:
      """Observes humanoid body position, velocities, and angles."""
      position = data.qpos
      if self._exclude_current_positions_from_observation:
         position = position[2:]

      # external_contact_forces are excluded
      return jp.concatenate([
         position,
         data.qvel,
         data.cinert[1:].ravel(),
         data.cvel[1:].ravel(),
         data.qfrc_actuator,
      ])


envs.register_environment('humanoid', Humanoid)
if __name__ == '__main__':
   env = envs.get_environment('humanoid')
   #print(env.sys.init_q)
   #print(env.sys.link_types)
   #tray_id = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tray")
   tray_id_x = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_XBODY, "tray")
   box_id_x = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_XBODY, "box")
   state = env.reset(jax.random.PRNGKey(0))
   tray_pos = state.pipeline_state.x.pos[tray_id_x]
   box_pos = state.pipeline_state.x.pos[box_id_x]
   print(tray_pos, '\n', box_pos)
   ctrl = -0.1 * jp.ones(env.sys.nu)
   state_p = env.step(state, ctrl)


#print(state.pipeline_state.body("tray"))
# #state = env.reset(np.random.randint(0, 50))
# print(state.pipeline_state.q)
# print(state.obs)
# print(state.done)

