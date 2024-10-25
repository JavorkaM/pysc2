# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.lib import features


class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    super(RandomAgent, self).step(obs)
    function_id = np.random.choice(obs.observation.available_actions)
    args = []
    for arg in actions.FUNCTIONS[function_id].args:
      if arg.name in ['screen', 'minimap', 'screen2']:
        args.append([np.random.randint(0, d) for d in obs.observation.feature_screen.shape[:2]])
      elif arg.name == 'queued':
        args.append([np.random.randint(0, 2)])
      elif arg.name == 'control_group_act':
        args.append([np.random.randint(0, 5)])
      elif arg.name == 'control_group_id':
        args.append([np.random.randint(0, 10)])
      elif arg.name == 'select_point_act':
        args.append([np.random.randint(0, 4)])
      elif arg.name == 'select_add':
        args.append([np.random.randint(0, 2)])
      elif arg.name == 'select_worker':
        args.append([np.random.randint(0, 4)])
      elif arg.name == 'build_queue_id':
        args.append([np.random.randint(0, 10)])
      elif arg.name == 'unload_id':
        args.append([np.random.randint(0, 500)])
      else:
        args.append([np.random.randint(0, size) for size in arg.sizes])
    return actions.FunctionCall(function_id, args)


def main(unused_argv):
  agent = RandomAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="Simple64",
          players=[sc2_env.Agent(sc2_env.Race.terran),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64),
              use_feature_units=True),
          step_mul=8,
          game_steps_per_episode=0,
          visualize=True) as env:
        
        agent.setup(env.observation_spec(), env.action_spec())

        timesteps = env.reset()
        agent.reset()

        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)

  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)
