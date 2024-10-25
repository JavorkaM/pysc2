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
"""A random Protoss agent for StarCraft II."""

import numpy as np
import random
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env
from pysc2.env import available_actions_printer

# Import the ordered functions
from protoss_ordered_functions import ORDERED_FUNCTIONS, _NO_OP

class RandomProtossAgent(base_agent.BaseAgent):
    """A random Protoss agent for StarCraft II."""

    def step(self, obs):
        super(RandomProtossAgent, self).step(obs)
        
        # Try functions in random order
        random.shuffle(ORDERED_FUNCTIONS)
        
        for func in ORDERED_FUNCTIONS:
            action = func(obs)
            if action[0] != _NO_OP:
                # Extra safety check
                if action[0] in obs.observation.available_actions:
                    return action
        
        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    agent = RandomProtossAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.protoss),
                         sc2_env.Bot(sc2_env.Race.random,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=True) as env:
                
                # Wrap the environment with AvailableActionsPrinter
                env = available_actions_printer.AvailableActionsPrinter(env)
                
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
