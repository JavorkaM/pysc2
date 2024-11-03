"""A learning Protoss agent for StarCraft II."""

import numpy as np
import random
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env
from pysc2.env import available_actions_printer

# Import the ordered functions
from protoss_ordered_functions import ORDERED_FUNCTIONS, _NO_OP

class LearningProtossAgent(base_agent.BaseAgent):
    """A learning Protoss agent for StarCraft II using Q-learning."""

    def __init__(self):
        super().__init__()
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        # Create a simple state representation
        return (
            obs.observation.player.minerals,
            obs.observation.player.vespene,
            len([unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]),
            len([unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.ENEMY])
        )

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max(ORDERED_FUNCTIONS, key=lambda a: self.get_q_value(next_state, a.__class__.__name__))
        best_next_q_value = self.get_q_value(next_state, best_next_action.__class__.__name__)

        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q_value - current_q)
        self.q_table[(state, action)] = new_q

    def step(self, obs):
        super(LearningProtossAgent, self).step(obs)

        current_state = self.get_state(obs)
        reward = obs.reward

        # Update Q-table if not the first step
        if self.previous_state is not None:
            self.update_q_value(self.previous_state, self.previous_action, reward, current_state)

        # Choose action (epsilon-greedy)
        if random.random() < self.epsilon:
            chosen_function = random.choice(ORDERED_FUNCTIONS)
        else:
            chosen_function = max(ORDERED_FUNCTIONS, key=lambda a: self.get_q_value(current_state, a.__class__.__name__))

        action = chosen_function(obs)

        # If the chosen action is not available, try others
        if action[0] == _NO_OP or action[0] not in obs.observation.available_actions:
            for func in ORDERED_FUNCTIONS:
                action = func(obs)
                if action[0] != _NO_OP and action[0] in obs.observation.available_actions:
                    chosen_function = func
                    break

        self.previous_state = current_state
        self.previous_action = chosen_function.__class__.__name__

        return action

def main(unused_argv):
    agent = LearningProtossAgent()
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
