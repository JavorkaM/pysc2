from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features, actions
from simple_agent import SimpleProtossAgent

def main(unused_argv):
    agent = SimpleProtossAgent()
    try:
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.protoss),
                         sc2_env.Bot(sc2_env.Race.random,
                                     sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                    feature_dimensions=features.Dimensions(
                        screen=84,
                        minimap=64)),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=False) as env:  # Set visualize to False
            
            for _ in range(10):  # Play 10 episodes
                agent.reset()
                timesteps = env.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
