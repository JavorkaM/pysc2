import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop


class RawAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RawAgent, self).__init__()
        self.base_top_left = None

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(RawAgent, self).step(obs)

        if obs.first():
            nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)[0]
            self.base_top_left = (nexus.x < 32)

        pylons = self.get_my_units_by_type(obs, units.Protoss.Pylon)
        completed_pylons = self.get_my_completed_units_by_type(
            obs, units.Protoss.Pylon)

        gateways = self.get_my_units_by_type(obs, units.Protoss.Gateway)
        completed_gateways = self.get_my_completed_units_by_type(
            obs, units.Protoss.Gateway)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        zealots = self.get_my_units_by_type(obs, units.Protoss.Zealot)

        if len(pylons) == 0 and obs.observation.player.minerals >= 100:
            probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                pylon_xy = (22, 20) if self.base_top_left else (35, 42)
                distances = self.get_distances(obs, probes, pylon_xy)
                probe = probes[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)

        if (len(completed_pylons) > 0 and len(gateways) == 0 and
                obs.observation.player.minerals >= 150):
            probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                gateway_xy = (22, 24) if self.base_top_left else (35, 45)
                distances = self.get_distances(obs, probes, gateway_xy)
                probe = probes[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Gateway_pt(
                    "now", probe.tag, gateway_xy)

        if (len(completed_gateways) > 0 and obs.observation.player.minerals >= 100
                and free_supply >= 2):
            gateway = gateways[0]
            if gateway.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Zealot_quick("now", gateway.tag)

        if free_supply < 2 and len(zealots) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, zealots, attack_xy)
            zealot = zealots[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", zealot.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        
        if len(zealots) > 0:
            enemy_base_top_left = (38, 44)
            enemy_base_top_right = (38, 21)
            enemy_base_bottom_left = (21, 44)
            enemy_base_bottom_right = (21, 21)

            possible_enemy_bases = [enemy_base_top_left, enemy_base_top_right, 
                                    enemy_base_bottom_left, enemy_base_bottom_right]
            
            # Remove our base location from possible enemy bases
            our_base = enemy_base_bottom_right if self.base_top_left else enemy_base_top_left
            possible_enemy_bases.remove(our_base)

            # Choose a random base location to attack
            target = random.choice(possible_enemy_bases)

            return actions.RAW_FUNCTIONS.Attack_minimap("now", [z.tag for z in zealots], target)
        
        # Check if enemy units are near our base
        enemy_units = self.get_enemy_units_by_type(obs, units.Protoss.Zealot)  # You might want to expand this to include other unit types
        probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
        
        if len(enemy_units) > 0 and len(probes) > 0:
            enemy_near_base = False
            for enemy in enemy_units:
                for probe in probes:
                    if self.get_distance(enemy.x, enemy.y, probe.x, probe.y) < 20:  # Adjust this distance as needed
                        enemy_near_base = True
                        break
                if enemy_near_base:
                    break
            
            if enemy_near_base and len(zealots) > 0:
                # Recall zealots to defend
                defend_xy = (22, 22) if self.base_top_left else (35, 40)  # Adjust these coordinates as needed
                return actions.RAW_FUNCTIONS.Attack_pt(
                    "now", [z.tag for z in zealots], defend_xy)


def main(unused_argv):
    agent = RawAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.protoss),
                             sc2_env.Bot(sc2_env.Race.protoss,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        action_space=actions.ActionSpace.RAW,
                        use_raw_units=True,
                        raw_resolution=64,
                    ),
            ) as env:
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
