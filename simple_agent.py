from pysc2.agents import base_agent
from pysc2.lib import actions, units
from pysc2.lib.actions import RAW_FUNCTIONS
import numpy as np
import random

class SimpleProtossAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SimpleProtossAgent, self).__init__()
        self.attack_coordinates = None

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type]

    def step(self, obs):
        super(SimpleProtossAgent, self).step(obs)
        
        if obs.first():
            nexus = self.get_units_by_type(obs, units.Protoss.Nexus)[0]
            self.attack_coordinates = (nexus.x, nexus.y)

        zealots = self.get_units_by_type(obs, units.Protoss.Zealot)
        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
        nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
        gateways = self.get_units_by_type(obs, units.Protoss.Gateway)

        if len(zealots) >= 3:
            if self.attack_coordinates:
                return RAW_FUNCTIONS.Attack_pt("now", [unit.tag for unit in zealots], self.attack_coordinates)

        if len(probes) < 16 and len(nexus) > 0 and obs.observation.player.minerals >= 50:
            return RAW_FUNCTIONS.Train_Probe_quick("now", nexus[0].tag)

        if len(pylons) == 0 and len(probes) > 0 and obs.observation.player.minerals >= 100:
            pylon_xy = (nexus[0].x + 5, nexus[0].y + 5)
            return RAW_FUNCTIONS.Build_Pylon_pt("now", probes[0].tag, pylon_xy)

        if len(gateways) == 0 and len(pylons) > 0 and len(probes) > 0 and obs.observation.player.minerals >= 150:
            gateway_xy = (nexus[0].x - 5, nexus[0].y - 5)
            return RAW_FUNCTIONS.Build_Gateway_pt("now", probes[0].tag, gateway_xy)

        if len(gateways) > 0 and obs.observation.player.minerals >= 100:
            return RAW_FUNCTIONS.Train_Zealot_quick("now", gateways[0].tag)

        if len(probes) > 0:
            return RAW_FUNCTIONS.Harvest_Gather_unit("now", probes[0].tag, self.get_nearest_mineral_field(obs, nexus[0]).tag)

        return RAW_FUNCTIONS.no_op()

    def get_nearest_mineral_field(self, obs, nexus):
        mineral_fields = [unit for unit in obs.observation.raw_units
                          if unit.unit_type in [units.Neutral.MineralField,
                                                units.Neutral.MineralField750]]
        distances = [(unit.x - nexus.x)**2 + (unit.y - nexus.y)**2 for unit in mineral_fields]
        return mineral_fields[np.argmin(distances)]
