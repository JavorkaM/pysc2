from pysc2.lib import actions, features, units
import numpy as np
import random

# Define some constants
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_PROTOSS_NEXUS = units.Protoss.Nexus
_PROTOSS_PROBE = units.Protoss.Probe
_PROTOSS_PYLON = units.Protoss.Pylon
_PROTOSS_GATEWAY = units.Protoss.Gateway
_PROTOSS_ZEALOT = units.Protoss.Zealot

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY = actions.FUNCTIONS.Build_Gateway_screen.id
_BUILD_NEXUS = actions.FUNCTIONS.Build_Nexus_screen.id
_TRAIN_PROBE = actions.FUNCTIONS.Train_Probe_quick.id
_TRAIN_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

_NOT_QUEUED = [0]
_QUEUED = [1]

_MINERAL_FIELD = units.Neutral.MineralField  # Corrected this line

class OrderedFunction:
    def __init__(self):
        pass

    def __call__(self, obs):
        pass

    def is_point_on_screen(self, x, y, screen_size=84):
        return 0 <= x < screen_size and 0 <= y < screen_size

    def get_enemy_base_location(self, obs):
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == _PLAYER_ENEMY).nonzero()
        if enemy_y.any():
            return [int(enemy_x.mean()), int(enemy_y.mean())]
        return None

    def get_expansion_location(self, obs):
        # Find all mineral fields on the map
        mineral_y, mineral_x = (obs.observation.feature_minimap.unit_type == _MINERAL_FIELD).nonzero()
        mineral_locations = list(zip(mineral_x, mineral_y))
        
        # Find all current Nexus locations
        nexus_y, nexus_x = (obs.observation.feature_minimap.unit_type == _PROTOSS_NEXUS).nonzero()
        nexus_locations = set(zip(nexus_x, nexus_y))
        
        # Find a mineral field that's not too close to existing Nexus locations
        for mineral_loc in mineral_locations:
            if all(self.distance(mineral_loc, nexus_loc) > 10 for nexus_loc in nexus_locations):
                return mineral_loc
        
        return None

    def distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

class SelectProbe(OrderedFunction):
    def __call__(self, obs):
        probes = [unit for unit in obs.observation.feature_units 
                  if unit.unit_type == _PROTOSS_PROBE]
        if len(probes) > 0:
            probe = random.choice(probes)
            if self.is_point_on_screen(probe.x, probe.y):
                return actions.FUNCTIONS.select_point("select", (probe.x, probe.y))
        return actions.FUNCTIONS.no_op()

class SelectIdleProbe(OrderedFunction):
    def __call__(self, obs):
        if _SELECT_IDLE_WORKER in obs.observation.available_actions:
            return actions.FUNCTIONS.select_idle_worker("select")
        return actions.FUNCTIONS.no_op()

class SelectNexus(OrderedFunction):
    def __call__(self, obs):
        nexuses = [unit for unit in obs.observation.feature_units 
                   if unit.unit_type == _PROTOSS_NEXUS]
        if len(nexuses) > 0:
            nexus = nexuses[0]
            if self.is_point_on_screen(nexus.x, nexus.y):
                return actions.FUNCTIONS.select_point("select", (nexus.x, nexus.y))
        return actions.FUNCTIONS.no_op()

class SelectGateway(OrderedFunction):
    def __call__(self, obs):
        gateways = [unit for unit in obs.observation.feature_units 
                    if unit.unit_type == _PROTOSS_GATEWAY]
        if len(gateways) > 0:
            gateway = random.choice(gateways)
            if self.is_point_on_screen(gateway.x, gateway.y):
                return actions.FUNCTIONS.select_point("select", (gateway.x, gateway.y))
        return actions.FUNCTIONS.no_op()

class BuildPylon(OrderedFunction):
    def __call__(self, obs):
        if _BUILD_PYLON not in obs.observation.available_actions:
            return actions.FUNCTIONS.no_op()
        # Find area
        protoss_struct_y, protoss_struct_x = (obs.observation.feature_screen.unit_type == _PROTOSS_PYLON).nonzero()
        
        if len(protoss_struct_x) > 0:
            # Build near existing structures with offset
            target_x = protoss_struct_x[0] + random.randint(-15, 15)
            target_y = protoss_struct_y[0] + random.randint(-15, 15)
        else:
            # build near the center of the screen
            target_x = random.randint(30, 50)
            target_y = random.randint(30, 50)
        
        # within screen bounds?
        target_x = max(0, min(83, target_x))
        target_y = max(0, min(83, target_y))
        
        return actions.FUNCTIONS.Build_Pylon_screen("now", (target_x, target_y))

class BuildGateway(OrderedFunction):
    def __call__(self, obs):
        if _BUILD_GATEWAY in obs.observation.available_actions:
            probes = [unit for unit in obs.observation.feature_units 
                      if unit.unit_type == _PROTOSS_PROBE]
            if len(probes) > 0:
                pylons = [unit for unit in obs.observation.feature_units 
                          if unit.unit_type == _PROTOSS_PYLON]
                
                if pylons:
                    # Try to find a valid position near a pylon
                    for pylon in pylons:
                        for _ in range(10):  # Try 10 times for each pylon
                            target_x = pylon.x + random.randint(-15, 15)
                            target_y = pylon.y + random.randint(-15, 15)
                            
                            # Ensure coordinates are within screen bounds
                            target_x = max(0, min(83, target_x))
                            target_y = max(0, min(83, target_y))
                            
                            # Check if the position is buildable
                            if obs.observation.feature_screen.buildable[target_y, target_x]:
                                return actions.FUNCTIONS.Build_Gateway_screen("now", (target_x, target_y))
                
                # If no valid position found near pylons, try center of the screen
                for _ in range(10):
                    target_x = random.randint(20, 60)
                    target_y = random.randint(20, 60)
                    if obs.observation.feature_screen.buildable[target_y, target_x]:
                        return actions.FUNCTIONS.Build_Gateway_screen("now", (target_x, target_y))
        
        return actions.FUNCTIONS.no_op()

class TrainZealot(OrderedFunction):
    def __call__(self, obs):
        gateways = [unit for unit in obs.observation.feature_units 
                    if unit.unit_type == _PROTOSS_GATEWAY]
        if len(gateways) > 0:
            if _TRAIN_ZEALOT in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Zealot_quick("now")
        return actions.FUNCTIONS.no_op()

class AttackEnemy(OrderedFunction):
    def __call__(self, obs):
        zealots = [unit for unit in obs.observation.feature_units 
                   if unit.unit_type == _PROTOSS_ZEALOT]
        if len(zealots) > 0:
            if _SELECT_ARMY in obs.observation.available_actions:
                actions.FUNCTIONS.select_army("select")
            if _ATTACK_MINIMAP in obs.observation.available_actions:
                enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == _PLAYER_ENEMY).nonzero()
                if enemy_y.any():
                    index = np.argmax(enemy_y)
                    target = [enemy_x[index], enemy_y[index]]
                    return actions.FUNCTIONS.Attack_minimap("now", target)
        return actions.FUNCTIONS.no_op()

class TrainProbe(OrderedFunction):
    def __call__(self, obs):

        nexuses = [unit for unit in obs.observation.feature_units 
                   if unit.unit_type == _PROTOSS_NEXUS]
        if len(nexuses) > 0:
            if _TRAIN_PROBE in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Probe_quick("now")
        return actions.FUNCTIONS.no_op()

class SelectArmy(OrderedFunction):
    def __call__(self, obs):
        army_units = [unit for unit in obs.observation.feature_units 
                      if unit.unit_type in [_PROTOSS_ZEALOT]  # Add other military unit types here
                      and unit.alliance == features.PlayerRelative.SELF]
        
        if army_units:
            if len(army_units) > 1:
                # If we have multiple units, select them all
                xy = (army_units[0].x, army_units[0].y)
                return actions.FUNCTIONS.select_army("select")
            else:
                # If we have only one unit, select it directly
                xy = (army_units[0].x, army_units[0].y)
                return actions.FUNCTIONS.select_point("select", xy)
        
        return actions.FUNCTIONS.no_op()

class AttackEnemyBase(OrderedFunction):
    def __call__(self, obs):
        if _ATTACK_MINIMAP not in obs.observation.available_actions:
            return actions.FUNCTIONS.no_op()
        
        # Check if we have selected military units
        selected_units = obs.observation.single_select
        multi_select = obs.observation.multi_select
        
        if selected_units.size == 0 and multi_select.size == 0:
            return actions.FUNCTIONS.no_op()
        
        if selected_units.size > 0:
            unit_type = selected_units[0].unit_type
        elif multi_select.size > 0:
            unit_type = multi_select[0].unit_type
        else:
            return actions.FUNCTIONS.no_op()
        
        if unit_type not in [_PROTOSS_ZEALOT]:  # Add other military unit types here
            return actions.FUNCTIONS.no_op()
        
        enemy_structures_y, enemy_structures_x = (obs.observation.feature_minimap.unit_type == _PROTOSS_NEXUS).nonzero()
        
        if enemy_structures_y.size > 0:
            # attack  first enemy structure found
            target = [enemy_structures_x[0], enemy_structures_y[0]]
        else:
            # fall back to the mean enemy unit position
            enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == _PLAYER_ENEMY).nonzero()
            if enemy_y.size > 0:
                target = [int(enemy_x.mean()), int(enemy_y.mean())]
            else:
                # attack a random corner
                target = [random.randint(0, 63), random.randint(0, 63)]
        
        return actions.FUNCTIONS.Attack_minimap("now", target)

class Expand(OrderedFunction):
    def __call__(self, obs):
        if _BUILD_NEXUS in obs.observation.available_actions:
            expansion_loc = self.get_expansion_location(obs)
            if expansion_loc:
                return actions.FUNCTIONS.Build_Nexus_screen("now", expansion_loc)
        return actions.FUNCTIONS.no_op()

# list of functions
ORDERED_FUNCTIONS = [
    SelectProbe(),
    SelectIdleProbe(),
    SelectNexus(),
    SelectGateway(),
    BuildPylon(),
    BuildGateway(),
    TrainProbe(),
    TrainZealot(),
    SelectArmy(),
    AttackEnemyBase(),
    Expand()
]
