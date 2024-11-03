import random

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env.environment import TimeStep
from pysc2.lib.named_array import NamedNumpyArray
from Agent.ddpg_agent import DDPGAgent
import numpy as np
import wandb
import math
from pysc2.lib import units

# Functions
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_COMMAND_CENTER = actions.FUNCTIONS.Build_CommandCenter_screen.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
_ATTACK_MOVE_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_HARVEST_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_ENGINEERING_BAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_RESEARCH_TERRAN_INFANTRY_WEAPONS = actions.FUNCTIONS.Research_TerranInfantryWeapons_quick.id
_RESEARCH_TERRAN_INFANTRY_ARMOR = actions.FUNCTIONS.Research_TerranInfantryArmor_quick.id
_BUILD_TECHLAB_QUICK = actions.FUNCTIONS.Build_TechLab_quick.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_TRAIN_HELLION = actions.FUNCTIONS.Train_Hellion_quick.id
_TRAIN_TANK = actions.FUNCTIONS.Train_SiegeTank_quick.id
_RESEARCH_COMBAT_SHIELD = actions.FUNCTIONS.Research_CombatShield_quick.id
_RESEARCH_CONCUSSIVE_SHELLS = actions.FUNCTIONS.Research_ConcussiveShells_quick.id
_RESEARCH_STIMPACK = actions.FUNCTIONS.Research_Stimpack_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = units.Terran.CommandCenter
_TERRAN_SUPPLYDEPOT = units.Terran.SupplyDepot
_TERRAN_SCV = units.Terran.SCV
_TERRAN_MARINE = units.Terran.Marine
_TERRAN_BARRACKS = units.Terran.Barracks
_VESPENE_GEYSER = units.Neutral.VespeneGeyser
_REFINERY = units.Terran.Refinery
_ENGINEERING_BAY = units.Terran.EngineeringBay
_TERRAN_FACTORY = units.Terran.Factory
_TERRAN_TECHLAB = units.Terran.TechLab

# Parameters
_PLAYER_SELF = 1
_PLAYER_NEUTRAL = 3
_ENEMY_ID = 4
_MINERAL_FIELD = units.Neutral.MineralField
_NOT_QUEUED = [0]
_QUEUED = [1]


class OrderedFunction:
    def __init__(self):
        pass

    def __call__(self, obs, args):
        pass


class FindWorker(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if obs.observation.single_select is not None:
            if obs.observation.single_select[0][0] == 45:
                return actions.FunctionCall(_NOOP, [])

        if obs.observation["player"]["idle_worker_count"] > 0:
            return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
        else:
            # Find the closest set of coordinates from 2 arrays
            pass


class SelectWorker(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if obs.observation["player"]["idle_worker_count"] > 0:
            return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])

        if len(obs.observation.single_select) > 0:
            if obs.observation.single_select[0][0] == 45:
                return actions.FunctionCall(_NOOP, [])

        if len(obs.observation.multi_select) > 0:
            select_count = len(obs.observation.multi_select)
            return actions.FunctionCall(_SELECT_UNIT, [[0], [random.randrange(select_count)]])

        print("No workers found")
        return actions.FunctionCall(_NOOP, [])


class MoveCamera(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        return actions.FunctionCall(_MOVE_CAMERA, [[self.x, self.y]])


class BuildCommandCenter(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _BUILD_COMMAND_CENTER in obs.observation.available_actions:
            return actions.FunctionCall(_BUILD_COMMAND_CENTER, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class BuildEngineeringBay(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _BUILD_ENGINEERING_BAY in obs.observation.available_actions:
            return actions.FunctionCall(_BUILD_ENGINEERING_BAY, [_NOT_QUEUED, [self.x, self.y]])
        return actions.FunctionCall(_NOOP, [])


class UpgradeInfantryArmor(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _RESEARCH_TERRAN_INFANTRY_ARMOR in obs.observation.available_actions:
            return actions.FunctionCall(_RESEARCH_TERRAN_INFANTRY_ARMOR, [_NOT_QUEUED])
        return actions.FunctionCall(_NOOP, [])


class UpgradeInfantryWeapons(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _RESEARCH_TERRAN_INFANTRY_WEAPONS in obs.observation.available_actions:
            return actions.FunctionCall(_RESEARCH_TERRAN_INFANTRY_WEAPONS, [_NOT_QUEUED])
        return actions.FunctionCall(_NOOP, [])


class MoveCameraToBuildable(OrderedFunction):
    x: int
    y: int
    buildable_mask: np.ndarray

    def __init__(self, x, y, mask=None):
        super().__init__()
        self.x = x
        self.y = y
        self.buildable_mask = mask

    def __call__(self, obs, args):
        _visibility_map = obs.observation.feature_minimap.visibility_map
        _buildable = obs.observation.feature_minimap.buildable

        _visible_y, _visible_x = _visibility_map.nonzero()
        _buildable_y, _buildable_x = _buildable.nonzero()

        _visible_set = set(zip(_visible_x, _visible_y))

        while True:
            _coords = random.choice(list(_visible_set))
            if self.buildable_mask[_coords[1], _coords[0]] == 1:
                break

        return actions.FunctionCall(_MOVE_CAMERA, [[_coords[0], _coords[1]]])


class BuildRefinery(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _BUILD_REFINERY in obs.observation.available_actions:
            # find vespene geyser in feature units
            for unit in obs.observation.feature_units:
                if unit[0] == _VESPENE_GEYSER:
                    self.x = unit[12]
                    self.y = unit[13]

                    if not 0 <= self.x < 83 or not 0 <= self.y < 83:
                        return actions.FunctionCall(_NOOP, [])

                    return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class SaturateRefinery(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        for unit in obs.observation.feature_units:
            if unit[0] == _REFINERY:
                self.x = unit[12]
                self.y = unit[13]

                if not 0 <= self.x < 128 or not 0 <= self.y < 128:
                    return actions.FunctionCall(_NOOP, [])

                # check if refinery is saturated
                if unit[23] < unit[24]:
                    return actions.FunctionCall(_HARVEST_SCREEN, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class BuildSupplyDepot(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _BUILD_SUPPLY_DEPOT in obs.observation.available_actions:
            return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class SelectCommandCenterScreen(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        for unit in obs.observation.feature_units:
            if unit[0] == 18:
                self.x = unit[12]
                self.y = unit[13]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class BuildBarracks(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _BUILD_BARRACKS in obs.observation.available_actions:
            return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class BuildTechlab(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _BUILD_TECHLAB_QUICK in obs.observation.available_actions:
            return actions.FunctionCall(_BUILD_TECHLAB_QUICK, [_NOT_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class ResearchCombatShield(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _RESEARCH_COMBAT_SHIELD in obs.observation.available_actions:
            return actions.FunctionCall(_RESEARCH_COMBAT_SHIELD, [_NOT_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class ResearchConcussiveShells(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _RESEARCH_CONCUSSIVE_SHELLS in obs.observation.available_actions:
            return actions.FunctionCall(_RESEARCH_CONCUSSIVE_SHELLS, [_NOT_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class ResearchStimpack(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _RESEARCH_STIMPACK in obs.observation.available_actions:
            return actions.FunctionCall(_RESEARCH_STIMPACK, [_NOT_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class BuildFactory(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _BUILD_FACTORY in obs.observation.available_actions:
            return actions.FunctionCall(_BUILD_FACTORY, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class RallyToMinerals(OrderedFunction):
    # Minerals ids: 341, 483
    # RallyWorkersScreen: 343

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        for unit in obs.observation.feature_units:
            if unit[0] == 341 or unit[0] == 483:
                self.x = unit[12]
                self.y = unit[13]

                if 343 not in obs.observation.available_actions:
                    return actions.FunctionCall(_NOOP, [])
                return actions.FunctionCall(343, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class TrainWorker(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):

        _cc = None
        for unit in obs.observation.feature_units:
            if unit[0] == 18 and unit[17] == 1:
                _cc = unit
                break

        if _cc is None:
            return actions.FunctionCall(_NOOP, [])

        if _TRAIN_SCV in obs.observation.available_actions:
            if _cc[23] < _cc[24]:
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class TrainMarine(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _TRAIN_MARINE in obs.observation.available_actions:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class TrainMarauder(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _TRAIN_MARAUDER in obs.observation.available_actions:
            return actions.FunctionCall(_TRAIN_MARAUDER, [_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class TrainHellion(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _TRAIN_HELLION in obs.observation.available_actions:
            return actions.FunctionCall(_TRAIN_HELLION, [_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class TrainTank(OrderedFunction):
    x: int
    y: int

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _TRAIN_TANK in obs.observation.available_actions:
            return actions.FunctionCall(_TRAIN_TANK, [_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class SelectUnitFromMultiSelect(OrderedFunction):
    unit_id: int
    select_type: str

    def __init__(self, unit_id, select_type):
        super().__init__()
        self.unit_id = unit_id
        self.select_type = select_type

    def __call__(self, obs, args):
        if len(obs.observation.multi_select) > 0:
            return actions.FunctionCall(_SELECT_UNIT, [self.select_type, self.unit_id])
        else:
            return actions.FunctionCall(_NOOP, [])


class SelectPointUnitID(OrderedFunction):
    x: int
    y: int
    action_type: int
    unit_id: int

    def __init__(self, action_type, unit_id):
        super().__init__()
        self.action_type = action_type
        self.unit_id = unit_id

        # Action types
        # 0 = select
        # 1 = toggle
        # 2 = select all type
        # 3 = add all type

    def __call__(self, obs, args):

        for unit in obs.observation.feature_units:
            if unit[0] == self.unit_id:
                self.x = unit[12]
                self.y = unit[13]

                if 0 < self.x < 83 and 0 < self.y < 83:
                    return actions.FunctionCall(_SELECT_POINT, [self.action_type, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class SelectGroup(OrderedFunction):
    group_id: int

    def __init__(self, group_id):
        super().__init__()
        self.group_id = group_id

    def __call__(self, obs, args):
        _select_id = [0]

        return actions.FunctionCall(_SELECT_CONTROL_GROUP, [_select_id, [self.group_id]])


class SelectArea(OrderedFunction):
    x: int
    y: int
    delta_x: int
    delta_y: int

    def __init__(self, x, y, delta_x, delta_y):
        super().__init__()
        self.x = x
        self.y = y
        self.delta_x = delta_x
        self.delta_y = delta_y

    def __call__(self, obs, args):
        return actions.FunctionCall(_SELECT_RECT,
                                    [_NOT_QUEUED, [self.x, self.y], [self.x + self.delta_x, self.y + self.delta_y]])


class AddToControlGroup(OrderedFunction):
    group_id: int
    unit_id: int

    def __init__(self, group_id, unit_id):
        super().__init__()
        self.group_id = group_id
        self.unit_id = unit_id

    def __call__(self, obs, args):
        _add_id = [2]

        if len(obs.observation.single_select) > 0:
            if obs.observation.single_select[0][0] != self.unit_id:
                return actions.FunctionCall(_NOOP, [])

        if len(obs.observation.multi_select) > 0:
            if not all([unit[0] == self.unit_id for unit in obs.observation.multi_select]):
                return actions.FunctionCall(_NOOP, [])

        return actions.FunctionCall(_SELECT_CONTROL_GROUP, [_add_id, [self.group_id]])


class MoveCameraToSelectedUnit(OrderedFunction):

    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if len(obs.observation.single_select) > 0:
            unit_y, unit_x = obs.observation.feature_minimap.selected.nonzero()

            if np.array_equal(unit_x, np.array([])) or np.array_equal(unit_y, np.array([])):
                return actions.FunctionCall(_NOOP, [])

            x = unit_x.mean()
            y = unit_y.mean()

            print("Moving camera to selected unit: ", x, y)

            return actions.FunctionCall(_MOVE_CAMERA, [[x, y]])

        return actions.FunctionCall(_NOOP, [])


class SelectArmy(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        if _SELECT_ARMY in obs.observation.available_actions:
            return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NOOP, [])


class AttackMove(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):

        if _ATTACK_MOVE_MINIMAP in obs.observation.available_actions:
            # find enemy unit on minimap
            player_relative = obs.observation.feature_minimap.player_relative
            enemy_y, enemy_x = (player_relative == _ENEMY_ID).nonzero()

            if enemy_y.any():
                # random enemy unit
                i = random.randint(0, len(enemy_y) - 1)
                self.x = enemy_x[i]
                self.y = enemy_y[i]

                # find closest zero coords
                while True:
                    if player_relative[self.y, self.x] == 0:
                        break
                    else:
                        self.x += 1
                        self.y += 1

            print("Attacking enemy at: ", self.x, self.y)
            return actions.FunctionCall(_ATTACK_MOVE_MINIMAP, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class RallyPoint(OrderedFunction):
    x: int
    y: int

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __call__(self, obs, args):
        if _RALLY_UNITS_MINIMAP in obs.observation.available_actions:
            return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [self.x, self.y]])

        return actions.FunctionCall(_NOOP, [])


class PrintXYFeatureUnits(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        # Feature units command center X Y coords
        for unit in obs.observation.feature_units:
            if unit[0] == _TERRAN_COMMANDCENTER:
                print("Command center X Y: ", unit[12], unit[13])
        return actions.FunctionCall(_NOOP, [])


class Noop(OrderedFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, obs, args):
        return actions.FunctionCall(_NOOP, [])
