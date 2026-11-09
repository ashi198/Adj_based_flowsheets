# environment_actions_graph.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from core.abstract import BaseTrajectory
from core.utils import softmax
import numpy as np
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph
import copy, torch
from torch import nn
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph
import traceback, random, string



class FlowsheetDesign:

    """
    Graph-based action environment:

      Level 0: choose OPEN stream (or index 0 = TERMINATE)
      Level 1: choose UNIT type 
      Level 2: choose parameters for the following UNITS <distillation_column, add_solvent, decanter, split, recycle, mixer>
      Level 3: choose further parameters for <add_solvent, mixer>

    After an action completes successfully:
      - placement/wiring happens (add_unit / add_recycle)
      - simulate() runs
      - NPV is recomputed
      - state resets to level 0

    On any failure:
      - we roll back partial changes (including recycle edges)
      - reset to level 0 (no loops)
    """

    def __init__(self, random_instance: Dict[str, Any], gen_config, env_config, 
                 simulator_class=FlowsheetSimulationGraph):
        self.gen_config = gen_config
        self.env_config = env_config 
        self.sim = simulator_class(random_instance, self.env_config)
        self.num_units = len(self.env_config.unit_types)
        self.problem_instance = random_instance

        # add feed nodes
        self.sim.feed_nodes = []
        for feed in random_instance["list_feed_streams"]:
            self.sim.feed_nodes.append(self.sim.add_feed(feed))

        # current action state
        self.level = 0 
        self.failed_simulator_call = 0
        self.valid_nodes: torch.Tensor = None
        self.intermediate_rewards: List[float] = [0.0]
        self.intermediate_action_rewards: List[float] = [0.0]

        # counters (limits)
        self.counts = {
            "distillation_column": 0,
            "decanter": 0,
            "split": 0,
            "mixer": 0,
            "recycle": 0,
            "add_solvent": 0,
        }

        self.total_units_placed = 0
        self.history: List[int] = []
        self.level_list: List[int] = []
        self._action_seq_start = 0
        self.recycle_edge_fail = 0
        self.advantage = None

        self.current_action_mask: Optional[np.array] = None # The action mask indicates before each action what is feasible at the current level.

        # limits on units 
        self.max_total_units = self.env_config.action_limits[self.problem_instance['system_name']]['max_total_units'] # overall cap on placed units (excluding feed)
        self.min_total_units = self.env_config.action_limits[self.problem_instance['system_name']]['min_total_units']
        self.max_distillation_columns = self.env_config.action_limits[self.problem_instance['system_name']]['max_distillation_columns']
        self.max_decanters = self.env_config.action_limits[self.problem_instance['system_name']]['max_decanters']
        self.max_split = self.env_config.action_limits[self.problem_instance['system_name']]['max_split']
        self.max_mixer = self.env_config.action_limits[self.problem_instance['system_name']]['max_mixer']
        self.max_recycle = self.env_config.action_limits[self.problem_instance['system_name']]['max_recycle']
        self.max_solvent = self.env_config.action_limits[self.problem_instance['system_name']]['max_solvent']

        # initial simulate to populate open streams/NPV
        self.sim.simulate()
        self.objective = 0.0
        self.action_mapping = {}      
        self.identifier: str = None
        self.current_state = self.get_current_state()
        self.get_feasible_actions()

    def get_current_state(self) -> Dict[str, Any]:
        state = {
            "current_level": self.level,
            "open_streams": self._enumerate_open_streams(), # # (node_id, label) of exisiting streams 
            "chosen_open_stream": None, # Optional[Tuple[int, str]] # (node_id, label)
            "chosen_unit": None, # Optional[Tuple[int, str]] 

            "pending_params": {
            "distillation_column": None,
            "split": None,
            "recycle": None, 
            "add_solvent": {
                "index_for_comp": None,
                "name_comp": None,
                "index_for_amount": None,
                "amount_value": None},
            "mixer": None}, #Optional[Dict] = (int, int) # tuple of index, value of the 2nd o/p stream
            
            "npv_raw": None, #raw values of nvp simulation
            "npv_norm": None, #norm values 
            "completed_design": None, # True only when termination is selected
            "second_open_stream_dest_node": None, #Optional[int] 
            "second_open_stream": None, #Optional[Tuple[int, str]]
            "recycle_dest_unit": None, #Optional[int]
            "current_action_mask": self.current_action_mask,
                            }
        return state


    def get_feasible_actions(self) -> np.ndarray:

        """
        Return a 0/1 vector mask feasible actions for the current level:

        Level 0: indices enumerate all open streams, + 1 for "terminate"
        Level 1: indices enumerate unit choices (0...num_units-1) 
        Level 2: select parameters values for the following units (distillation column: DF value (100), split: split_ratio (100), 
        mixer: 2nd open node (N = num of nodes within the graph), add_solvent: select a compound (from 5 options)
        recycler: destination node (N = num of nodes within the graph))
        Level 3: select amount for selected compound for add_solvent (100) or select open stream for the selected node for mixer 

        0: the action is masked. 1 = action is allowed

        """
        if self.level == 0:
            open_streams = self._enumerate_open_streams()
            mask = np.zeros(len(open_streams) + 1, dtype=int)
            self.action_mapping = {'level_0 (terminate or stream_selection)': {i: None for i in range(len(mask))}}

            for i, stream in enumerate(open_streams):
                if self.valid_units_dest_for_stream(stream):
                    mask[i + 1] = 1
                    self.action_mapping['level_0 (terminate or stream_selection)'][i + 1] = stream

            if self.forced_termination() or not np.any(mask):   
                mask[0] =  1              
                mask[1:] = 0
                self.action_mapping['level_0 (terminate or stream_selection)'][0] = 'terminate'

            elif self.optional_termination():
                mask[0] = 1
                self.action_mapping['level_0 (terminate or stream_selection)'][0] = 'terminate'

            # enable all available stream slots (everything is available)
            self.current_state["open_streams"] = open_streams
            self.current_action_mask = mask 
            return mask   

        # select units now 
        elif self.level == 1:
            unit_params_mask = np.zeros(self.env_config.num_units, dtype=int)
            self.action_mapping = {'level_1 (unit_selection)': {i: None for i in range(len(unit_params_mask))}}
            for idx, unit_name in enumerate(self.env_config.units_map_indices_type):
                if self._unit_available(unit_name):
                    unit_params_mask[idx] = 1
                    self.action_mapping['level_1 (unit_selection)'][idx] = unit_name
            self.current_action_mask = unit_params_mask
            return unit_params_mask

        elif self.level == 2:
            open_streams = self.current_state["open_streams"]
            _, chosen_unit_name = self.current_state["chosen_unit"]

            if chosen_unit_name in ["distillation_column", "split"]:
                if chosen_unit_name == 'distillation_column':
                    params_mask = np.ones(len(self.env_config.DF_distillation_map), dtype=int)
                    self.action_mapping = {'level_2 (distillation_column_ratio)': {i: float(df) for i, df in enumerate(self.env_config.DF_distillation_map)}}
                else:
                    params_mask = np.ones(len(self.env_config.split_ratio_map), dtype=int)
                    self.action_mapping = {'level_2 (split_ratio)': {i: float(df) for i, df in enumerate(self.env_config.split_ratio_map)}}

            elif chosen_unit_name == "add_solvent":
                params_mask = np.zeros(len(self.env_config.component_names), dtype= int)
                self.action_mapping = {'level_2 (add_solvent_component_selection)': {i: None for i in range(len(params_mask))}}
                for i in self.problem_instance['possible_ind_add_comp']:
                    params_mask[i] = 1
                    self.action_mapping['level_2 (add_solvent_component_selection)'][i] = self.env_config.component_names[i]

            elif chosen_unit_name == "recycle":
                node_ids = list(self.sim.graph.nodes)
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                params_mask = np.zeros(len(node_ids), dtype=int) 
                self.action_mapping = {'level_2 (recycle_destination_selection)': {i: None for i in range(len(params_mask))}}
                recycle_dests = self._eligible_recycle_destinations(stream=self.current_state["chosen_open_stream"])
                for node_id in recycle_dests:
                    params_mask[id_to_idx[node_id]] = 1
                    self.action_mapping['level_2 (recycle_destination_selection)'][id_to_idx[node_id]] = id_to_idx[node_id]

            elif chosen_unit_name == "mixer":
                node_ids = list(self.sim.graph.nodes)
                id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                params_mask = np.zeros(len(node_ids), dtype=int) # number of nodes present in the graph
                self.action_mapping = {'level_2 (mixer_destination_selection)': {i: None for i in range(len(params_mask))}}
                for node_id, _ in self._eligible_mixer_streams(self.current_state["chosen_open_stream"]):
                    params_mask[id_to_idx[node_id]] = 1
                    self.action_mapping['level_2 (mixer_destination_selection)'][id_to_idx[node_id]] = id_to_idx[node_id]
            
            self.current_action_mask = params_mask
            return params_mask

        elif self.level == 3:
            _, unit_name = self.current_state["chosen_unit"]
            if unit_name == "add_solvent":
                _, component, _, _ = self.current_state["pending_params"]["add_solvent"].values()
                if component:
                    params_mask =  np.ones(len(self.env_config.add_solvent_comp_map[component]), dtype=int)
                    self.action_mapping = {'level_3 (add_solvent_amount)': {i: float(amo) for i, amo in enumerate(self.env_config.add_solvent_comp_map[component])}}

            if unit_name == "mixer":
                params_mask = np.zeros(self.env_config.max_outlets, dtype=int)
                self.action_mapping = {'level_3 (mixer_destination_outlets)': {i: None for i in range(len(params_mask))}}
                for i, name in self._eligible_mixer_streams(self.current_state["chosen_open_stream"]):
                    if i == self.current_state['second_open_stream_dest_node']:
                        if name == 'out0':
                            params_mask[0] = 1
                            self.action_mapping['level_3 (mixer_destination_outlets)'][i] = 'out0'
                        elif name == 'out1':
                            params_mask[1] = 1
                            self.action_mapping['level_3 (mixer_destination_outlets)'][i] = 'out1'
            
            self.current_action_mask = params_mask # decide which outlet to select for available nodes

            return params_mask
        
        return np.array([], dtype=int)

    def take_action(self, action_index: int, next_level: None) -> Tuple[bool, float, bool]:

        """
        A action index that agent takes across different levels. 
        Actions can be taken across 4 levels in a hierarchical way. 

        Level 0: choose either to terminate design <index 0> or select an open stream 
        Level 1: For a given stream, select a unit type: <distillation_column, add_solvent, decanter, split, recycle, mixer>
        Level 2: Select parameters values for the following units 
         -- distillation column: select a distillation fraction value 
         -- decanter: nothing 
         -- split: select split ratio
         -- recycle: select the destination node, given a source node 
         -- mixer: select the destination node with open streams 
         -- add_solvent: select a compound between 5 global available compounds 
        
        Level 3: Further parameter selection for units add_solvent and mixer 
        -- select amount for selected compound for add_solvent 
        -- select open stream for the selected node for mixer 

        Returns:
          finished_design (bool) = true if termination chosen or max units reached
          reward (float) = current (raw) NPV after completing an action
          move_worked (bool) = false if placement failed (e.g., PEQ failure / convergence fail)

        """

        assert not self.current_state['completed_design'], "Taking action on an already terminated design!"

        action_index = int(action_index)
        if action_index < 0 or action_index >= len(self.current_action_mask):
            raise ValueError(f"Invalid action {action_index}, mask size {len(self.current_action_mask)}")
        if self.current_action_mask[action_index] == 0:
            raise AssertionError(f"Illegal action {action_index} on level {self.level}: action is masked."
            )

        try:
            self.current_state['current_level'] = self.level 
            self.current_state['current_action_mask'] = self.current_action_mask
            open_streams = self._enumerate_open_streams()
            self.current_state['open_streams'] = open_streams

            if self.level == 0:
                self._action_seq_start = len(self.history)
                if action_index == 0: 
                    self.current_state['completed_design'] = True 
                    self.level_list.append(self.level)
                    self.history.append(action_index)
                    self.identifier = self.generate_custom_flowsheet_id()
                    self.intermediate_action_rewards.append(self.objective)
                    return True, self.objective, True

                # if not terminate index, then open_stream selected 
                selected_stream = open_streams[action_index - 1], 
                self.current_state['chosen_open_stream'] = selected_stream[0]
                self.level_list.append(self.level)
                self.level = 1 if next_level == None else next_level
                self.history.append(action_index)
                self.intermediate_action_rewards.append(self.objective)
                self.get_feasible_actions()
                return False, self.objective, True

            elif self.level == 1:
                unit_idx = action_index
                if unit_idx < 0 or unit_idx >= self.env_config.num_units:
                    raise ValueError("Illegal unit index.")
                unit_name = self.env_config.units_map_indices_type[unit_idx]
                if not self._unit_available(unit_name):
                    raise ValueError("Unit not available due to limits or feasibility.")
            
                if unit_name == "distillation_column":
                    # add a new category for DF ratio
                    #self.current_state["pending_params"]["distillation_column"] = None # tuple of index, value of the chosen DF 
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.intermediate_action_rewards.append(self.objective)
                    self.get_feasible_actions()
                    return False, self.objective, True
                 
                if unit_name == "split":
                    # add a new category for DF ratio
                    #self.current_state["pending_params"]["split"] = None # tuple of index, value of the chosen ratio
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.intermediate_action_rewards.append(self.objective)
                    self.get_feasible_actions()
                    return False, self.objective, True

                if unit_name == "add_solvent":
                    #self.current_state["pending_params"]["add_solvent"] = {}
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    self.intermediate_action_rewards.append(self.objective)
                    return False, self.objective, True

                # recycle (destination selection)
                if unit_name == "recycle":
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.get_feasible_actions()
                    self.intermediate_action_rewards.append(self.objective)
                    return False, self.objective, True
                
                if unit_name == "mixer":
                    self.level_list.append(self.level)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)
                    self.history.append(action_index)
                    self.level = 2 if next_level == None else next_level
                    self.intermediate_action_rewards.append(self.objective)
                    self.get_feasible_actions()
                    return False, self.objective, True 

                if unit_name == "decanter":
                    # immediate place (no continuous param, no second stream)
                    self.level_list.append(self.level)
                    self.history.append(action_index)
                    self.current_state['chosen_unit'] = (unit_idx, unit_name)                  
                    done, reward, worked = self._complete_action_place_and_simulate()
                    self.current_state["completed_design"] = done 
                    self.objective = reward
                    self.intermediate_action_rewards.append(self.objective) 
                    self.current_state['open_streams'] = self._enumerate_open_streams()
                    self.level = 0 if next_level == None else next_level
                    self.get_feasible_actions()
                    return done, self.objective, worked

            elif self.level == 2:
                # for distillation, decide on which DF ratio value to choose 
                if self._chosen_unit_name() == "distillation_column":
                    if action_index > len (self.env_config.DF_distillation_map):
                        raise ValueError("Distillation fraction value selected more than the permissible limit.")
                    else:
                        self.current_state["pending_params"]["distillation_column"] = (action_index, self.env_config.DF_distillation_map[action_index])
                        self.history.append(action_index)
                        self.level_list.append(self.level)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.intermediate_action_rewards.append(self.objective)
                        self.current_state['open_streams'] = self._enumerate_open_streams()

                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                        return done, self.objective, worked

                # for selecting split ratio 
                elif self._chosen_unit_name() == "split":
                    if action_index > len(self.env_config.split_ratio_map):
                        raise ValueError("Split ratio value selected more than the permissible limit.")
                    else:
                        self.level_list.append(self.level)
                        self.current_state["pending_params"]["split"] = (action_index, self.env_config.split_ratio_map[action_index])
                        self.history.append(action_index)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.intermediate_action_rewards.append(self.objective) 
                        self.current_state['open_streams'] = self._enumerate_open_streams()
                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                        return done, self.objective, worked
                    
                elif self._chosen_unit_name() == "add_solvent":
                    self.chosen_add_solvent_comp = self.env_config.component_names[action_index]
                    self.current_state["pending_params"]["add_solvent"] = {
                        "index_for_comp": action_index,
                        "name_comp": self.chosen_add_solvent_comp,
                        "index_for_amount": None,
                        "amount_value": None,
                    }
                    self.level_list.append(self.level)
                    self.history.append(action_index)

                    self.level = 3 if next_level == None else next_level
                    self.intermediate_action_rewards.append(self.objective)
                    self.get_feasible_actions()
                    return False, self.objective, True
                        
                # choose destination stream for recycle
                elif self._chosen_unit_name() == "recycle":
                    stream = self.current_state["chosen_open_stream"]
                    dests = self._eligible_recycle_destinations(stream)

                    node_ids = list(self.sim.graph.nodes)
                    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                    idx_to_id = {v: k for k, v in id_to_idx.items()}
                    if len(dests) == 0:
                        raise ValueError("No destination stream available for recycle ")
                    else:
                        # select destination and simulate 
                        if idx_to_id[action_index] in dests:
                            self.current_state["pending_params"]["recycle"] = idx_to_id[action_index] 
                            self.current_state["recycle_dest_unit"] = idx_to_id[action_index] 
                        else:
                            raise RuntimeError("Recycle requires a destination unit to be chosen.")
                        
                        self.history.append(idx_to_id[action_index])
                        self.level_list.append(self.level)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.intermediate_action_rewards.append(self.objective)
                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                        self.current_state['open_streams'] = self._enumerate_open_streams()
                        return done, self.objective, worked

                # mixer: choose second stream
                elif self._chosen_unit_name() == "mixer":
                    candidates = self._eligible_mixer_streams(self.current_state["chosen_open_stream"])
                    src_node, _ = self.current_state["chosen_open_stream"]
                    candidate_nodes = sorted({i for i, _ in candidates if i != src_node})
                    node_ids = list(self.sim.graph.nodes)
                    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                    idx_to_id = {v: k for k, v in id_to_idx.items()}
                    
                    if idx_to_id[action_index] in candidate_nodes:
                        self.current_state["second_open_stream_dest_node"] = idx_to_id[action_index] 
                        
                    if self.current_state["second_open_stream_dest_node"] is None:
                        raise RuntimeError("Selection for mixer node not valid")
                    
                    self.level_list.append(self.level)
                    self.history.append(idx_to_id[action_index]) 
                    self.intermediate_action_rewards.append(self.objective)
                    self.level = 3 if next_level == None else next_level
                    self.get_feasible_actions()
                    return False, self.objective, True

            elif self.level == 3:
                # select amount for the select parameter of add solvent 
                if self._chosen_unit_name() == "add_solvent":
                    _, component, _, _ = self.current_state["pending_params"]["add_solvent"].values()
                    if component:
                        selected_amount = self.env_config.add_solvent_comp_map[component][action_index]
                        self.current_state["pending_params"]["add_solvent"]["index_for_amount"] = action_index
                        self.current_state["pending_params"]["add_solvent"]["amount_value"] = selected_amount
                        self.history.append(action_index)
                        #print(f"solvent: {component}, solvent_amount: {selected_amount}")

                        self.level_list.append(self.level)
                        done, reward, worked = self._complete_action_place_and_simulate()
                        self.current_state["completed_design"] = done 
                        self.objective = reward
                        self.intermediate_action_rewards.append(self.objective)
                        self.level = 0 if next_level == None else next_level
                        self.get_feasible_actions()
                
                elif self._chosen_unit_name() == "mixer":
                    index_value = self.current_state["second_open_stream_dest_node"] 
                    all_candidates = self._enumerate_open_streams_excluding(exclude = self.current_state["chosen_open_stream"])
                    src_node, _ = self.current_state["chosen_open_stream"]
                    true_cands = []
                    for i, out in all_candidates:
                        if i == index_value and i != src_node:
                            true_cands.append((i, out))

                    out_value = None
                    for i, name in true_cands:
                        if action_index == 1 and name == 'out1':
                            out_value = 'out1'
                        elif action_index == 0 and name == 'out0':
                            out_value = 'out0'

                    if out_value is None:
                        raise RuntimeError("Selected mixer outlet is not valid for the chosen destination node.")

                    self.current_state["pending_params"]["mixer"] = (index_value, out_value)
                    self.history.append(action_index) 
                    self.current_state["second_open_stream"] = (index_value, out_value)

                    done, reward, worked = self._complete_action_place_and_simulate()
                    self.current_state["completed_design"] = done 
                    self.objective = reward
                    self.intermediate_action_rewards.append(self.objective)
                    
                    self.level_list.append(self.level)
                    self.level = 0 if next_level == None else next_level
                    self.get_feasible_actions()
                    self.current_state['open_streams'] = self._enumerate_open_streams()
 
                else:
                    if self._chosen_unit_name() not in ["add_solvent", "mixer"]:
                        raise RuntimeError("Only add_solvent or mixer related decisions allowed on Level 3")
                
                return done, self.objective, worked
            

        except Exception as e:
            # failed move (e.g., PEQ fail, convergence fail)
            print("take_action error:")
            print("  exception type:", type(e))
            print("  exception value:", e)
            traceback.print_exc()
            # reset to prevent level loops
            self._reset_action_state()
            self._rollback_action_history()
            return False, self.objective, False

        return False, self.objective, True

    # ------------- internals -------------#

    def valid_units_dest_for_stream(self, stream) -> List:
        """Return the unit types that can legally consume this exact open stream."""
        return [
            unit_name
            for unit_name in self.env_config.units_map_indices_type
            if self._unit_available(unit_name, stream=stream)
        ]

    def _unit_available(self, unit_name: str, stream = None) -> bool:
        input_stream = stream if stream is not None else self.current_state["chosen_open_stream"]
        if input_stream is None:
            return False

        if unit_name != 'recycle':
            if self.total_units_placed >= self.max_total_units:
                return False

        cap_map = {
            "distillation_column": self.max_distillation_columns,
            "decanter": self.max_decanters,
            "split": self.max_split,
            "mixer": self.max_mixer,
            "recycle": self.max_recycle,
            "add_solvent": self.max_solvent,
        }
        if unit_name in cap_map and (
            cap_map[unit_name] is not None
            and self.counts[unit_name] >= cap_map[unit_name]
        ):
            return False

        if unit_name == "mixer":
            if not self._eligible_mixer_streams(input_stream):
                return False

        if unit_name == "recycle":
            return bool(self._eligible_recycle_destinations(input_stream))
        return True

    def _eligible_mixer_streams(self, stream: Tuple[int, str]) -> List[Tuple[int, str]]:
        """Return the mixed companion streams that can legally enter a mixer."""
        return [
            candidate
            for candidate in self._enumerate_open_streams_excluding(exclude=stream) if candidate[0] != stream[0]]

    def _chosen_unit_name(self) -> Optional[str]:
        chosen = self.current_state["chosen_unit"]
        if chosen is None:
            return None
        chosen_unit_index, _ = chosen
        if chosen_unit_index is None:
            return None
        return self.env_config.units_map_indices_type[chosen_unit_index]

    def _enumerate_open_streams(self) -> List[Tuple[int, str]]:
        return self.sim.get_open_streams()

    def _enumerate_open_streams_excluding(self, exclude: Optional[Tuple[int, str]]) -> List[Tuple[int, str]]:
        all_ops = self.sim.get_open_streams()
        if exclude is None:
            return all_ops
        return [(n, l) for (n, l) in all_ops if not (n == exclude[0] and l == exclude[1])]
    

    def generate_custom_flowsheet_id (self) -> str:
        
        """
        Generates a custom identifier for a newly generated flowsheez 

        """

        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=4))
        return f"{random_chars}"

    
    def _all_units_at_max_capacity(self) -> bool:
        cap_map = {
            "distillation_column": self.max_distillation_columns,
            "decanter": self.max_decanters,
            "split": self.max_split,
            "mixer": self.max_mixer,
            "recycle": self.max_recycle,
            "add_solvent": self.max_solvent,
        }

        for unit, max_cap in cap_map.items():
            # if a unit has no cap, it never blocks termination
            if max_cap is None:
                return False

            if self.counts.get(unit, 0) < max_cap:
                return False

        return True
    
    def _node_already_has_recycle(self, stream: Tuple[int, str]) -> bool:
        node_id, outlet = stream
        for u, v, data in self.sim.graph.edges(data=True):
            if (
                u == node_id
                and data.get("output_label") == outlet
                and data.get("is_recycle", False)
            ):
                return True
        return False

    def _assert_stream_is_open(self, stream: Tuple[int, str]) -> None:
        opens = set(self._enumerate_open_streams())
        if stream not in opens:
            raise RuntimeError(f"Chosen stream {stream} is no longer open.")

    def _complete_action_place_and_simulate(self) -> Tuple[bool, float, bool]:
        
        
        """
        Place the chosen unit (or recycle), wire edges, run simulate(), compute NPV.
        Resets to level 0 when done (unless terminated).
        
        """
        chosen_unit_index, unit_name = self.current_state["chosen_unit"]
        if self.current_state["chosen_open_stream"] is None or chosen_unit_index is None or unit_name is None:
            raise RuntimeError("Action incomplete (missing stream or unit).")

        # Ensure the source stream is still open
        self._assert_stream_is_open(self.current_state["chosen_open_stream"])
        src_node, src_label = self.current_state["chosen_open_stream"]

        # Build params per unit
        params: Dict[str, Any] = {}
        created_node_id: Optional[int] = None
        snap = None
        edge_snap = None

        try:
            # Continuous param (if any)
            if unit_name == "distillation_column":
                _, cont_val = self.current_state["pending_params"]["distillation_column"]
                params["df"] = cont_val
            elif unit_name == "split":
                _, cont_val = self.current_state["pending_params"]["split"]
                params["split_ratio"] = cont_val

            elif unit_name == "add_solvent":
                index, component, index_for_amount, amount_value = self.current_state["pending_params"][
                    "add_solvent"].values()
                params = {
                    "index_new_component": index,
                    "solvent_amount": float(amount_value),
                    "component_name": component
                }

            snap = self.sim.snapshot(include_phase=(unit_name == "add_solvent"))

            # Actually place
            if unit_name == "mixer":
                index, second_o_str_name = self.current_state["pending_params"]['mixer']
                if second_o_str_name is None:
                    raise RuntimeError("Mixer requires a second open stream.")
                n2, l2 = index, second_o_str_name
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label), (n2, l2)],
                    "mixer",
                    params={},
                    num_outputs=1
                )

            elif unit_name == "recycle":
                recycle_dest = self.current_state["pending_params"]["recycle"]
                edge_snap = self.sim.snapshot_edges()
                feed_nodes = set(getattr(self.sim, "feed_nodes", []))
                if recycle_dest in feed_nodes:
                    raise ValueError(f"Cannot recycle into feed node {recycle_dest}")
                if recycle_dest is None:
                    raise RuntimeError("Recycle requires a destination unit to be chosen.")
                if recycle_dest == src_node:
                    raise ValueError("Cannot recycle a stream back into its own producing unit.")
                
                # Add recycle edge (transactional)
                self.sim.add_recycle(src_node, src_label, recycle_dest)
                created_node_id = None  # no new node

            elif unit_name == "add_solvent":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "add_solvent",
                    params=params,
                    num_outputs=1
                )

            elif unit_name == "distillation_column":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "distillation_column",
                    params=params,
                    num_outputs=2
                )

            elif unit_name == "decanter":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "decanter",
                    params={},
                    num_outputs=2
                )

            elif unit_name == "split":
                created_node_id = self.sim.add_unit(
                    [(src_node, src_label)],
                    "split",
                    params=params,
                    num_outputs=2
                )

            else:
                raise ValueError(f"Unknown unit type: {unit_name}")

            # simulate + NPV
            self.sim.simulate()
            self.failed_simulator_call = 0
            if unit_name != 'recycle': 
                self.intermediate_rewards.append(self.sim.current_net_present_value_normed)

        except Exception as e:
            print("Placement/simulation failed:", e)
            print(f"{self.problem_instance['feed_situation_index'], self.problem_instance['list_feed_streams']}")
            if unit_name == "recycle" and edge_snap is not None:
                try:
                    self.sim.restore_edges(edge_snap)
                except Exception:
                    pass
            elif created_node_id is not None:
                try:
                    self.sim.remove_node_and_restore_upstream_open(created_node_id)
                except Exception:
                    pass
            if snap is not None:
                try:
                    self.sim.restore(snap)
                except Exception:
                    pass
            self._reset_action_state()
            self._rollback_action_history()
            self.failed_simulator_call += 1
            self.failed_outlet = self.current_state['chosen_open_stream']
            return False, 0.0, False

        # update counts if worked and a *new* node was placed (recycle places no node)
        if unit_name == "recycle":
            self.counts["recycle"] += 1
        else:
            self.counts[unit_name] += 1
            self.total_units_placed += 1

        # reward = normalized NPV
        self.current_state['npv_raw'] = self.sim.current_net_present_value
        self.current_state['npv_norm'] = self.sim.current_net_present_value_normed 
        reward = self.sim.current_net_present_value_normed #or 0.0
        
        return False, reward, True

    def _rollback_action_history(self):
        start = getattr(self, "_action_seq_start", 0)
        self.history = self.history[:start]
        self.level_list = self.level_list[:start]

    def _reset_action_state(self):
        """
        Reset all current state variables in case the simulation fails
        """
        self.level = 0
        self.current_state["current_level"] = 0
        self.current_state["open_streams"] = self._enumerate_open_streams()
        self.current_state["chosen_unit"] = None
        self.current_state["chosen_open_stream"] = None
        self.current_state["pending_params"] = {
            "distillation_column": None,
            "split": None,
            "recycle": None,
            "add_solvent": {
                "index_for_comp": None,
                "name_comp": None,
                "index_for_amount": None,
                "amount_value": None,
            },
            "mixer": None,
        }
        self.current_state["npv_raw"] = None
        self.current_state["npv_norm"] = None
        self.current_state["completed_design"] = False
        self.current_state["second_open_stream_dest_node"] = None
        self.current_state["second_open_stream"] = None
        self.current_state["recycle_dest_unit"] = None
        self.current_state["current_action_mask"] = None
        self.get_feasible_actions()


    def _eligible_recycle_destinations(self, stream = None) -> List[int]:
        """
        Return the exact list of destination unit node_ids that are legal for the
        currently chosen open stream:
          - must have available input capacity per simulator
          - must not be a feed
          - must not be the origin (producer) unit of the chosen stream
          - if env_config.allow_forward_recycles is False, destination must be
            upstream/older than the source node, i.e. dest_node_id < origin_node_id
        """
        if stream is None:
            stream = self.current_state["chosen_open_stream"]
        if stream is None:
            return []

        origin_node_id = stream[0]
        dests = list(self.sim.get_units_with_available_input(exclude=origin_node_id, max_inputs=2))

        # Cannot recycle into feeds.
        feed_nodes = set(getattr(self.sim, "feed_nodes", []))
        dests = [nid for nid in dests if nid not in feed_nodes]

        # Optional true-recycle restriction: block forward pseudo-recycles such as
        # split out1 -> later unit that was already fed by split out0.
        if not bool(getattr(self.env_config, "allow_forward_recycles", True)):
            dests = [nid for nid in dests if nid < origin_node_id] 

        if self._node_already_has_recycle(stream):
            return []

        return [node_id for node_id in dests]

    def compute_recycler_masks(flowsheets, device: torch.device = None) -> torch.Tensor:

        """
        Make attention masks for recycler expert unit. 
        0 = not allowed, 1 = allowed 

        mask: torch.tensor of shape (B, max_nodes, max_nodes)

        """

        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        mask_device = torch.device(device or flowsheets[0].gen_config.training_device)
        attention_mask = torch.zeros(len(flowsheets), max_nodes, max_nodes, device=mask_device)

        for fs_num, fs in enumerate(flowsheets):
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            N = len(node_ids)

            # chosen source node id (graph-space)
            if fs.current_state["chosen_open_stream"] == None: 
                continue 
            else:
                source_node = fs.current_state["chosen_open_stream"][0]
                if source_node not in id_to_idx:
                    continue
                src_idx = id_to_idx[source_node]

            # candidate destination node ids (graph-space)
            cand_node_ids = fs._eligible_recycle_destinations(stream=fs.current_state["chosen_open_stream"])
            if len(cand_node_ids) == 0:
                continue
        
            for dst_node_id in cand_node_ids:
                attention_mask[fs_num, src_idx, id_to_idx[dst_node_id]] = 1
        
        return attention_mask

    
    def compute_mixer_masks(flowsheets, device: torch.device = None) -> torch.Tensor:

        """
        Make attention masks for mixer expert unit. 
        0 = not allowed, 1 = allowed 
        mask: torch.tensor of shape (B, max_nodes, max_nodes)

        """
        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        mask_device = torch.device(device or flowsheets[0].gen_config.training_device)
        attention_mask = torch.zeros(len(flowsheets), max_nodes, max_nodes, device=mask_device)

        for fs_num, fs in enumerate(flowsheets):
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

            src_node = fs.current_state["chosen_open_stream"] 
            if src_node is None:
                continue

            src_node_id, _ = src_node  
            src_idx = id_to_idx[src_node_id]

            for dst_node_id, _ in fs._eligible_mixer_streams(fs.current_state["chosen_open_stream"]):
                dst_idx = id_to_idx[dst_node_id]
                attention_mask[fs_num, src_idx, dst_idx] = 1
        
        return attention_mask
    
    def masked_log_probs_for_current_action_level(self, logits: np.ndarray) -> np.ndarray:
        
        """
        Apply current_action_mask to logits and return normalized log-probs.
        """

        mask = self.current_action_mask.astype(bool)
        logits = logits.copy()
        logits[~mask] = -np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log(softmax(logits))

        return log_probs

    def forced_termination(self):
        unit_budget_full = self.total_units_placed >= self.max_total_units 
        self.recycle_budget_full = self.counts["recycle"] >= self.max_recycle
        no_capacity_left = self._all_units_at_max_capacity() and self.recycle_budget_full
        too_many_failures = (self.failed_simulator_call >= self.env_config.max_simulator_tries)
        too_many_recycle_edge_failures = (self.recycle_edge_fail >= 5)

        return self.level == 0 and (too_many_failures or no_capacity_left or too_many_recycle_edge_failures or (unit_budget_full and self.recycle_budget_full))
    
    def optional_termination(self) -> bool:
        good_enough = self.objective >= self.env_config.min_obj_for_terminate
        return self.level == 0 and self.total_units_placed >= self.min_total_units and good_enough

    
    def is_terminable(self):
        
        return self.forced_termination() or self.optional_termination()
    

    @staticmethod
    def get_open_stream_mask_padded(flowsheets: List['FlowsheetDesign'], device: torch.device = None):
        
        batch_open_streams = []
        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        batch_node_outlet_masks = []

        for fs in flowsheets:
            open_streams = fs.current_state["open_streams"]
            batch_open_streams.append(open_streams)
        
        
        for streams, fs in zip(batch_open_streams, flowsheets):
            
            #canonical node ordering
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            outlet_mask = torch.zeros(
                max_nodes, 2, dtype=torch.bool,
                device=device or flowsheets[0].gen_config.training_device,
            )

            for node_id, outlet in streams:
                outlet_mask[id_to_idx[node_id], fs.env_config.outlet_to_idx[outlet]] = True
            
            batch_node_outlet_masks.append(outlet_mask)

        return torch.stack(batch_node_outlet_masks, dim=0)


    @staticmethod
    def get_raw_state_batch(flowsheets: List['FlowsheetDesign'], device: torch.device = None):
        """Return padded simulator state without policy-generated latents."""
        assert flowsheets, "Empty batch of flowsheets"
        env_config = flowsheets[0].env_config
        batch_device = torch.device(device or flowsheets[0].gen_config.training_device)
        batch_size = len(flowsheets)
        max_nodes = max(fs.sim.graph.number_of_nodes() for fs in flowsheets)
        num_components = env_config.max_number_of_components
        max_outlets = env_config.max_outlets
        max_parallel_edges = 1
        for fs in flowsheets:
            edge_counts = {}
            for source, destination, _, _ in fs.sim.graph.edges(keys=True, data=True):
                pair = (source, destination)
                edge_counts[pair] = edge_counts.get(pair, 0) + 1
            if edge_counts:
                max_parallel_edges = max(max_parallel_edges, max(edge_counts.values()))
        unit_to_idx = {name: i for i, name in enumerate(env_config.units_map_indices_type)}

        component_properties = torch.zeros(batch_size, num_components, 3, dtype=torch.float32, device=batch_device)
        component_interactions = torch.zeros(batch_size, num_components, num_components, 1, dtype=torch.float32, device=batch_device)
        node_output_flows = torch.zeros(batch_size, max_nodes, max_outlets, num_components, dtype=torch.float32, device=batch_device)
        node_unit_types = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=batch_device)
        node_continuous_params = torch.zeros(batch_size, max_nodes, 2, dtype=torch.float32, device=batch_device)
        edge_exists = torch.zeros(batch_size, max_nodes, max_nodes, max_parallel_edges, dtype=torch.bool, device=batch_device)
        edge_recycle = torch.zeros(batch_size, max_nodes, max_nodes, max_parallel_edges, dtype=torch.bool, device=batch_device)
        edge_outlet_indices = torch.zeros(batch_size, max_nodes, max_nodes, max_parallel_edges, dtype=torch.long, device=batch_device)
        original_feed_composition = torch.zeros((batch_size, len(env_config.component_names)), device=batch_device)
        original_feed_items = torch.zeros((batch_size, len(env_config.component_names)), dtype=torch.float32, device=batch_device)
        valid_add_solvent_comps =  torch.zeros((batch_size, len(env_config.component_names)), dtype =torch.bool, device=batch_device)
        valid_nodes = torch.zeros(batch_size, max_nodes + 1, dtype=torch.bool, device=batch_device)
        valid_nodes[:, 0] = True

        #intermediate rewards 
        rewards = torch.zeros(batch_size, max_nodes, dtype=torch.float32, device=batch_device)
        

        for batch_idx, fs in enumerate(flowsheets):
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
            valid_nodes[batch_idx, 1:len(node_ids) + 1] = True
            feed_nodes = [node_id for node_id in node_ids if fs.sim.graph.nodes[node_id]["unit_type"] == "feed"]

            # add intermediate rewards 
            n = len(fs.intermediate_rewards)
            rewards[batch_idx, :n] = torch.as_tensor(fs.intermediate_rewards, dtype=torch.float32, device=batch_device)

            if feed_nodes:
                # capture what components are present in the feed for feed context
                indices = torch.tensor(fs.problem_instance['indices_components_in_feeds'], dtype=torch.long,device=batch_device)
                original_feed_items[batch_idx, indices] = 1.0

                # capture the feed concentration
                feed_conc = torch.tensor(fs.problem_instance['list_feed_streams'][0], dtype=torch.float32, device=batch_device)
                original_feed_composition[batch_idx, indices] = feed_conc[:len(indices)]

                # mask for add solvent model 
                avail_solv_indices = torch.tensor(fs.problem_instance['possible_ind_add_comp'],dtype=torch.long,device=batch_device)
                valid_add_solvent_comps[batch_idx, avail_solv_indices] = True

                feed_params = fs.sim.graph.nodes[feed_nodes[0]].get("params", {})
                metadata = getattr(fs.sim, "system_metadata", {})
                properties = np.asarray(feed_params.get("system_pure_crit", metadata.get("pure_critical", [])), dtype=np.float32).reshape(-1, 3)

                # Normalize component pure_critical properites 
                component_reference = env_config.components_tensor.detach().float().cpu()
                property_mean = component_reference.mean(dim=0).numpy()
                property_std = component_reference.std(dim=0).clamp_min(1e-6).numpy()
                valid_mask = np.any(properties != 0, axis=1) # do not normalize [0.0 0.0 0.0]

                normalized = np.zeros_like(properties)
                normalized[valid_mask] = (properties[valid_mask] - property_mean) / property_std

                component_properties[batch_idx, :min(num_components, len(normalized))] = (torch.as_tensor(normalized[:num_components], device=batch_device))
                gamma = np.asarray(feed_params.get("system_gammas_inf", metadata.get("gamma_inf", [])),dtype=np.float32,).reshape(-1)
                
                # Normalize gamma interaction parameters before ending to Flow and Component experts 
                normalized_gamma = np.zeros_like(gamma)
                valid_gamma_mask = gamma > 0
                normalized_gamma[valid_gamma_mask] = (np.log(gamma[valid_gamma_mask]) - env_config.gamma_mean) / env_config.gamma_std

                gamma_matrix = np.zeros((num_components, num_components), dtype=np.float32)
                gamma_idx = 0
                for i in range(num_components):
                    for j in range(i + 1, num_components):
                        if gamma_idx + 1 < len(gamma):
                            gamma_matrix[i, j] = normalized_gamma[gamma_idx]
                            gamma_matrix[j, i] = normalized_gamma[gamma_idx + 1]
                        gamma_idx += 2
                component_interactions[batch_idx] = torch.as_tensor(gamma_matrix[..., None], device=batch_device)

            for node_id, node_data in fs.sim.graph.nodes(data=True):
                node_idx = id_to_idx[node_id]
                unit_name = node_data["unit_type"]

                if unit_name != "feed":
                    node_unit_types[batch_idx, node_idx] = unit_to_idx[unit_name]
                    params = node_data.get("params", {})
                    node_continuous_params[batch_idx, node_idx, 0] = float(params.get("df", 0.0))
                    node_continuous_params[batch_idx, node_idx, 1] = float(params.get("split_ratio", 0.0))
                for outlet, outlet_idx in env_config.outlet_to_idx.items():
                    raw_values = node_data.get("output_flows", {}).get(outlet, [])
                    if raw_values is None:
                        continue
                    values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
                    if len(values):

                        # normalize flow with max permission solvent amount 
                        values = values / fs.env_config._amount_grid[-1]
                        node_output_flows[batch_idx, node_idx, outlet_idx, :min(num_components, len(values))] = torch.as_tensor(values[:num_components], device=batch_device)

            edge_slots = {}
            for source, destination, _, edge_data in fs.sim.graph.edges(keys=True, data=True):
                source_idx = id_to_idx[source]
                destination_idx = id_to_idx[destination]
                slot = edge_slots.get((source_idx, destination_idx), 0)
                edge_slots[(source_idx, destination_idx)] = slot + 1
                edge_exists[batch_idx, source_idx, destination_idx, slot] = True
                edge_recycle[batch_idx, source_idx, destination_idx, slot] = bool(edge_data.get("is_recycle", False))
                edge_outlet_indices[batch_idx, source_idx, destination_idx, slot] = env_config.outlet_to_idx.get(edge_data.get("output_label", "out0"), 0)

        return {
            "batch_component_properties": component_properties,
            "batch_component_interactions": component_interactions,
            "batch_node_output_flows": node_output_flows,
            "batch_node_unit_types": node_unit_types,
            "batch_node_continuous_params": node_continuous_params,
            "batch_edge_exists": edge_exists,
            "batch_edge_recycle": edge_recycle,
            "batch_edge_outlet_indices": edge_outlet_indices,
            "valid_nodes": valid_nodes,
            "batch_original_feed_composition":original_feed_composition,
            "valid_avail_solvent_mask": valid_add_solvent_comps, 
            "original_feed_items": original_feed_items,
            "open_stream_mask": FlowsheetDesign.get_open_stream_mask_padded(flowsheets, device=batch_device),
            "rewards": rewards
        }
    
    
    # ---- Implementation of abstract methods from `BaseTrajectory`
    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        copied_fs = copy.deepcopy(self)
        copied_fs.take_action(action, None)
        return copied_fs, copied_fs.current_state["completed_design"]

    
    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            raise ValueError("Objective is `None`. Check if Flowsheet Simulator really works")
        return self.objective
    
    @staticmethod
    def log_probability_fn(trajectories: List['FlowsheetDesign'], network: nn.Module, device: torch.device, gen_config= None, 
                           env_config = None) -> List[np.array]:
        
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.

        """
        log_probs_to_return: List[np.array] = []
        device = torch.device("cpu") if device is None else device
        network.eval()
        with torch.no_grad():
            with torch.amp.autocast(enabled=False, device_type=gen_config.training_device):
                batch = FlowsheetDesign.list_to_batch(flowsheets=trajectories, device=network.device, env_config=env_config)
                lvl_0_logits, unit_predictions = network(batch)
                padded_open_stream_masks = batch['open_stream_mask']
                valid_nodes = batch['valid_nodes']
                for i, fs in enumerate(trajectories):
                    # get logits for this sequence and corresponding level
                    if fs.level == 0:
                        terminate_logits_per_batch = lvl_0_logits['terminate_logits'][i, :]
                        open_stream_logits_per_batch= lvl_0_logits['open_stream_logits'][i, :, :]
                        open_stream_valid_logits = open_stream_logits_per_batch[padded_open_stream_masks[i]] #isolate non padded nodes 
                        logits = torch.cat([terminate_logits_per_batch, open_stream_valid_logits],dim=0)
                        logits = np.array(logits.float().cpu())
                    
                    if fs.level == 1:
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        node_ids = list(fs.sim.graph.nodes)
                        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
                        node_id = id_to_idx[node_id]

                        # collect logits for units corresponding to ONLY this open stream 
                        logits = []
                        for _, unit_name in enumerate(fs.env_config.units_map_indices_type):
                            logit = unit_predictions[unit_name]["picked_logit"][i, node_id, 0] # scalar logit per unit for ith batch, at node node_id and removing any singleton dimension
                            logits.append(logit.float().detach().cpu())
                        logits = np.array(logits) # (length of all possible units, _)
                    
                    if fs.level == 2: # extract parameter predictions
                        node_ids = list(fs.sim.graph.nodes)
                        id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
                        _, unit_name = fs.current_state["chosen_unit"]
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        node_id = id_to_idx[node_id]

                        logits = []
                        if unit_name == "distillation_column":
                            logits = unit_predictions[unit_name]["distillate_fraction_categorical"][i, node_id, :]
                        if unit_name == "mixer":
                            target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                            logits = target_scores[valid_nodes[i][1:]]
                        if unit_name == "recycle":
                            target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                            logits = target_scores[valid_nodes[i][1:]]
                        if unit_name == "split":
                            logits = unit_predictions[unit_name]["split_ratio_categorical"][i, node_id, :]

                        if unit_name == "add_solvent":
                            logits = unit_predictions[unit_name]["component_logit"][i, node_id, :] 
                        
                        logits = np.array(logits.float().detach().cpu())

                    if fs.level == 3:
                        logits = []
                        node_ids = list(fs.sim.graph.nodes)
                        id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
                        _, unit_name = fs.current_state["chosen_unit"]
                        node_id, _ = fs.current_state["chosen_open_stream"]
                        node_id = id_to_idx[node_id]

                        if unit_name == "add_solvent":
                            index_comp, comp_name, _, _ = fs.current_state["pending_params"]["add_solvent"].values()
                            logits = unit_predictions[unit_name]["component_amount"][i, node_id, index_comp, :]
                        elif unit_name == "mixer":
                            dest_node = fs.current_state["second_open_stream_dest_node"]
                            outlet_logits = unit_predictions[unit_name]["destinate_node_outlets"][i, id_to_idx[dest_node], :]
                            logits = outlet_logits

                        logits = np.array(logits.float().detach().cpu())
                    
                    log_probs_to_return.append(fs.masked_log_probs_for_current_action_level(logits))
        return log_probs_to_return
    
    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
        Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}
    
    @staticmethod
    def design_flowsheets(random_instance: Dict[str, Any], gen_config, env_config) -> List['FlowsheetDesign']:

        """
        Returns list of flowsheet designs based on a starting problem instance 

        """
        instance_list = []
        flowsheet_traj = FlowsheetDesign(random_instance, gen_config, env_config)
        #instance_list.append(flowsheet_traj)
        return flowsheet_traj
    
    @staticmethod
    def list_to_batch(flowsheets: List['FlowsheetDesign'], include_feasibility_masks: bool = False, device: torch.device = None, 
                      env_config = None) -> dict:
        
        """
        Given a list of flowsheet graphs, prepares a batch that can be passed through the network.

        The batch is given as a dictionary with the following keys and values:
        * "level_idx": tensor containing information about the current level that the agent is on 
        * "embedding_dict": a dict containing stacked tensors of node and edge embeddings, as processed from unit/open_stream experts and masks for open_stream and valid nodes 
        * "recycler_masks": stacked tensors of padded recycler_masks, which will be passed to recycle unit expert
        * "mixer_masks": stacked tensors of padded mixer_masks, which will be passed to mixer unit expert

        if `include_feasibility_masks` is set to True, we also return feaisbility masks 
        """

        assert len(flowsheets) > 0, "Empty batch of flowsheets"

        raw_state = FlowsheetDesign.get_raw_state_batch(flowsheets=flowsheets, device=device)

        batch_device = raw_state["batch_node_output_flows"].device
        recycler_masks = FlowsheetDesign.compute_recycler_masks(flowsheets=flowsheets, device=batch_device)
        mixer_masks = FlowsheetDesign.compute_mixer_masks(flowsheets=flowsheets, device=batch_device)

        # levels info
        batch_levels_idx = [fs.level for fs in flowsheets]

        return_dict = dict(
            **raw_state,
            recycler_masks = recycler_masks,        
            mixer_masks = mixer_masks, 
            levels = torch.tensor(batch_levels_idx, dtype=torch.long, device=device), 
        )

        if include_feasibility_masks:

            # Build per-level feasibility masks, padded across the batch to each level's max action count.
            feasibility_mask_per_level = []

            # False will allow that action, true masks it 

            # feasibility masks are made in such a way:
            # lvl 0: num of open stream + terminate action 
            # lvl 1: num of units 
            # lvl 2: for DF and splits -> len of corresponding DF/split ratio maps (100), for mixers and recycles -> num of nodes in the graph, for add_solvent -> num of available components
            # lvl 3: for mixers -> num of available outlets for the chosen stream (2), for add_solvent -> len of corresponding solvent quantity maps

            num_actions_per_level_and_flowsheet = [
                [len(fs.current_state['open_streams']) + 1 for fs in flowsheets],  # lvl 0 
                [fs.env_config.num_units for fs in flowsheets],  # lvl 1
                
                [len(env_config.DF_distillation_map) if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] in ('distillation_column', 'split')
                else len(fs.sim.graph.nodes) if fs.current_state['chosen_unit'] != None and fs.current_state['chosen_unit'][1] in ('recycle', 'mixer')
                else len(env_config.component_names) if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else len(env_config.DF_distillation_map) for fs in flowsheets], #lvl 2 

                [env_config.max_outlets if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'mixer' 
                else len(env_config._amount_grid) for fs in flowsheets],   # lvl 3
                
            ]

            for lvl, num_actions_per_fs in enumerate(num_actions_per_level_and_flowsheet):
                max_num_actions = max(num_actions_per_fs)
                max_num_actions = len(env_config.DF_distillation_map) if lvl == 3 and max_num_actions == 0 else max_num_actions
                feasibility_mask_per_level.append(
                torch.from_numpy(
                    np.stack([
                        np.pad(
                            ~fs.current_action_mask.astype(bool),
                            (0, max_num_actions - fs.current_action_mask.shape[0]),
                            mode='constant', constant_values=True
                        ) if fs.level  == lvl 
                        else np.ones(max_num_actions, dtype=bool)
                        for i, fs in enumerate(flowsheets)
                    ])
                ).bool().to(device)
            )
        
            # Add to return_dict
            return_dict["feasibility_mask_level_zero"] = feasibility_mask_per_level[0] 
            return_dict["feasibility_mask_level_one"] = feasibility_mask_per_level[1]  
            return_dict["feasibility_mask_level_two"] = feasibility_mask_per_level[2]  
            return_dict["feasibility_mask_level_three"] = feasibility_mask_per_level[3]  

        return return_dict
