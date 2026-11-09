from typing import Optional, Tuple, List, Dict

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import copy
from environment.environment_actions_graphs import FlowsheetDesign
from tqdm import tqdm

def _clone_sequence(fs: FlowsheetDesign) -> FlowsheetDesign:

    # Prefer a cheap custom copy, if not possible, make deep copy
    if hasattr(fs, "copy") and callable(fs.copy):
        return fs.copy()
    return copy.deepcopy(fs)

class RandomDataset(Dataset):

    """
    Dataset for supervised training of flowsheet design given as a list pseudo-expert flowsheet.
    Each flowsheet is given as a dictionary with the following keys and values
          "start_residue": [int] the int representing the residue from which to start
          "action_seq": List[List[int]] Actions which need to be taken on each index to create the flowsheet
          "smiles": [str] Corresponding flowsheet string
          "obj": [float] Objective function evaluation

    Each datapoint in this dataset is a partial flowsheet: We sample an instance, randomly choose an index up to which
    all actions will be performed. Then, ending up at action index 0, we take the next item in the action seq
    (which corresponds to a list all actions that need to be taken from index to index) as training target.
    As the number of nodes will be different for flowsheets in a batch, we pad the nodes, and set all labels corresponding
    to the padded nodes to -1 (in the CE-loss, this will be specified as `ignore_index=-1`.

    """
    def __init__(self, epoch, gen_config, env_config,  path_to_pickle: str, batch_size: int, custom_num_batches: Optional[int],
                 no_random: bool = False):
        self.epoch = epoch
        self.gen_config = gen_config
        self.env_config = env_config
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches

        '''if epoch == self.gen_config.gumbeldore_config['replay_every']:
            replay_path = f"{gen_config.results_path}/replay_buffer.pickle"
            self.path_to_pickle = path_to_pickle
            with open(self.path_to_pickle, "rb") as f:
                expert_traj = pickle.load(f)  # list of dictionaries
            with open(replay_path, "rb") as f:  
                replay_traj = pickle.load(f)  # list of dictionaries
            replay_traj = random.sample(replay_traj, k=min(80, len(replay_traj)))
            self.instances = expert_traj + replay_traj
            print("--- Training on replay buffer ---")
        else:'''
        self.path_to_pickle = path_to_pickle
        with open(path_to_pickle, "rb") as f:
            self.instances = pickle.load(f) 

        # We want to uniformly sample from partial flowsheets. So for each instance, check how many partial flowsheets
        # there are, and create a list of them where each entry is a tuple (int, int), where first entry is index of
        # the instance, and second entry is the index in the action sequence which is the training target.
        self.targets_to_sample: List[Tuple[int, int]] = []

        # Keep original tuple list: (instance_idx, target_idx)
        self.targets_to_sample: List[Tuple[int, int]] = []
        self._flat_sequences: List[FlowsheetDesign] = []   # state BEFORE taking the target action
        self._flat_targets:   List[int] = []              # the target action (int)
        self._flat_levels:    List[int] = []              # current_action_level at that state
        self._flat_advantages: List[int] = []

        # Map (instance_idx, target_idx) -> flat index in the lists above
        self._tuple_to_flat: Dict[Tuple[int, int], int] = {}

        # precompute everything once 
        for i, instance in tqdm(enumerate(self.instances), total=len(self.instances), desc="Computing dataset"):
            fs = FlowsheetDesign(instance['problem_instance'], gen_config=gen_config, env_config=env_config)
            sequence_of_actions_idx = list(range(len(instance["action_seq"])))
            self.targets_to_sample.extend([(i, j) for j in sequence_of_actions_idx])
            # For each target_idx == j, we want the sequence state *before* taking action j (i.e., after j-1 actions)
            for j, action in enumerate(instance["action_seq"]):

                flat_idx = len(self._flat_targets)
                self._tuple_to_flat[(i, j)] = flat_idx

                # snapshot current state (this corresponds to partial sequence up to j-1)
                fs_copy = _clone_sequence(fs)
                self._flat_sequences.append(fs_copy)
                self._flat_targets.append(action) 
                self._flat_advantages.append(instance['advantage'][j]) if not gen_config.if_pretrain else self._flat_advantages.append(instance['advantage'])
                fs.level = instance['levels'][j] 
                self._flat_levels.append(fs.level)
                fs.take_action(action, None)

        print(f"Loaded dataset. {len(self.instances)} sequences with a total of {len(self.targets_to_sample)} datapoints.")

        if custom_num_batches is None:
            self.length = len(self.targets_to_sample) // self.batch_size  # one item is a batch of datapoints.
        else:
            self.length = custom_num_batches

        self.no_random = no_random

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx: is not used, as we directly randomly sample a full batch from the datapoints here.

        Returns: Dictionary with keys:

        """
        partial_sequences: List[FlowsheetDesign] = []   # partial flowsheets which will become the batch
        instance_targets: List[List[int]] = []  # corresponding targets taken from the instances

        if self.no_random:
            batch_to_pick = self.targets_to_sample[idx * self.batch_size: (idx+1) * self.batch_size]
        else:
            batch_to_pick = random.choices(self.targets_to_sample, k=self.batch_size)  # with replacement

        # Map each (instance_idx, target_idx) to the precomputed flat index
        flat_indices = [self._tuple_to_flat[tup] for tup in batch_to_pick]

        # Gather precomputed partial sequences and their targets/levels
        partial_sequences = [self._flat_sequences[k] for k in flat_indices]
        instance_targets  = [self._flat_targets[k]   for k in flat_indices]
        levels_list       = [self._flat_levels[k]    for k in flat_indices]
        indices_for_tracking = [k for k in flat_indices]
        advantages = [self._flat_advantages[k] for k in flat_indices]

        # Create the input batch from the partial flowsheets.
        batch_input = FlowsheetDesign.list_to_batch(flowsheets=partial_sequences,
                                                   device=torch.device("cpu"),
                                                   include_feasibility_masks=True, env_config=self.env_config)

        # We now create the targets. We separate it into targets for level 0, 1, 2, and 3.
        # We only set the target action as target for the current level the flowsheet is in.
        # For all other levels, we set it to -1 for a flow. (ignore)
        # Vectorized level-specific targets
        targets = torch.tensor(instance_targets, dtype=torch.long)   
        levels  = torch.tensor(levels_list,      dtype=torch.long)   
        advantages = torch.tensor(advantages, dtype= torch.float16)

        batch_targets = [
            torch.where(
                levels == level, 
                targets, # # We only set the target action as target for the current level the molecule is in.
                torch.full_like(targets, -1)) # # For all other levels, we set it to -1 for a molecule.
            for level in (0, 1, 2, 3)
        ]

        samples = dict(
            input=batch_input,
            target_zero=batch_targets[0],
            target_one=batch_targets[1], 
            target_two = batch_targets[2], 
            target_three = batch_targets[3],
            advantages = advantages, 
            indices_for_tracking = indices_for_tracking, 
        )

        return samples
