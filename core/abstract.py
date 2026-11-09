import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Union, Tuple

Instance = Any  # Type alias. An instance is any initial representation of a problem instance, like a dictionary.


class BaseTrajectory(ABC):
    """
    Abstract trajectory representing a partial (and eventually, finished ;) ) episode of a problem instance.
    """
    @staticmethod
    @abstractmethod
    def init_batch_from_instance_list(config, instances: List[Instance], network: torch.nn.Module, device: torch.device):
        """
        Takes a list of problem instances and returns a list of `BaseTrajectory`.

        Parameters:
            config: Configuration object
            instances [List[Instance]]: List of problem instances. Should be copies (!) as underlying memory might be
                shared.
            network [torch.nn.Module]: Policy/encoding network which might be used to encode the instance
                (e.g., this is needed for LEHD networks to encode the problem).
            device [torch.device]: CPU/GPU Device on which to store tensors for fast access.
        """
        pass

    @staticmethod
    @abstractmethod
    def log_probability_fn(trajectories: List['BaseTrajectory'], network: torch.nn.Module) -> List[np.array]:
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays of length num_actions OR a torch tensor
        of shape (num_trajectories, num_actions), where each numpy array/tensor is a log-probability
        distribution over the next actions.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the logits for i-th trajectory.
        """
        pass

    @abstractmethod
    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """
        A function that takes the idx of an action and returns a tuple (BaseTrajectory, bool), where the first entry
        is a new trajectory obtained by executing the action on the trajectory. You must make sure that executing the
        action does NOT alter the trajectory itself, so it's safest to make a copy first.
        The second entry `is_finished` is a bool variable indicating whether the new trajectory has reached the end
        of the episode or not.
        """
        pass

    @abstractmethod
    def to_max_evaluation_fn(self) -> float:
        """
        This method is called only on finished trajectories and it should return an objective which is to _maximize_.
        E.g., for routing problems, this should return the negative tour length.
        """
        pass

    @abstractmethod
    def num_actions(self) -> int:
        """
        Returns number of current _feasible_ actions.
        """
        pass