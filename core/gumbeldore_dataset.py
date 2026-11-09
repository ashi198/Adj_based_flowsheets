import copy, os, pickle, sys, ray, torch 
from model.policy_arch import FlowsheetNetwork
from environment.environment_actions_graphs import FlowsheetDesign
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm
from core.abstract import Instance
import core.stochastic_beam_search as sbs
from typing import List, Tuple, Any, Optional
from core.incremental_sbs import IncrementalSBS
from utils import dump_top_flowsheets_txt, process_test_results
from collections import defaultdict

os.environ["RAY_DEDUP_LOGS"] = "0"

@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results
    

class GumbeldoreDataset:
    def __init__(self, gen_config, env_config, train_instances):
        self.gen_config = gen_config
        self.gumbeldore_config = gen_config.gumbeldore_config
        self.env_config = env_config
        self.devices_for_workers: List[str] = self.gumbeldore_config["devices_for_workers"],
        self.train_instances = train_instances 

    def generate_dataset(self, network_weights: dict, best_objective: Optional[float] = None, memory_aggressive: bool = False, if_test: bool= None, if_eval: bool = None, 
                         epoch: int = None):
        
        """
        Parameters:
            network_weights: [dict] Network weights to use for generating data.
            memory_aggressive: [bool] If True, IncrementalSBS is performed "memory aggressive" meaning that
                intermediate states in the search tree are not stored after transitioning from them, only their
                policies.
        """

        batch_size_gpu, batch_size_cpu = (self.gumbeldore_config["batch_size_per_worker"],
                                            self.gumbeldore_config["batch_size_per_cpu_worker"])

        problem_instances = []
        train_instances_search = self.train_instances 
        for instance in train_instances_search:
            problem_instance = FlowsheetDesign.design_flowsheets(instance, self.gen_config, self.env_config)
            problem_instances.append(problem_instance)

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.gumbeldore_config["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        future_tasks = [
            async_sbs_worker.remote(
                self.gen_config, self.env_config, job_pool, network_weights, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                cpu_cores[i], best_objective, memory_aggressive
            )
            for i, device in enumerate(self.devices_for_workers)
        ] 

        with tqdm(total=len(problem_instances)) as progress_bar:
            while True:
                # Check if all workers are done. If so, break after this iteration
                do_break = len(ray.wait(future_tasks, num_returns=len(future_tasks), timeout=0.5)[1]) == 0
                fetched_results = ray.get(job_pool.fetch_results.remote()) 
                for (i, result) in fetched_results:
                    results[i] = result
                if len(fetched_results):
                    progress_bar.update(len(fetched_results))
                if do_break:
                    break

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        self.calculate_relative_advantages(results)

        if if_test: 
            process_test_results(problem_instances, results, self.gen_config.results_path, epoch, if_test=True)
        elif if_eval:
            return self.process_results(problem_instances=problem_instances, results=results, destination_path=self.gen_config.results_path, epoch=epoch,  if_evaluate=True)     
        else:
            return self.process_results(problem_instances, results, self.gen_config.results_path, epoch)


    def update_replay_buffer(self, instances, destination_path):

        replay_path = f"{destination_path}/replay_buffer.pickle"

        replay_min_obj = self.gumbeldore_config.get("replay_min_obj", 0.80)

        replay_top_k = self.gumbeldore_config.get("replay_top_k_per_instance", 5)

        # Load existing replay buffer
        if os.path.isfile(replay_path):
            with open(replay_path, "rb") as f:
                replay_fs = pickle.load(f)
        else:
            replay_fs = []

        # Only VERY GOOD candidates from this iteration
        new_replay_candidates = [fs for fs in instances if fs["obj"] >= replay_min_obj]
        replay_fs.extend(new_replay_candidates)

        # Group by actual problem/feed instance
        grouped = defaultdict(list)

        for fs in replay_fs:
            key = (fs["problem_instance"]["feed_situation_index"], tuple(np.round( fs["problem_instance"]["list_feed_streams"][0],6)))
            grouped[key].append(fs)

        # Keep only strongest trajectories per instance
        cleaned_replay = []
        for key, candidates in grouped.items():
            candidates = sorted(candidates, key=lambda x: x["obj"], reverse=True)
            cleaned_replay.extend(candidates[:replay_top_k])

        # Save
        with open(replay_path, "wb") as f:
            pickle.dump(cleaned_replay, f)

        print(f"[Replay buffer], " f"{len(new_replay_candidates)} new high-quality trajectories, " f"{len(cleaned_replay)} stored total.")

        return replay_path

 
    def process_results(self, problem_instances, results, destination_path, epoch, if_test=False, if_evaluate=False):
            

            """
            Processes the results from Gumbeldore search and save it to a pickle. Each trajectory will be represented as a dict with the
            following keys and values
            "action_seq": List[List[int]] Actions which need to be taken on each index to create the molecule
            "obj": [float] NPV value

            Then:
            1. If the dataset already exists at the path where to save, we load it, merge them and take the best from the
                merged dataset.

            Then returns the following dictionary:
            - "mean_gen_obj": Mean generated obj. -> over the unmerged best flowsheets generated
            - "best_gen_obj": Best generated obj. -> Best obj. of the unmerged flowsheets generated
            - "worst_gen_obj": Worst generated obj. -> Worst obj. of the unmerged flowsheets generated
            """

            metrics_return = dict()
            per_feed_index = {}
            instances = []
            best_generated_objs = []
            wor_generated_objs = []
            best_per_instance_flowsheets = []

            for i, _ in enumerate(problem_instances):
                per_instances = [] 
                for flowsheet in results[i]: 
                    if flowsheet.objective > float("-inf"):
                        per_instances.append(dict(
                            problem_instance = flowsheet.problem_instance,
                            identifier = flowsheet.identifier, 
                            action_seq=flowsheet.history,
                            obj=flowsheet.objective,
                            graph = flowsheet.sim.graph, 
                            total_units_placed = flowsheet.total_units_placed,
                            levels = flowsheet.level_list,
                            npv_raw = flowsheet.sim.current_net_present_value,
                            npv_normed = flowsheet.sim.current_net_present_value_normed,
                            per_ratio = flowsheet.sim.performance_ratio,
                            npv_wo_app_cost = flowsheet.sim.npv_without_app_cost,
                            subsystem = flowsheet.problem_instance['system_name'],
                            advantage = flowsheet.advantage, 
                            intermediate_rewards = flowsheet.intermediate_action_rewards, 

                        ))
                per_instances.sort(key=lambda x: x["obj"], reverse=True)
                best_instance_fs = per_instances[0]
                best_per_instance_flowsheets.append(best_instance_fs)
                instances.extend(per_instances)
                key = (results[i][0].problem_instance["feed_situation_index"], tuple(np.round(results[i][0].problem_instance["list_feed_streams"][0], 6)))
                per_feed_index[key] = per_instances
                if if_evaluate: 
                    best_obj = np.mean([x['obj'] for x in per_instances[:5]])
                    min_obj = np.mean([x['obj'] for x in per_instances[-5:]])
                else:
                    best_obj = max(x["obj"] for x in per_instances)
                    min_obj = min(x["obj"] for x in per_instances)
                best_generated_objs.append(best_obj)
                wor_generated_objs.append(min_obj)

            # Log metrics
            generated_fs = instances
            generated_fs = sorted(generated_fs, key=lambda x: x["obj"], reverse=True)
            generated_objs = np.array([x["obj"] for x in generated_fs]) 
            metrics_return["mean_gen_obj"] = generated_objs.mean()
            metrics_return["best_gen_obj"] = np.mean(best_generated_objs)
            metrics_return["worst_gen_obj"] = np.mean(wor_generated_objs)

            # Now check if there already is a data file, and if so, load it and merge it.
            if destination_path is not None:
                if not if_evaluate:
                    destination_full_path = f"{destination_path}/generated_flowsheets.pickle"
                    destination_full_path_complete = f"{destination_path}/complete_generated_flowsheets.pickle"
                else:
                    destination_full_path = f"{destination_path}/eval_flowsheets.pickle"
                    destination_full_path_complete = f"{destination_path}/complete_eval_flowsheets.pickle"

                if if_evaluate:
                    # Current evaluation only
                    eval_fs = []
                    for key, current_instances in per_feed_index.items():
                        valid_instances = [fs for fs in current_instances if int(self.env_config.action_limits[fs["subsystem"]]["min_total_units"])
                            <= fs["total_units_placed"] <= int(self.env_config.action_limits[fs["subsystem"]]["max_total_units"])]
                        top_k = sorted(valid_instances, key=lambda x: x["obj"], reverse=True)[:self.gumbeldore_config["num_trajectories_to_keep"]]
                        eval_fs.extend(top_k)

                    with open(destination_full_path, "wb") as f:
                        pickle.dump(eval_fs, f)

                    dump_top_flowsheets_txt(eval_fs, destination_path, epoch, if_test, if_evaluate=if_evaluate)
                
                else:
                    # Load global historical reserve
                    existing_fs = []

                    if os.path.isfile(destination_full_path_complete):
                        with open(destination_full_path_complete, "rb") as f:
                            existing_fs = pickle.load(f)

                    existing_per_instance = defaultdict(list)

                    for fs in existing_fs:
                        key = (fs["problem_instance"]["feed_situation_index"], tuple(np.round(fs["problem_instance"]["list_feed_streams"][0], 6)))
                        existing_per_instance[key].append(fs)


                    # ---------------------------------------------------------
                    # For every instance sampled THIS round:
                    #
                    # old flowsheets for this instance
                    #            +
                    # newly generated flowsheets
                    #            ↓
                    #          Top-K
                    # ---------------------------------------------------------
                    expert_min_obj = self.gumbeldore_config.get("expert_min_obj", 0.30)
                    merged_fs = []

                    for key, current_instances in per_feed_index.items():
                        previous_instances = existing_per_instance.get(key, [])

                        # Best result found for this feed THIS round
                        best_current_obj = max((fs["obj"] for fs in current_instances), default=float("-inf"))

                        if best_current_obj >= expert_min_obj:
                            current_expert_candidates = [fs for fs in current_instances if fs["obj"] >= expert_min_obj]
                        else:
                            current_expert_candidates = []

                        all_instances = previous_instances + current_expert_candidates
                        valid_instances = [fs for fs in all_instances if int(self.env_config.action_limits[fs["subsystem"]]["min_total_units"]) <= fs["total_units_placed"] <= int(self.env_config.action_limits[fs["subsystem"]]["max_total_units"])]
                            
                        # Top-K over HISTORICAL + CURRENT solutions
                        top_k = sorted(valid_instances, key=lambda x: x["obj"], reverse=True)[:self.gumbeldore_config["num_trajectories_to_keep"]]
                        merged_fs.extend(top_k)
                    
                    # Save expert trajectory dataset
                    with open(destination_full_path, "wb") as f:
                        pickle.dump(merged_fs, f)

                    dump_top_flowsheets_txt(merged_fs, destination_path, epoch, if_test, if_evaluate=if_evaluate)

                    # Update elite replay buffer
                    self.update_replay_buffer(instances=instances, destination_path=destination_path)

                    # complete_generated_flowsheets.pickle:
                    # global reserve containing ALL newly sampled flowsheets
                    complete_fs = existing_fs + instances

                    # Pickle complete data
                    with open(destination_full_path_complete, "wb") as f:
                        pickle.dump(complete_fs, f)

            return metrics_return, destination_full_path
    
    '''def calculate_relative_advantages(self, results):
        eps = 1e-8
        for result in results:
            rewards = np.array([traj.objective for traj in result])
            mean = rewards.mean()
            std = rewards.std()
            if std < eps:
                terminal_advantage = np.zeros_like(rewards)
            else:
                terminal_advantage = (rewards - mean) / (std + eps)

            all_deltas = []
            trajectory_deltas = []

            for traj in result:
                rewards = np.asarray(
                    traj.intermediate_action_rewards,
                    dtype=np.float32
                )

                deltas = np.diff(rewards)
                trajectory_deltas.append(deltas)
                all_deltas.extend(deltas)

            all_deltas = np.asarray(all_deltas, dtype=np.float32)
            delta_mean = all_deltas.mean() if len(all_deltas) else 0.0
            delta_std = all_deltas.std() if len(all_deltas) else 0.0

            for traj, term_adv, intermed_adv in zip(result, terminal_advantage, trajectory_deltas):
                if delta_std < eps:
                    intermediate_advantages = np.zeros_like(intermed_adv)
                else:
                    intermediate_advantages = (
                        (intermed_adv - delta_mean) / (delta_std + eps)
                    )
                combined_advantages = (0.8 * term_adv + 0.2 * intermediate_advantages)
                traj.advantage = combined_advantages'''
    

    def calculate_relative_advantages(self, results):
        eps = 1e-8
        gamma = 0.99 

        for result in results:

            # terminal reward calculation 

            rewards = np.array([traj.objective for traj in result])
            mean = rewards.mean()
            std = rewards.std()
            if std < eps:
                terminal_advantage = np.zeros_like(rewards)
            else:
                terminal_advantage = (rewards - mean) / (std + eps)

            trajectory_returns = []
            all_returns = []

            # Calculate future returns for each trajectory

            for traj in result:
                rewards = np.asarray(
                    traj.intermediate_action_rewards,
                    dtype=np.float32
                )

                deltas = np.diff(rewards)
                future_returns = np.zeros_like(deltas)
                running_return = 0.0

                for k in reversed(range(len(deltas))):
                    running_return = deltas[k] + gamma * running_return
                    future_returns[k] = running_return

                trajectory_returns.append(future_returns)
                all_returns.extend(future_returns)

            # Standard future rewards 
            all_returns = np.asarray(all_returns, dtype=np.float32)
            if len(all_returns) == 0:
                return_mean = 0.0
                return_std = 0.0
            else:
                return_mean = all_returns.mean()
                return_std = all_returns.std()
            
            for traj, term_adv, future_returns in zip(result, terminal_advantage, trajectory_returns):
                if return_std < eps:
                    future_advantage = np.zeros_like(future_returns)
                else:
                    future_advantage = (
                        (future_returns - return_mean) / (return_std + eps)
                    )
                combined_advantages = (0.8 * term_adv + 0.2 * future_advantage)
                traj.advantage = combined_advantages

    '''def calculate_relative_advantages(self, results):
        for result in results:
            rewards = np.array([traj.objective for traj in result])
            mean = rewards.mean()
            advantages = rewards - mean

            for traj, adv in zip(result, advantages):
                traj.advantage = adv'''

@ray.remote(max_calls=1)
def async_sbs_worker(gen_config, env_config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False
                     ):
    
    def child_log_probability_fn(trajectories: List[FlowsheetDesign]) -> [np.array]:
        return FlowsheetDesign.log_probability_fn(trajectories=trajectories, network=network, device=device, 
                                                  gen_config = gen_config, env_config=env_config)
    
    def batch_leaf_evaluation_fn(trajectories: List[FlowsheetDesign]) -> np.array:
        for traj in trajectories:
            npv = traj.objective 
            traj.objective = npv
        
        objs = [traj.objective for traj in trajectories]
        return objs

    def child_transition_fn(trajectory_action_pairs: List[Tuple[FlowsheetDesign, int]]):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]
    
    # Pin worker to core if wanted
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():

        if gen_config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = gen_config.CUDA_VISIBLE_DEVICES

        device = torch.device(device)
        network = FlowsheetNetwork(gen_config, env_config, device)
        network.load_state_dict(network_weights)
        network.to(network.device)
        network.eval()

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))

            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = [instance for _, instance in batch]

            if gen_config.gumbeldore_config["search_type"] == "beam_search":

                # Deterministic beam search.
                beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=root_nodes,
                    beam_width=gen_config.gumbeldore_config["beam_width"],
                    deterministic=True
                )
            else:
                inc_sbs = IncrementalSBS(root_nodes, child_log_probability_fn, child_transition_fn,
                                         leaf_evaluation_fn=FlowsheetDesign.to_max_evaluation_fn,
                                         batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                                         memory_aggressive=False)

                if gen_config.gumbeldore_config["search_type"] == "tasar":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_tasar(
                        beam_width=gen_config.gumbeldore_config["beam_width"],
                        deterministic=gen_config.gumbeldore_config["deterministic"],
                        nucleus_top_p=gen_config.gumbeldore_config["nucleus_top_p"],
                        replan_steps=gen_config.gumbeldore_config["replan_steps"],
                        sbs_keep_intermediate=gen_config.gumbeldore_config["keep_intermediate_trajectories"]
                    )
                elif gen_config.gumbeldore_config["search_type"] == "wor":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_incremental_sbs(
                        beam_width=gen_config.gumbeldore_config["beam_width"],
                        num_rounds=gen_config.gumbeldore_config["num_rounds"],
                        nucleus_top_p=gen_config.gumbeldore_config["nucleus_top_p"],
                        sbs_keep_intermediate=gen_config.gumbeldore_config["keep_intermediate_trajectories"],
                        best_objective=best_objective
                    )

            results_to_push = []
            for j, result_idx in enumerate(idx_list):
                result: List[FlowsheetDesign] = [x.state for x in beam_leaves_batch[j][:gen_config.gumbeldore_config["beam_width"]]]

                # Check if they need objective evaluation (this will only be true for deterministic beam search
                if result[0].objective is None:
                    batch_leaf_evaluation_fn(result)
                results_to_push.append((result_idx, result))

            ray.get(job_pool.push_results.remote(results_to_push)) 

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()
