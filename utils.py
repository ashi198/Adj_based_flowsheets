import os 
import numpy as np 
import torch
import pickle
import copy
from collections import defaultdict
import random
import argparse
import torch.nn.functional as F
from collections import defaultdict
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def set_mlflow_connection():
    #os.environ["AWS_ACCESS_KEY_ID"] = "minio_id"
    #os.environ["AWS_SECRET_ACCESS_KEY"] = "XaFC8sHiRHr5uQTdfVQQ"
    #os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:5001" 
    #os.environ['MLFLOW_TRACKING_USERNAME'] = "hilbert" #"fhaselbeck"
    #os.environ['MLFLOW_TRACKING_PASSWORD'] = "schneckenfunktion"
    #os.environ['GIT_PYTHON_REFRESH'] ="quiet"
    remote_server_uri = "http://0.0.0.0:5001"  # "http://10.154.6.32:5100"  # set to MLFlow server URI (host ip and PORT in .env)
    mlflow.set_tracking_uri(remote_server_uri)


def build_logit_tensors_per_level(dataset, input_data, terminate_or_open_streams_logits, unit_predictions, indices_for_tracking):

    '''

    Takes logits across the entire space and build its per level. Returns a padded tensor per level, 
    with size (num_flowsheet_dataset, max_possible_action_per_lvl)

    Args:
        dataset: derived from RandomDataset function, which contains information about all flowsheets and action sequences derived from .pickle file. 
        input_data: dict containing information about masking for valid nodes and open streams. 
        terminate_or_open_streams_logits: dict of logits across lvl0 derived from the network. 
        unit_predictions; dict of logits across lvls 1, 2, and 3 derived from the network.
        indices_for_tracking: list of indices for flowsheets that are actually loaded in the dataloader. 

    '''

    indices_for_tracking = [t.squeeze().item() for t in indices_for_tracking]
    actual_flowsheets_dataset = [dataset._flat_sequences[i] for i in indices_for_tracking]

    num_actions_per_level_and_flowsheet = [
                [len(fs.current_state['open_streams']) + 1 for fs in actual_flowsheets_dataset],  # lvl 0 
                [fs.env_config.num_units for fs in actual_flowsheets_dataset],  # lvl 1
                [len(fs.env_config.DF_distillation_map) if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] in ('distillation_column', 'split')
                else len(fs.sim.graph.nodes) if fs.current_state['chosen_unit'] != None and fs.current_state['chosen_unit'][1] in ('recycle', 'mixer')
                else 5 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else len(fs.env_config.DF_distillation_map)  for fs in actual_flowsheets_dataset], #lvl 2 
                [len(fs.env_config._amount_grid)  if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'add_solvent'
                else 2 if fs.current_state['chosen_unit'] is not None and fs.current_state['chosen_unit'][1] == 'mixer' 
                else len(fs.env_config._amount_grid) for fs in actual_flowsheets_dataset],  # lvl 3
                
            ]
    
    all_lvl_zero_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[0])).to(actual_flowsheets_dataset[0].gen_config.training_device)
    all_lvl_one_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[1])).to(actual_flowsheets_dataset[0].gen_config.training_device)
    all_lvl_two_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[2])).to(actual_flowsheets_dataset[0].gen_config.training_device)
    all_lvl_three_logits = torch.zeros(len(indices_for_tracking), max(num_actions_per_level_and_flowsheet[3])).to(actual_flowsheets_dataset[0].gen_config.training_device)

    for i, fs in enumerate(actual_flowsheets_dataset):
        
        if fs.level == 0:
            terminate_logits_fs = terminate_or_open_streams_logits['terminate_logits'][i, :]
            open_stream_logits_fs= terminate_or_open_streams_logits['open_stream_logits'][i, :, :]
            open_stream_valid_logits = open_stream_logits_fs[input_data['open_stream_mask'][i]] #isolate non padded nodes 
            lvl_zero_logits = torch.cat([terminate_logits_fs, open_stream_valid_logits],dim=0)
            all_lvl_zero_logits[i, :lvl_zero_logits.shape[0]] = lvl_zero_logits

        if fs.level == 1:
            node_id, _ = fs.current_state["chosen_open_stream"]
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            node_id = id_to_idx[node_id]

            # collect logits for units corresponding to ONLY this open stream 
            lvl_one_logits = []
            for _, unit_name in enumerate(fs.env_config.units_map_indices_type):
                logit = unit_predictions[unit_name]["picked_logit"][i, node_id, 0] 
                lvl_one_logits.append(logit)

            lvl_one_logits = torch.stack(lvl_one_logits, dim=0)
            all_lvl_one_logits[i, :lvl_one_logits.shape[0]] = lvl_one_logits

        if fs.level == 2:
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
            _, unit_name = fs.current_state["chosen_unit"]
            node_id, _ = fs.current_state["chosen_open_stream"]
            node_id = id_to_idx[node_id]

            if unit_name == "distillation_column":
                lvl_two_logits = unit_predictions[unit_name]["distillate_fraction_categorical"][i, node_id, :]
            if unit_name == "mixer":
                target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                lvl_two_logits = target_scores[input_data['valid_nodes'][i][1:]]
            if unit_name == "recycle":
                target_scores = unit_predictions[unit_name]["target_scores"][i, node_id, :]
                lvl_two_logits = target_scores[input_data['valid_nodes'][i][1:]]
            if unit_name == "split":
                lvl_two_logits = unit_predictions[unit_name]["split_ratio_categorical"][i, node_id, :]
            if unit_name == "add_solvent":
                lvl_two_logits = unit_predictions[unit_name]["component_logit"][i, node_id, :] 

            all_lvl_two_logits[i, :lvl_two_logits.shape[0]] = lvl_two_logits

        if fs.level == 3:
            lvl_three_logits = []
            node_ids = list(fs.sim.graph.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)} 
            _, unit_name = fs.current_state["chosen_unit"]
            node_id, _ = fs.current_state["chosen_open_stream"]
            node_id = id_to_idx[node_id]

            if unit_name == "add_solvent":
                index_comp, _, _, _ = fs.current_state["pending_params"]["add_solvent"].values()
                lvl_three_logits = unit_predictions[unit_name]["component_amount"][i, node_id, index_comp, :]
            elif unit_name == "mixer":
                dest_node = fs.current_state["second_open_stream_dest_node"]
                outlet_logits = unit_predictions[unit_name]["destinate_node_outlets"][i, id_to_idx[dest_node], :]
                lvl_three_logits = outlet_logits

            all_lvl_three_logits[i, :lvl_three_logits.shape[0]] = lvl_three_logits

    return all_lvl_zero_logits, all_lvl_one_logits, all_lvl_two_logits, all_lvl_three_logits
    

def generate_test_sets(config, env_config, path):
    """
    Generate balanced train, validation, and test sets.

    Guarantees:
    - Equal number of instances per subsystem in train.
    - Equal number of instances per subsystem in validation.
    - Equal number of instances per subsystem in test.
    - No duplicate instances within each subsystem's train/val split.
    - Reproducible generation using config.seed.

    Note:
    - config.eval_steps is the total validation set size.
    - num is retained for compatibility with the original signature.
    """

    np.random.seed(config.seed)
    random.seed(config.seed)

    steps = config.steps

    systems = {
        "acetone_chloroform": (
            ["acetone", "chloroform"],
            [np.array([0.5, 0.5, 0])]
        ),
        "ethanol_water": (
            ["ethanol", "water"],
            [np.array([0.5, 0.5, 0])]
        ),
        "n-butanol_water": (
            ["n-butanol", "water"],
            [np.array([0.4, 0.6, 0])]
        ),
        "water_pyridine": (
            ["water", "pyridine"],
            [np.array([0.9, 0.1, 0])]
        ),
    }

    train_instances = []
    val_instances = []
    test_instances = []

    # Collect subsystem indices and literature test instances
    subsystem_indices = {}

    for system_name, (names, feeds) in systems.items():

        if not env_config.systems_allowed[system_name]:
            continue

        temp_train, temp_test = helper_test_set_generation(
            names=names,
            config=env_config,
            steps=steps,
            set_feeds=[feeds]
        )

        subsystem_indices[system_name] = list(set(
            instance["feed_situation_index"]
            for instance in temp_train
        ))

        test_instances.extend(unique_instances(temp_test))

    n_systems = len(subsystem_indices)

    if n_systems == 0:
        raise ValueError("No subsystems enabled.")

    # Number of validation instances per subsystem
    val_per_system = config.eval_steps // n_systems

    # Number of training instances per subsystem
    '''if config.balanced:
        train_per_system = config.num_epochs 
    else:
        train_per_system = (
            config.num_epochs * config.num_instances_per_batch
        )'''

    if n_systems == 4:
        train_per_system = config.num_epochs * (config.num_instances_per_batch // 2)
    else:
        train_per_system = config.num_epochs * config.num_instances_per_batch

    # Generate train and validation instances separately per subsystem
    for system_name, indices in subsystem_indices.items():

        if not indices:
            raise ValueError(
                f"No feed indices found for {system_name}"
            )

        n_required = train_per_system + val_per_system

        instances = []
        attempts = 0
        max_attempts = max(1000, 20 * n_required)

        while len(instances) < n_required:

            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not generate {n_required} unique "
                    f"instances for {system_name}."
                )

            index = random.choice(indices)

            instance = env_config.create_random_problem_instance(index)

            instance = find_global_gamma_interaction_parameters(
                instance, env_config
            )

            instances = unique_instances(instances + [instance])

            attempts += 1

        random.shuffle(instances)

        val_instances.extend(instances[:val_per_system])
        train_instances.extend(
            instances[val_per_system:val_per_system + train_per_system]
        )

    # Deduplicate test instances
    test_instances = unique_instances(test_instances)

    # Balance test instances across subsystems
    test_by_system = {}

    for instance in test_instances:
        index = instance["feed_situation_index"]

        for system_name, indices in subsystem_indices.items():
            if index in indices:
                test_by_system.setdefault(system_name, []).append(instance)
                break

    if len(test_by_system) != n_systems:
        raise ValueError(
            "Some subsystems have no test instances."
        )

    test_per_system = min(
        len(instances) for instances in test_by_system.values()
    )

    test_instances = []

    for system_name in subsystem_indices:
        instances = test_by_system[system_name]
        random.shuffle(instances)
        test_instances.extend(instances[:test_per_system])

    # Shuffle final sets
    random.shuffle(train_instances)
    random.shuffle(val_instances)
    random.shuffle(test_instances)

    # Save
    save_path = os.path.join(os.getcwd(), path)
    os.makedirs(save_path, exist_ok=True)

    for filename, instances in [
        ("train_instances.pickle", train_instances),
        ("val_instances.pickle", val_instances),
        ("test_instances.pickle", test_instances),
    ]:
        with open(os.path.join(save_path, filename), "wb") as f:
            pickle.dump(instances, f)

    return train_instances, val_instances, test_instances, n_systems

    
def find_global_gamma_interaction_parameters(problem_instance, env_config):
    
    gamma_lookup = {}
    sim = FlowsheetSimulationGraph(problem_instance, env_config)
    feed_indicies = problem_instance["indices_components_in_feeds"]

    # base system: [A, B]
    gamma_lookup[tuple(feed_indicies)] = sim._build_gamma_inf_vector(feed_indicies)

    # every allowed solvent: [A, B, S]
    for solvent_idx in problem_instance["possible_ind_add_comp"]:
        indices = list(feed_indicies) + [solvent_idx]

        gamma_lookup[tuple(indices)] = sim._build_gamma_inf_vector(indices)

    problem_instance["gamma_lookup"] = gamma_lookup

    return problem_instance


def compute_gamma_normalization(train_instances):
        unique_gamma_vectors = set()
        all_gamma_values = []

        for instance in train_instances:
            for gamma_vec in instance["gamma_lookup"].values():

                gamma_vec = np.asarray(gamma_vec, dtype=np.float32)

                # Avoid counting exactly the same thermodynamic system many times
                key = tuple(np.round(gamma_vec, 8))

                if key in unique_gamma_vectors:
                    continue

                unique_gamma_vectors.add(key)

                # Remove padded zeros
                valid_gamma = gamma_vec[gamma_vec > 0]

                all_gamma_values.extend(valid_gamma.tolist())

        all_gamma_values = np.asarray(all_gamma_values, dtype=np.float32)

        # log transform first
        log_gamma = np.log(all_gamma_values)

        gamma_mean = log_gamma.mean()
        gamma_std = max(log_gamma.std(), 1e-6)

        return gamma_mean, gamma_std
        

def helper_test_set_generation(names, config, steps, set_feeds=None):
    train_instances = []
    test_instances = []

    # set_feeds is a list of lists, containing preset feed stream lists
    if set_feeds is not None:
        for feeds in set_feeds:
            instance = find_sit_create_instance(spec_feeds=feeds,
                                                names_comps=names,
                                                config=config)
            
            instance = find_global_gamma_interaction_parameters(instance, config)

            train_instances.append(instance)
            test_instances.append(instance)

    # Generate fixed test compositions
    for i in range(steps):
        x = i / (steps)
        instance = find_sit_create_instance(
            spec_feeds=[np.array([x, 1 - x, 0])],
            names_comps=names,
            config=config)
        
        instance = find_global_gamma_interaction_parameters(instance, config)

        test_instances.append(instance)

    assert len(unique_instances(test_instances)) == steps
    return train_instances, unique_instances(test_instances)


def find_sit_create_instance(spec_feeds, names_comps, config):
    for sit_ind in range(len(config.phase_eq_generator.feed_situations)):
        if len(config.phase_eq_generator.feed_situations[sit_ind][0]) == len(names_comps):
            all_in = True
            for name in names_comps:
                if not config.phase_eq_generator.names_components.index(name) in \
                       config.phase_eq_generator.feed_situations[sit_ind][0]:
                    all_in = False
                    break

            if all_in:
                index = sit_ind
                break

    situation = copy.deepcopy(config.phase_eq_generator.feed_situations[index])

    # we do not shuffle in this case
    # get names in feed streams
    names_in_streams = []
    for i in situation[0]:
        names_in_streams.append(config.phase_eq_generator.names_components[i])

    instance = {"feed_situation_index": index,
                "indices_components_in_feeds": situation[0],
                "list_feed_streams": spec_feeds,
                "possible_ind_add_comp": situation[1],
                "comp_order_feeds": names_in_streams,
                "system_name": "_".join(names_in_streams), 
                "lle_for_start": None,
                "vle_for_start": None}

    return instance


def batch_instances_for_dataset(config, train_instances):
    if config.balanced == True:
        grouped_instances = defaultdict(list)
            
        for instance in train_instances:
            idx = instance["feed_situation_index"]
            grouped_instances[idx].append(instance)

            # convert to list of batches
            batches = list(grouped_instances.values())

        for instances in grouped_instances.values():
            random.shuffle(instances)

        # create balanced batches
        batches = []

        # number of complete batches possible
        num_batches = min(len(v) for v in grouped_instances.values())

        feed_indices = sorted(grouped_instances.keys())

        for i in range(num_batches):
            batch = []
            for idx in feed_indices:
                batch.append(grouped_instances[idx][i])

            batches.append(batch)
    else:
        batches = []
        random.shuffle(train_instances)
        for i in range(0, len(train_instances), config.num_instances_per_batch):
            batch = train_instances[i:i + config.num_instances_per_batch]
            batches.append(batch)
    
    return batches


def batch_instances_for_dataset_eval(config, train_instances, n_system, if_eval):
    batches = []
    random.shuffle(train_instances)
    if if_eval: 
        total_instances = config.num_instances_per_batch * n_system 
    else:
        if n_system == 4: 
            total_instances = config.num_instances_per_batch * 2
        else:
            total_instances = config.num_instances_per_batch
    for i in range(0, len(train_instances), total_instances):
        batch = train_instances[i:i + total_instances]
        batches.append(batch)
    
    return batches

'''def batch_instances_for_dataset(config, train_instances):

    # target compositions = difficult cases
    difficult_feeds = {
        "acetone_chloroform": np.array([0.5, 0.5, 0.0]),
        "ethanol_water": np.array([0.5, 0.5, 0.0]),
        "water_pyridine": np.array([0.9, 0.1, 0.0]),
    }

    def difficulty_score(instance):
        subsystem = instance["system_name"]   # adapt key if yours is named differently
        feed = np.asarray(instance["list_feed_streams"][0])

        target = difficult_feeds[subsystem]

        # smaller distance = harder / higher priority
        return np.linalg.norm(feed - target)

    if config.balanced:
        grouped_instances = defaultdict(list)

        for instance in train_instances:
            idx = instance["feed_situation_index"]
            grouped_instances[idx].append(instance)

        # difficult feeds FIRST inside every subsystem
        for idx, instances in grouped_instances.items():
            instances.sort(key=difficulty_score)

        batches = []

        num_batches = min(
            len(v) for v in grouped_instances.values()
        )

        feed_indices = sorted(grouped_instances.keys())

        for i in range(num_batches):
            batch = []

            for idx in feed_indices:
                batch.append(grouped_instances[idx][i])

            batches.append(batch)

    else:
        # difficult instances first globally
        train_instances = sorted(
            train_instances,
            key=difficulty_score
        )

        batches = []

        for i in range(
            0,
            len(train_instances),
            config.num_instances_per_batch
        ):
            batch = train_instances[
                i:i + config.num_instances_per_batch
            ]
            batches.append(batch)

    return batches'''


def set_seed(seed=0, full_deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if full_deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            # Enable CuDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def dump_top_flowsheets_txt(input_fs, results_path, round_idx, if_test, if_evaluate):
    if if_test: 
        txt_path = f"{results_path}/top_test_flowsheets.txt"
        merged_fs = []
        for _, instances in input_fs.items():
            merged_fs.extend(instances)

    elif if_evaluate:
        txt_path = f"{results_path}/top_eval_flowsheets.txt"
        merged_fs = input_fs
    else:
        txt_path = f"{results_path}/top_train_flowsheets.txt"
        merged_fs = input_fs

    best_per_system = {}
    for x in merged_fs:
        index = (x["problem_instance"]["feed_situation_index"], tuple(np.round(x["problem_instance"]["list_feed_streams"][0], 6)))

        if index not in best_per_system or x["obj"] > best_per_system[index]["obj"]:
            best_per_system[index] = x

    if not if_test and not if_evaluate:
        if round_idx == 0:
            mode = "w"
        else:
            mode = "a"
    elif if_evaluate:
        mode = "a"
        round_idx = round_idx
    else:
        mode = "a"
        round_idx = 'test'

    
    with open(txt_path, mode) as f:
        f.write(f"round: {round_idx}\n\n")

        for index, x in best_per_system.items():
            pi = x["problem_instance"]
            graph = x["graph"]
            identifier = x["identifier"]
            f.write(f"identifier: {identifier}, ")
            f.write(f"subsystem: {pi.get('system_name')}, ")
            f.write(f"situation index: {pi.get('feed_situation_index')}, ")
            f.write(f"components in feed: {pi.get('indices_components_in_feeds')}, ")
            f.write(f"obj: {x.get('obj')}, ")
            f.write(f"feeds: {pi.get('list_feed_streams')}, ")
            f.write(f"npv_normed: {x.get('npv_normed')}, ")
            f.write(f"per_ratio: {x.get('per_ratio')}, ")
            f.write(f"npv_wo_app_cost: {x.get('npv_wo_app_cost')}, ")
            f.write(f"npv_raw: {x.get('npv_raw')}, ")
            f.write(f"total_units_placed: {x.get('total_units_placed')}, ")
            f.write("Nodes:\n")
            for node_idx, node_data in graph._node.items():
                f.write(f"node {node_idx}:\n")
                for k, v in node_data.items():
                    f.write(f"{k}: {v}\n")
            f.write("\n")
            f.write("Edges:\n") 
            for src, dst, key, data in graph.edges(keys=True, data=True): 
                f.write(f"\nedge {src} -> {dst}\n") 
                f.write(f" output_label : {data.get('output_label')}, ") 
                f.write(f" is_recycle : {data.get('is_recycle')}, ") 
                flow = data.get("stream", {}).get("flow") 
                if flow is not None: 
                    f.write(f" flow : {flow.tolist()}") 
                    f.write("\n\n")
            f.write("=" * 120 + "\n\n")

def unique_instances(instance_list):
    unique_instances = []
    seen = set()

    for instance in instance_list:
        key = (
            instance["feed_situation_index"],
            tuple(np.round(instance["list_feed_streams"][0], 6))
        )

        if key not in seen:
            seen.add(key)
            unique_instances.append(instance)

    random.shuffle(unique_instances)
    return unique_instances


def process_test_results(problem_instances, results, destination_path, epoch, if_test):
        
        """
        Processes the results from Gumbeldore search and save it to a pickle. Each trajectory will be represented as a dict with the
        following keys and values
        "action_seq": List[List[int]] Actions which need to be taken on each index to create the molecule
        "obj": [float] NPV value

        Then:
        1. If the dataset already exists at the path where to save, we load it, merge them and take the best from the
            merged dataset.

        Then returns the following dictionary:
        - "mean_best_gen_obj": Mean best generated obj. -> over the unmerged best flowsheets generated
        - "best_gen_obj": Best generated obj. -> Best obj. of the unmerged flowsheets generated
        - "worst_gen_obj": Worst generated obj. -> Worst obj. of the unmerged flowsheets generated
        - "mean_top_20_obj": Mean top 20 obj. -> over the merged best flowsheets
        - "top_20_flowsheets": A list of flowsheets with obj. of the top 20 obj.
        """

        per_feed_index = {}
        instances = []

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
                        advantage = flowsheet.advantage,
                        npv_raw = flowsheet.sim.current_net_present_value,
                        npv_normed = flowsheet.sim.current_net_present_value_normed,
                        per_ratio = flowsheet.sim.performance_ratio,
                        npv_wo_app_cost = flowsheet.sim.npv_without_app_cost,

                    ))

            instances.extend(per_instances)
            key = (results[i][0].problem_instance["feed_situation_index"], tuple(np.round(results[i][0].problem_instance["list_feed_streams"][0], 6)))
            per_feed_index[key] = per_instances

        # Now check if there already is a data file, and if so, load it and merge it.
        if destination_path is not None:
            destination_full_path = f"{destination_path}/generated_test_flowsheets.pickle"
            existing_fs = []
            if os.path.isfile(destination_full_path):
                with open(destination_full_path, "rb") as f:
                    existing_fs = pickle.load(f) 

            merged_fs = []
            for _, instances in per_feed_index.items():
                # this will select top K flowsheets PER system for training
                top_k = sorted(instances, key=lambda x: x["obj"], reverse=True)[:10]
                merged_fs.extend(top_k)

            merged_fs = existing_fs + merged_fs
            # Pickle the generated data again
            with open(destination_full_path, "wb") as f:
                pickle.dump(merged_fs, f)
        
        dump_top_flowsheets_txt(per_feed_index, destination_path, epoch, if_test, if_evaluate=False)