"""Convert MCTS blueprints into current hierarchical actions and verify replay.

Only pair[1] is read. Verification uses FlowsheetDesign.take_action and the
simulator's real graph mutation methods, with search masks/limits relaxed.
The default checks design equality, not thermodynamic or objective equality.
Run with --simulate to additionally attempt physical replay and compare flows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import pickle, random
from types import SimpleNamespace

import networkx as nx
import numpy as np

from config import EnvConfig, GeneralConfig
from environment.environment_actions_graphs import FlowsheetDesign
from environment.flowsheet_simulation_graph import FlowsheetSimulationGraph

ROOT = Path(__file__).resolve().parent
PARAM_ATOL = 1e-12
PROFILE = "graph_replay_without_search_restrictions_v1"


class StructuralSimulation(FlowsheetSimulationGraph):
    """Use production graph operations, without phase equilibrium calculations.

    Zero placeholders mark outlets as available to the production enumerator.
    They are NOT physical flows and never come from the reference blueprint.
    """

    def __init__(self, feed_stream_information, env_config):
        self.env_config = env_config
        self.feed_stream_information = feed_stream_information
        self.graph = nx.MultiDiGraph()
        self.feed_nodes = []
        self.next_node_id = 0
        self.current_net_present_value = 0.0
        self.current_net_present_value_normed = 0.0
        self._last_recycle_guess = None
        self.current_indices = list(feed_stream_information["indices_components_in_feeds"])
        self.current_phase_eq = None
        self.system_metadata = {}

    def simulate(self):
        for _, node in self.graph.nodes(data=True):
            if node["unit_type"] == "add_solvent":
                index = node["params"]["index_new_component"]
                if index not in self.current_indices:
                    self.current_indices.append(index)
            for label, value in node["output_flows"].items():
                if value is None:
                    node["output_flows"][label] = np.zeros(self.env_config.max_number_of_components)


class BlueprintReplayDesign(FlowsheetDesign):
    """Production action decoder with structural, rather than search, legality.

    No unit-count, composition, minimum-length, or recycle search restrictions.
    Index bounds, open outlets, distinct mixer outlets and valid recycle endpoints
    remain checked. This profile is not the default policy training environment.
    """

    def __init__(self, problem_instance, env_config, simulate=False):
        config = copy.copy(env_config)
        config.allow_forward_recycles = True
        super().__init__(copy.deepcopy(problem_instance), None, config,
                         simulator_class=FlowsheetSimulationGraph if simulate else StructuralSimulation)

    def _unit_available(self, unit_name, stream=None):
        stream = stream if stream is not None else self.current_state["chosen_open_stream"]
        if stream not in self._enumerate_open_streams():
            return False
        if unit_name == "mixer":
            return bool(self._eligible_mixer_streams(stream))
        if unit_name == "recycle":
            return bool(self._eligible_recycle_destinations(stream))
        return unit_name in self.env_config.units_map_indices_type

    def _eligible_recycle_destinations(self, stream=None):
        stream = stream if stream is not None else self.current_state["chosen_open_stream"]
        if stream is None:
            return []
        return [nid for nid, node in self.sim.graph.nodes(data=True)
                if nid != stream[0] and node["unit_type"] != "feed"]

    def _eligible_mixer_streams(self, stream):
        return [other for other in self._enumerate_open_streams()
                if other != stream]

    def forced_termination(self):
        return False

    def optional_termination(self):
        return True


def exact_index(grid, value, description):
    matches = np.flatnonzero(np.isclose(np.asarray(grid), value, rtol=0, atol=PARAM_ATOL))
    if len(matches) != 1:
        raise ValueError(f"{description}={value!r} has {len(matches)} exact grid matches; refusing to round")
    return int(matches[0])


'''def action_mapping(config):
    return {
        "unit_types": list(config.units_map_indices_type),
        "component_names": list(config.component_names),
        "df": config.DF_distillation_map.tolist(),
        "split_ratio": config.split_ratio_map.tolist(),
        "solvent_amount": {k: v.tolist() for k, v in config.add_solvent_comp_map.items()},
        "outlet_indices": dict(config.outlet_to_idx),
    }'''

def action_mapping(config, feed_components=None):
    """
    Build a conditional action space for Qwen.

    Args:
        config: Environment configuration.
        feed_components: Original feed component names, e.g.
                         ["ethanol", "water"].
    """

    solvent_rules = {
        frozenset(["ethanol", "water"]): [
            "benzene", "toluene", "tetrahydrofuran"
        ],
        frozenset(["n-butanol", "water"]): [
            "acetone", "benzene", "toluene"
        ],
        frozenset(["water", "pyridine"]): [
            "toluene"
        ],
        frozenset(["acetone", "chloroform"]): [
            "benzene", "toluene"
        ],
    }

    # Determine solvents allowed for the current feed.
    if feed_components is not None:
        feed_key = frozenset(feed_components)

        if feed_key not in solvent_rules:
            raise ValueError(
                f"Unknown feed system: {sorted(feed_key)}"
            )

        allowed_solvents = solvent_rules[feed_key]
    else:
        # No specific feed: expose all configured solvents.
        allowed_solvents = list(config.add_solvent_comp_map)

    # Only include solvents available in the actual configuration.
    solvent_amounts = {
        solvent: config.add_solvent_comp_map[solvent].tolist()
        for solvent in allowed_solvents
        if solvent in config.add_solvent_comp_map
    }

    unit_definitions = {
        "distillation_column": {
            "required_params": {
                "df": config.DF_distillation_map.tolist()
            },
        },
        "split": {
            "required_params": {
                "split_ratio": config.split_ratio_map.tolist()
            },
        },
        "add_solvent": {
            "required_params": {
                "solvent": list(solvent_amounts),
                "solvent_amount": solvent_amounts,
            },
        },
        "decanter": {
            "required_params": {},
        },
        "mixer": {
            "required_params": {},
        },
        "recycle": {
            "required_params": {},
        },
    }

    available_unit_types = list(config.units_map_indices_type)

    return {
        "unit_types": {
            unit_type: unit_definitions[unit_type]
            for unit_type in available_unit_types
            if unit_type in unit_definitions
        },
        "component_names": list(config.component_names),
        "outlet_indices": dict(config.outlet_to_idx),
    }


def apply_action(fs, action, expected_level=None):
    if fs.current_state["completed_design"]:
        raise ValueError("Action follows termination")
    if expected_level is not None and fs.level != expected_level:
        raise ValueError(f"Expected level {expected_level}, got {fs.level}")
    finished, _, worked = fs.take_action(int(action), None)
    if not worked:
        raise ValueError(f"Replay failed at action {action}, level {expected_level}")
    return finished


def replay_actions(record, config, simulate=False):
    """Rebuild from feed + actions only; never reads source nodes or edges."""
    if record["action_mapping"] != action_mapping(config):
        raise ValueError("Saved action mapping differs from current config")
    if record["replay_profile"] != PROFILE:
        raise ValueError("Unknown replay profile")
    if len(record["action_seq"]) != len(record["levels"]):
        raise ValueError("Action and level sequence lengths differ")
    fs = BlueprintReplayDesign(record["problem_instance"], config, simulate=simulate)
    for action, level in zip(record["action_seq"], record["levels"]):
        apply_action(fs, action, level)
    if not fs.current_state["completed_design"] or fs.level != 0:
        raise ValueError("Sequence must end with level-0 termination")
    return fs


def blueprint_graph(blueprint):
    graph = nx.MultiDiGraph()
    for node in blueprint["nodes"]:
        nid = node["node_id"]
        if nid in graph:
            raise ValueError(f"Duplicate node ID {nid}")
        graph.add_node(nid, **{k: copy.deepcopy(v) for k, v in node.items() if k != "node_id"})
    consumed = set()
    for edge in blueprint["edges"]:
        src, dst, label = edge["source"], edge["destination"], edge["output_label"]
        if src not in graph or dst not in graph:
            raise ValueError("Edge references a missing node")
        if label not in graph.nodes[src]["output_flows"]:
            raise ValueError(f"Missing outlet {(src, label)}")
        if (src, label) in consumed:
            raise ValueError(f"Outlet {(src, label)} has multiple consumers")
        consumed.add((src, label))
        graph.add_edge(src, dst, output_label=label, is_recycle=bool(edge["is_recycle"]))
    return graph


def edge_signature(graph):
    # Keys are storage identifiers; Counter preserves actual parallel edges.
    return Counter((u, v, d["output_label"], bool(d["is_recycle"]))
                   for u, v, d in graph.edges(data=True))


def assert_value_equal(actual, expected, description):
    if isinstance(expected, str):
        equal = actual == expected
    else:
        equal = np.allclose(actual, expected, rtol=0, atol=PARAM_ATOL)
    if not equal:
        raise ValueError(f"{description}: expected {expected!r}, got {actual!r}")


def verify_design(blueprint, fs, source_to_replay):
    reference = blueprint_graph(blueprint)
    if set(source_to_replay) != set(reference) or set(source_to_replay.values()) != set(fs.sim.graph):
        raise ValueError("Node mapping does not cover both graphs")
    expected = nx.relabel_nodes(reference, source_to_replay, copy=True)
    actual = fs.sim.graph
    if edge_signature(actual) != edge_signature(expected):
        raise ValueError("Source/destination/outlet/recycle edge multisets differ")
    for nid, node in expected.nodes(data=True):
        result = actual.nodes[nid]
        if node["unit_type"] != result["unit_type"]:
            raise ValueError(f"Unit type differs at node {nid}")
        if list(node["output_flows"]) != list(result["output_flows"]):
            raise ValueError(f"Outlet names/order differ at node {nid}")
        if node["unit_type"] == "feed":
            assert_value_equal(result["output_flows"]["out0"], node["output_flows"]["out0"], "Feed flow")
            if "system_indices" in node["params"]:
                assert_value_equal(fs.sim.current_indices, node["params"]["system_indices"], "Component order")
        else:
            if set(node["params"]) != set(result["params"]):
                raise ValueError(f"Parameter keys differ at node {nid}")
            for key, value in node["params"].items():
                assert_value_equal(result["params"][key], value, f"Node {nid} {key}")
    consumed = {(u, d["output_label"]) for u, _, d in expected.edges(data=True)}
    expected_opens = {(nid, label) for nid, n in expected.nodes(data=True)
                      for label in n["output_flows"] if (nid, label) not in consumed}
    if set(fs.sim.get_open_streams()) != expected_opens:
        raise ValueError("Open outlets differ")
    if "leaving_streams" in blueprint:
        leaving = {(source_to_replay[s["index"][0]], s["index"][1])
                   for s in blueprint["leaving_streams"]}
        if leaving != expected_opens:
            raise ValueError("Archive leaving-stream outlets disagree with replay")
    return {"design_equal": True, "node_ids_preserved": all(k == v for k, v in source_to_replay.items()),
            "nodes": len(actual), "edges": actual.number_of_edges(),
            "parameter_atol": PARAM_ATOL, "physical_results_verified": False}


def convert_blueprint(blueprint, config, source_index=0):
    target = blueprint_graph(blueprint)
    problem = copy.deepcopy(blueprint["initial_information"])
    fs = BlueprintReplayDesign(problem, config)
    feed_ids = [nid for nid, node in target.nodes(data=True) if node["unit_type"] == "feed"]
    if len(feed_ids) != len(fs.sim.feed_nodes):
        raise ValueError("Blueprint feed count differs from problem instance")
    mapping = dict(zip(feed_ids, fs.sim.feed_nodes))
    actions, levels = [], []

    def emit(action):
        levels.append(fs.level)
        actions.append(int(action))
        apply_action(fs, action, levels[-1])

    pending = [nid for nid in target if nid not in mapping]
    # Build the ordinary DAG first. Recycles consume otherwise-unused outlets,
    # so adding them last is structurally equivalent and avoids cyclic ordering.
    while pending:
        ready = None
        for nid in pending:
            inputs = [(u, edge["output_label"]) for u, _, edge in target.in_edges(nid, data=True)
                      if not edge["is_recycle"]]
            if inputs and all(u in mapping for u, _ in inputs):
                ready = (nid, inputs)
                break
        if ready is None:
            raise ValueError(f"Non-recycle cycle or disconnected units: {pending}")
        nid, inputs = ready
        node = target.nodes[nid]
        kind, params = node["unit_type"], node["params"]
        if len(inputs) != (2 if kind == "mixer" else 1):
            raise ValueError(f"Wrong input count for {kind} node {nid}")
        streams = [(mapping[u], label) for u, label in inputs]
        emit(fs._enumerate_open_streams().index(streams[0]) + 1)
        emit(config.units_map_indices_type.index(kind))
        if kind == "distillation_column":
            emit(exact_index(config.DF_distillation_map, params["df"], "df"))
        elif kind == "split":
            emit(exact_index(config.split_ratio_map, params["split_ratio"], "split_ratio"))
        elif kind == "add_solvent":
            comp = params["index_new_component"]
            if config.component_names[comp] != params["component_name"]:
                raise ValueError("Solvent component name/index mismatch")
            emit(comp)
            emit(exact_index(config.add_solvent_comp_map[params["component_name"]],
                             params["solvent_amount"], "solvent_amount"))
        elif kind == "mixer":
            emit(list(fs.sim.graph).index(streams[1][0]))
            emit(config.outlet_to_idx[streams[1][1]])
        elif kind != "decanter":
            raise ValueError(f"Unknown unit {kind}")
        if fs.level != 0 or len(fs.sim.graph) != len(mapping) + 1:
            raise ValueError(f"Placement of {nid} did not create exactly one node")
        mapping[nid] = list(fs.sim.graph)[-1]
        pending.remove(nid)

    for u, v, edge in target.edges(data=True):
        if edge["is_recycle"]:
            stream = (mapping[u], edge["output_label"])
            emit(fs._enumerate_open_streams().index(stream) + 1)
            emit(config.units_map_indices_type.index("recycle"))
            emit(list(fs.sim.graph).index(mapping[v]))
    emit(0)
    record = {
        "source_index": source_index, "source_graph_index": 1,
        "problem_instance": problem, "action_seq": actions, "levels": levels,
        "advantage": 1.0, "obj": blueprint.get("objective"),
        "explicit_npv": blueprint.get("explicit_npv"),
        "source_to_replay_node_ids": mapping,
        "action_mapping": action_mapping(config), "replay_profile": PROFILE,
    }
    # Independent fresh replay uses only the record's feed and integer decisions.
    rebuilt = replay_actions(record, config)
    record["verification"] = verify_design(blueprint, rebuilt, mapping)
    # Store design data only: placeholder flow vectors are deliberately excluded.
    record["rebuilt_graph"] = {
        "nodes": [{"node_id": nid, "unit_type": n["unit_type"],
                   "params": copy.deepcopy(n["params"]), "outlets": list(n["output_flows"])}
                  for nid, n in rebuilt.sim.graph.nodes(data=True)],
        "edges": [{"source": u, "destination": v, "output_label": d["output_label"],
                   "is_recycle": d["is_recycle"]} for u, v, d in rebuilt.sim.graph.edges(data=True)],
        "open_streams": rebuilt.sim.get_open_streams(),
        "component_indices": list(rebuilt.sim.current_indices),
    }
    return record


def verify_physics(blueprint, record, config):
    fs = replay_actions(record, config, simulate=True)
    verify_design(blueprint, fs, record["source_to_replay_node_ids"])
    mismatches = []
    for node in blueprint["nodes"]:
        nid = record["source_to_replay_node_ids"][node["node_id"]]
        for label, expected in node["output_flows"].items():
            actual = fs.sim.graph.nodes[nid]["output_flows"][label]
            if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                mismatches.append({"source_node": node["node_id"], "outlet": label,
                                   "max_abs_error": float(np.max(np.abs(actual - expected)))})
    return {"replay_succeeded": True, "node_output_flows_equal": not mismatches,
            "rtol": 1e-5, "atol": 1e-6, "mismatches": mismatches,
            "recomputed_objective": float(fs.objective)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "/home/akhanna/GumbelFlowsheets/evaluation_win_blueprint_bin.pickle")
    parser.add_argument("--output", type=Path, default=ROOT / "/home/akhanna/GumbelFlowsheets/train_actions_new.pickle")
    parser.add_argument("--simulate", action="store_true", help="Also attempt physical simulation replay")
    args = parser.parse_args()
    input_path, output_path = args.input.resolve(), args.output.resolve()
    if input_path == output_path:
        parser.error("Input and output must be different files")
    # EnvConfig locates phase equilibrium resources relative to cwd.
    os.chdir(ROOT)
    with input_path.open("rb") as handle:
        archive = pickle.load(handle)
    with contextlib.redirect_stdout(io.StringIO()):
        config = EnvConfig(SimpleNamespace(subsystem="all"))
    records, failures = [], []
    systems, units = Counter(), Counter()
    for index, pair in enumerate(archive):
        log = io.StringIO()
        try:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Expected a pair of blueprints")
            blueprint = pair[1]
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                record = convert_blueprint(blueprint, config, index)
                if args.simulate:
                    physical_log = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(physical_log), contextlib.redirect_stderr(physical_log):
                            record["physical_replay"] = verify_physics(blueprint, record, config)
                    except Exception as exc:
                        record["physical_replay"] = {
                            "replay_succeeded": False, "error": str(exc),
                            "simulator_log": physical_log.getvalue(),
                        }
            records.append(record)
            systems[blueprint["initial_information"]["system_name"]] += 1
            units.update(node["unit_type"] for node in blueprint["nodes"])
        except Exception as exc:
            failures.append({"source_index": index, "error": str(exc), "log": log.getvalue()})
        if (index + 1) % 25 == 0:
            print(f"Checked {index + 1}/{len(archive)}; verified {len(records)}", flush=True)
    report = {
        "input": str(input_path), "output": str(output_path), "selected_graph_index": 1,
        "total": len(archive), "verified": len(records), "failures": failures,
        "actions": sum(len(r["action_seq"]) for r in records),
        "systems": dict(systems), "units": dict(units),
        "recycle_edges": sum(sum(e["is_recycle"] for e in r["rebuilt_graph"]["edges"]) for r in records),
        "replay_profile": PROFILE, "action_mapping": action_mapping(config),
        "verification_scope": "Feed flows/component order, unit types, operating parameters, outlet order, edge multiplicity/labels/recycle flags, and open outlets. Feed critical/gamma metadata, derived stream flows and NPV are not design equality fields.",
        "physical_replay_requested": args.simulate,
        "physical_replay_summary": dict(Counter(
            "failed" if not r["physical_replay"]["replay_succeeded"] else
            "flows_equal" if r["physical_replay"]["node_output_flows_equal"] else "flows_differ"
            for r in records if "physical_replay" in r)),
        "physical_replay": [{"source_index": r["source_index"], **r["physical_replay"]}
                            for r in records if "physical_replay" in r],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not records:
        raise SystemExit("Archive contains no demonstrations; no dataset written")
    
    train_records, val_records = split_train_validation(records, val_fraction = 0.2)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(train_records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    valid_path=ROOT / "/home/akhanna/GumbelFlowsheets/val_actions_new.pickle"
    with valid_path.open("wb") as handle:
        pickle.dump(val_records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    temporary.replace(output_path)
    print(f"Verified {len(records)}/{len(archive)} designs, {report['actions']} actions. Saved {output_path}")
    print(f"Report: {report_path}")

def same_instance_3dp(a, b):
    system_a, feeds_a = a
    system_b, feeds_b = b

    if system_a != system_b:
        return False

    if len(feeds_a) != len(feeds_b):
        return False

    return all(
        np.array_equal(np.round(x, 3), np.round(y, 3))
        for x, y in zip(feeds_a, feeds_b)
    )


def split_train_validation(records, val_fraction = 0.2):
    seed = 42
    rng = random.Random(seed)

    filtered_records = []
    if os.path.isfile('/home/akhanna/GumbelFlowsheets/finetuning_pretrain_new/all/test_instances.pickle'):
        with open('/home/akhanna/GumbelFlowsheets/finetuning_pretrain_new/all/test_instances.pickle', "rb") as f:
            test_instances = pickle.load(f)

    test_flowsheets = []
    for fs in test_instances: 
        test_flowsheets.append((fs['feed_situation_index'], fs['list_feed_streams']))

    #remove problematic flowsheets
    problematic_flowsheets = [
        (3, [np.array([0.07724903, 0.92275097, 0.0])]),
        (0, [np.array([0.31811875, 0.68188125, 0.0])]),
        (2, [np.array([0.92510855, 0.07489145, 0.0])]),
        (1, [np.array([0.46468695, 0.53531305, 0.0])]),
        (3, [np.array([0.8685838,  0.1314162,  0.0])]),
        (1, [np.array([0.72221614, 0.27778386, 0.0])]),
        (3, [np.array([0.89750227, 0.10249773, 0.0])]),
        (1, [np.array([0.21921211, 0.78078789, 0.0])]),
        (3, [np.array([0.9465314, 0.0534686, 0.0])]) 
    ]

    n_test_overlap = 0
    n_problematic = 0
    n_low_obj = 0

    for record in records:
        record_tuple = (record['problem_instance']['feed_situation_index'], record['problem_instance']['list_feed_streams'])
        if any(
            same_instance_3dp(record_tuple, test)
            for test in test_flowsheets
        ):
            n_test_overlap += 1
            continue
        if any(
            same_instance_3dp(record_tuple, bad)
            for bad in problematic_flowsheets
        ):
            n_problematic += 1
            continue

        if record["obj"] <= 0.7:
            n_low_obj += 1
            continue
        
        filtered_records.append(record)
    
    print(f"Original records:       {len(records)}")
    print(f"Test overlap removed:   {n_test_overlap}")
    print(f"Problematic removed:    {n_problematic}")
    print(f"obj <= 0.7 removed:     {n_low_obj}")
    print(f"Remaining:              {len(filtered_records)}")

    print(
        f"Removed {len(records) - len(filtered_records)} records "
        f"due to test-set overlap."
    )

    indices = list(range(len(filtered_records)))
    rng.shuffle(indices)

    num_val = int(round(len(filtered_records) * val_fraction))

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_records = [filtered_records[i] for i in train_indices]
    val_records = [filtered_records[i] for i in val_indices]

    return train_records, val_records

if __name__ == "__main__":
    main()
