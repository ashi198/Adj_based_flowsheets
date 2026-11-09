# flowsheet_simulation_graph.py

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import networkx as nx
import scipy.optimize as opt
import copy

from environment import units

class FlowsheetSimulationGraph:

    """
    Graph-based flowsheet simulator.

    Nodes:
      - feed
      - add_solvent
      - distillation_column
      - decanter
      - split
      - mixer

    MultiDiGraph edges carry:
      - 'output_label' (e.g., 'out0', 'out1')
      - 'is_recycle' (bool)
      - 'stream' dict with 'flow': np.array([...])

    Node stores:
      - 'unit_type'
      - 'params'
      - 'output_flows' (dict: label -> np.array)
      - 'phase_eq' (when needed)
    """

    def __init__(self, feed_stream_information: Dict[str, Any], env_config):
        self.env_config = env_config 
        self.feed_stream_information = feed_stream_information

        self.graph = nx.MultiDiGraph()
        self.feed_nodes: list[int] = []
        self.next_node_id = 0

        # Track current NPV
        self.current_net_present_value: float = 0.0
        self.current_net_present_value_normed: float = 0.0

        # cache for recycle solving
        self._last_recycle_guess: Optional[np.ndarray] = None

        self.current_indices: list[int] = list(
            self.feed_stream_information["indices_components_in_feeds"]
        )

        # Active phase-equilibrium set for the flowsheet's current component set
        self.current_phase_eq: Optional[Dict[str, Any]] = None
        self._set_active_phase_eq_from_current_indices()  # build VLE/LLE for current components

        # system metadata (critical data + gamma_inf)
        self.system_metadata: Dict[str, Optional[np.ndarray]] = {
            "indices": list(self.current_indices),  # global indices in FS order
            "pure_critical": None,  # flat: TC0,PC0,ω0, TC1,PC1,ω1, TC2,PC2,ω2
            "gamma_inf": None,  # flat: γ01,γ10, γ02,γ20, γ12,γ21 (padded with zeros)
        }
        # build initial metadata from feed indices
        self._refresh_system_metadata()

    # ----- snapshot -----

    def snapshot(self, include_phase: bool = False) -> dict:
        """
        Selective snapshot: captures only the mutable numeric payload that simulate() can change.
        Does NOT deepcopy the NetworkX graph topology.
        """
        # node outputs
        node_outputs = {}
        for nid, ndata in self.graph.nodes(data=True):
            of = ndata.get("output_flows", {})
            node_outputs[nid] = {
                lbl: (arr.copy() if isinstance(arr, np.ndarray) else None)
                for lbl, arr in of.items()
            }

        # edge stream flows (incl. recycle tear edges)
        edge_flows = {}
        for u, v, k, edata in self.graph.edges(keys=True, data=True):
            flow = edata.get("stream", {}).get("flow", None)
            edge_flows[(u, v, k)] = (flow.copy() if isinstance(flow, np.ndarray) else None)

        snap = {
            "node_outputs": node_outputs,
            "edge_flows": edge_flows,
            "current_net_present_value": self.current_net_present_value,
            "current_net_present_value_normed": self.current_net_present_value_normed,
            "_last_recycle_guess": None if self._last_recycle_guess is None else self._last_recycle_guess.copy(),
        }

        # Only needed when an action can change component set / PEQ / metadata (add_solvent)
        if include_phase:
            snap["current_indices"] = list(self.current_indices)
            snap["current_phase_eq"] = copy.deepcopy(self.current_phase_eq)
            snap["system_metadata"] = copy.deepcopy(self.system_metadata)

        return snap

    def restore(self, snap: dict) -> None:
        """
        Restore from snapshot(). Assumes caller has already rolled back topology
        (removed newly added nodes/edges).
        """
        # restore node outputs
        saved_node_outputs = snap["node_outputs"]
        for nid, ndata in self.graph.nodes(data=True):
            if nid not in saved_node_outputs:
                continue
            of = ndata.get("output_flows", {})
            saved = saved_node_outputs[nid]
            for lbl in of.keys():
                arr = saved.get(lbl, None)
                of[lbl] = (arr.copy() if isinstance(arr, np.ndarray) else None)

        # restore edge flows
        saved_edge_flows = snap["edge_flows"]
        for u, v, k, edata in self.graph.edges(keys=True, data=True):
            key = (u, v, k)
            if key not in saved_edge_flows:
                continue
            flow = saved_edge_flows[key]
            edata.setdefault("stream", {})
            if flow is None:
                edata["stream"].pop("flow", None)
            else:
                edata["stream"]["flow"] = flow.copy()

        # restore caches/scalars
        self.current_net_present_value = snap["current_net_present_value"]
        self.current_net_present_value_normed = snap["current_net_present_value_normed"]
        self._last_recycle_guess = snap["_last_recycle_guess"]

        # restore phase state if present
        if "current_indices" in snap:
            self.current_indices = list(snap["current_indices"])
            self.current_phase_eq = snap["current_phase_eq"]
            self.system_metadata = snap["system_metadata"]

    def snapshot_edges(self):
        return [(u, v, k, copy.deepcopy(d)) for u, v, k, d in self.graph.edges(keys=True, data=True)]

    def restore_edges(self, edge_snapshot):
        self.graph.remove_edges_from(list(self.graph.edges(keys=True)))
        for u, v, k, d in edge_snapshot:
            self.graph.add_edge(u, v, key=k, **copy.deepcopy(d))

    # ----- graph builders -----

    def add_feed(self, flow: np.ndarray) -> int:
        flow = np.asarray(flow, dtype=float).ravel()
        if flow.ndim != 1 or len(flow) != self.env_config.max_number_of_components:
            raise ValueError("Feed flow must be a 1D vector of length max_number_of_components.")
        flow = np.maximum(flow, 0.0)
        nid = self._new_node_id()
        self.graph.add_node(
            nid,
            unit_type="feed",
            params={},
            output_flows={"out0": np.array(flow, dtype=float)},
        )
        return nid

    def add_unit(
        self,
        inputs: list[Tuple[int, str]],
        unit_type: str,
        params: Dict[str, Any],
        num_outputs: int,
    ) -> int:
        """
        Add a processing unit node and wire edges from given inputs (each is (node_id, output_label)).
        Returns the new node id.
        """
        nid = self._new_node_id()
        self.graph.add_node(
            nid,
            unit_type=unit_type,
            params=dict(params) if params else {},
            output_flows={f"out{k}": None for k in range(num_outputs)},
        )
        # wire inputs
        for src_node, src_label in inputs:
            self._consume_open_stream(src_node, src_label)  # ensure it was open
            self.graph.add_edge(
                src_node, nid,
                output_label=src_label,
                is_recycle=False,
                stream={},  # stream dict will be filled by simulate()
            )

            # prefill this exact edge right away (if upstream already has a computed output)
            try:
                edge_keys = list(self.graph[src_node][nid].keys())
                new_key = max(edge_keys)
                edata = self.graph[src_node][nid][new_key]

                upstream_outputs = self.graph.nodes[src_node].get("output_flows", {})
                flow = upstream_outputs.get(src_label)
                if flow is None and self.graph.nodes[src_node].get("unit_type") == "feed":
                    flow = upstream_outputs.get("out0")

                if flow is not None:
                    edata.setdefault("stream", {})
                    edata["stream"]["flow"] = np.asarray(flow, dtype=float).copy()
            except KeyError:
                pass  # edge not found as expected

        return nid

    def add_recycle(self, from_node: int, output_label: str, to_node: int) -> None:
        """
        Add a recycle edge from an existing node's output stream (still open) to an existing unit node.
        The destination must NOT be a feed and should have available input capacity.

        If env_config.allow_forward_recycles is False, only true backward recycles
        are allowed: to_node must be lower/older than from_node.
        """
        if from_node not in self.graph:
            raise ValueError(f"Recycle source node {from_node} does not exist.")
        if to_node not in self.graph:
            raise ValueError(f"Recycle destination node {to_node} does not exist.")
        if to_node in getattr(self, "feed_nodes", []):
            raise ValueError("Cannot recycle into a feed node.")
        if to_node == from_node:
            raise ValueError("Cannot recycle a stream back into its own producing unit.")
        if not bool(getattr(self.env_config, "allow_forward_recycles", True)):
            if to_node >= from_node:
                raise ValueError(
                    "Forward recycle is disabled: recycle destination node "
                    f"{to_node} must be lower than source node {from_node}."
                )

        # confirm source stream is still open
        self._consume_open_stream(from_node, output_label)  # checks the stream exists & is open

        # wire recycle edge
        self.graph.add_edge(
            from_node,
            to_node,
            output_label=output_label,
            is_recycle=True,
            stream={},  # will be filled by recycle solve
        )

    def remove_node_and_restore_upstream_open(self, node_id: int) -> None:
        if node_id in self.graph:
            self.graph.remove_node(node_id)

    def remove_recycle_edge(self, from_node: int, output_label: str, to_node: int) -> None:
        for u, v, k, d in list(self.graph.edges(keys=True, data=True)):
            if not d.get("is_recycle", False):
                continue
            if u == from_node and v == to_node and d.get("output_label") == output_label:
                self.graph.remove_edge(u, v, key=k)
                break

    # ------------------- simulate -------------------

    def simulate(self) -> None:
        """
        Propagate flows forward; if cycles (recycles) exist, solve tear streams.
        Then run mass-balance check and compute NPV.
        """
        if self._has_cycles():
            self._simulate_with_recycles()
        else:
            self._forward_propagation_without_cycles()

        # mass balance check (single call)
        self._mass_balance_check()

        # update system data
        self._update_system_metadata_on_feed()

        # compute NPV
        self.current_net_present_value, self.current_net_present_value_normed = self.compute_npv()

    # ------------------- forward pass (DAG) -------------------

    def _forward_propagation_without_cycles(self) -> None:
        order = list(nx.topological_sort(self.graph))
        for node_id in order:
            unit_type = self.graph.nodes[node_id].get("unit_type")
            if unit_type is None:
                continue

            if unit_type == "feed":
                continue

            total_input = self._sum_inbound_streams(node_id)
            outputs = self._simulate_unit(node_id, unit_type, total_input, self.graph.nodes[node_id].get("params", {}))
            outputs = [np.maximum(np.asarray(o, dtype=float), 0.0) for o in outputs]

            out_labels = sorted(self.graph.nodes[node_id]["output_flows"].keys())
            if len(outputs) != len(out_labels):
                raise RuntimeError(
                    f"Unit '{unit_type}' at node {node_id} returned {len(outputs)} "
                    f"outputs, but node declares {len(out_labels)}."
                )

            # write outputs
            for k, lbl in enumerate(out_labels):
                self.graph.nodes[node_id]["output_flows"][lbl] = np.array(outputs[k], dtype=float)

            # push downstream edges streams
            for u, v, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                lbl = data.get("output_label")
                if lbl in self.graph.nodes[u]["output_flows"]:
                    data["stream"]["flow"] = np.array(self.graph.nodes[u]["output_flows"][lbl], dtype=float)

    # ----- recycle (tear) solve -----

    def _simulate_with_recycles(self) -> None:
        """
        Recycle treatment:
          - variables: concatenation of all recycle edge flow vectors
          - function: difference between recomputed tear streams and guessed tear streams
        """
        rec_edges = self._list_recycle_edges()
        if not rec_edges:
            # cycle-free case fell through — just do forward
            self._forward_propagation_without_cycles()
            return

        num_comp = self.env_config.max_number_of_components
        def pack(vecs: list[np.ndarray]) -> np.ndarray:
            return np.concatenate([x.astype(float) for x in vecs]) if vecs else np.zeros(0)

        def unpack(x: np.ndarray) -> list[np.ndarray]:
            return [x[i*num_comp:(i+1)*num_comp] for i in range(len(rec_edges))]

        # initial guess:
        if self._last_recycle_guess is None or len(self._last_recycle_guess) != len(rec_edges)*num_comp:
            # take any existing stream on that edge, else zeros
            guess_vecs = []
            for u, v, k, d in rec_edges:
                flow = d.get("stream", {}).get("flow")
                if flow is None:
                    flow = np.zeros(num_comp)
                guess_vecs.append(np.array(flow, dtype=float))
            self._last_recycle_guess = pack(guess_vecs)

        def tear_residual(x: np.ndarray) -> np.ndarray:
            # Defensive: ensure right shape and nonnegativity only at assignment time
            vecs = unpack(x)  # list of arrays, len == len(rec_edges)
            if any(v.shape[0] != num_comp for v in vecs):
                raise ValueError("Recycle guess has wrong dimensionality.")

            # assign x to recycle edges
            for i, (u, v, k, d) in enumerate(rec_edges):
                d.setdefault("stream", {})
                # clip here; keep the residual continuous in regions x>=0
                d["stream"]["flow"] = np.maximum(vecs[i], 0.0)

            # run one forward pass where recycle edges act as fixed inbound streams
            self._forward_propagation_given_recycle_edges()

            # read back source outputs for each recycle edge
            y_list = []
            for i, (u, v, k, d) in enumerate(rec_edges):
                out_lbl = d.get("output_label")
                src_out = self.graph.nodes[u]["output_flows"].get(out_lbl)
                if src_out is None:
                    src_out = np.zeros(num_comp)
                y_list.append(np.asarray(src_out, dtype=float))
            y = pack(y_list)

            return x - y  # fixed point

        # root find
        out = opt.fsolve(
            tear_residual,
            self._last_recycle_guess,
            full_output=True,
            maxfev=self.env_config.max_num_root_finding_interactions,
        )
        x_sol, info, ier, mesg = out
        fvec = info.get("fvec", np.zeros_like(x_sol))
        self._last_recycle_guess = x_sol.copy()

        # Primary: fsolve’s status; Secondary: residual norm
        if ier != 1 or np.sum(np.abs(fvec)) > self.env_config.epsilon * max(1, len(x_sol)):
            # optional Wegstein fallback
            if self.env_config.use_wegstein and len(x_sol) > 0:
                x = self._last_recycle_guess.copy()
                converged = False
                for _ in range(self.env_config.wegstein_steps):
                    r = tear_residual(x)
                    y = x - r  # fixed-point map
                    new_x = (
                            self.env_config.wegstein_constant * x
                            + (1 - self.env_config.wegstein_constant) * y
                    )
                    # Better stop criterion: residual size
                    if np.sum(np.abs(tear_residual(new_x))) < self.env_config.epsilon * len(new_x):
                        converged = True
                        x = new_x
                        break
                    x = new_x
                if not converged:
                    raise RuntimeError(f"Recycle tear solving did not converge: {mesg}")
                self._last_recycle_guess = x.copy()
            else:
                raise RuntimeError(f"Recycle tear solving did not converge: {mesg}")

        # with solution, set recycle streams and run final forward
        vecs = unpack(self._last_recycle_guess)
        for i, (u, v, k, d) in enumerate(rec_edges):
            d.setdefault("stream", {})
            d["stream"]["flow"] = np.maximum(vecs[i], 0.0)

        self._forward_propagation_given_recycle_edges()

    def _forward_propagation_given_recycle_edges(self) -> None:
        """
        Same as forward pass, but inbound streams may include pre-set recycle edges.
        """
        # for safety, clear all non-feed outputs before recomputing
        for nid in self.graph.nodes:
            ut = self.graph.nodes[nid].get("unit_type")
            if ut and ut != "feed":
                of = self.graph.nodes[nid].get("output_flows", {})
                for k in of.keys():
                    of[k] = None

        # we can't topo-sort with cycles; iterate nodes a few passes
        # light Gauss-Seidel: update nodes when all non-recycle inputs are available
        nodes = list(self.graph.nodes)
        num_pass = max(3, len(nodes))
        for _ in range(num_pass):
            any_updated = False
            for node_id in nodes:
                ut = self.graph.nodes[node_id].get("unit_type")
                if ut is None or ut == "feed":
                    continue

                # ready check: all non-recycle inputs have upstream outputs; recycle inputs have stream flow
                ready = True
                for u, _, _, d in self.graph.in_edges(node_id, keys=True, data=True):
                    if d.get("is_recycle", False):
                        if "flow" not in d.get("stream", {}):
                            ready = False
                            break
                    else:
                        lbl = d.get("output_label")
                        if self.graph.nodes[u]["output_flows"].get(lbl) is None:
                            ready = False
                            break
                if not ready:
                    continue

                total_input = self._sum_inbound_streams(node_id)
                outputs = self._simulate_unit(node_id, ut, total_input, self.graph.nodes[node_id].get("params", {}))

                # Safety: enforce non-negativity and check arity
                outputs = [np.maximum(np.asarray(o, dtype=float), 0.0) for o in outputs]
                out_labels = sorted(self.graph.nodes[node_id]["output_flows"].keys())
                if len(outputs) != len(out_labels):
                    raise RuntimeError(
                        f"Unit '{ut}' at node {node_id} returned {len(outputs)} outputs, "
                        f"but node declares {len(out_labels)}."
                    )

                # write outputs
                for k, lbl in enumerate(out_labels):
                    prev = self.graph.nodes[node_id]["output_flows"].get(lbl)
                    newv = outputs[k]
                    self.graph.nodes[node_id]["output_flows"][lbl] = newv
                    # detect update (for early stop)
                    if prev is None or np.any(np.abs(newv - prev) > self.env_config.epsilon):
                        any_updated = True

                # push downstream
                for u, v, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                    lbl = data.get("output_label")
                    if lbl in self.graph.nodes[u]["output_flows"]:
                        data.setdefault("stream", {})
                        data["stream"]["flow"] = np.array(self.graph.nodes[u]["output_flows"][lbl], dtype=float)

            if not any_updated:
                break  # early stop if nothing changed this sweep

    # ----- unit simulation -----

    def _simulate_unit(self, node_id: int, unit_type: str, input_flow: np.ndarray, params: Dict[str, Any]):
        input_flow = np.maximum(np.array(input_flow, dtype=float), 0.0)

        if unit_type == "mixer":
            # already summed by inbound handling
            return [input_flow]

        if unit_type == "split":
            sr = float(params.get("split_ratio", 0.5))
            outs = units.split(feed_molar_flowrates=input_flow, split_ratio=sr)
            return outs

        if unit_type == "add_solvent":
            # params: index_new_component (global), solvent_amount
            idx_global = int(params["index_new_component"])
            amount = float(params["solvent_amount"])

            flow = input_flow.copy()
            last_pos = self.env_config.max_number_of_components - 1

            # Fixed-amount semantics (cap to requested total of last component)
            inc = amount - flow[last_pos]
            if inc > 0:
                flow[last_pos] += inc

            # If this solvent isn't in the flowsheet set yet, add it
            if idx_global not in self.current_indices:
                if len(self.current_indices) >= self.env_config.max_number_of_components:
                    raise RuntimeError("Cannot add a new component: flowsheet is at max_number_of_components.")
                self.current_indices.append(idx_global)

            # Rebuild the active PEQ for the *new* component set (like original env)
            self._set_active_phase_eq_from_current_indices()

            # Refresh state metadata (critical/gamma vectors)
            self._refresh_system_metadata()

            # (Optional) stash info on the node for traceability
            self.graph.nodes[node_id]["phase_eq"] = self.current_phase_eq

            return [np.maximum(flow, 0.0)]

        if unit_type == "distillation_column":
            if self.current_phase_eq is None:
                raise RuntimeError("No active phase-equilibrium model set for current components.")
            peq_order = self._peq_indices()
            if peq_order is None:
                raise RuntimeError("Active PEQ has no component index order ('indices_components').")

            # choose the VLE object from structure
            if "vle" not in self.current_phase_eq or "phase_eq" not in self.current_phase_eq["vle"]:
                raise RuntimeError("Active PEQ has no VLE data ('vle' -> 'phase_eq').")
            vle_model = self.current_phase_eq["vle"]["phase_eq"]

            fs_order = self.current_indices
            feed_peq = units.transform_stream_fs_to_stream_phase_eq(
                molar_flowrates_flowsheet=input_flow,
                order_components_flowsheet=fs_order,
                phase_eq_order_components=peq_order
            )

            df = float(params.get("df", 0.5))
            out_peq = units.distillation(
                transformed_feed_flowrates=feed_peq,
                df=df,
                column=self.env_config.distillation_column,
                current_vle=vle_model
            )

            outputs = []
            for s in out_peq:
                s_fs = units.transform_stream_phase_eq_to_stream_fs(
                    molar_flowrates_phase_eq=s,
                    phase_eq_order_components=peq_order,
                    order_components_flowsheet=fs_order,
                    max_num_components_flowsheet=self.env_config.max_number_of_components
                )
                outputs.append(np.maximum(s_fs, 0.0))

            if len(self.current_indices) == 2:
                for k in range(len(outputs)):
                    outputs[k][-1] = 0.0

            # (optional) keep for traceability
            self.graph.nodes[node_id]["phase_eq"] = self.current_phase_eq
            return outputs

        if unit_type == "decanter":
            if self.current_phase_eq is None:
                raise RuntimeError("No active phase-equilibrium model set for current components.")
            peq_order = self._peq_indices()
            if peq_order is None:
                raise RuntimeError("Active PEQ has no component index order ('indices_components').")

            if "lle" not in self.current_phase_eq or "phase_eq" not in self.current_phase_eq["lle"]:
                raise RuntimeError("Active PEQ has no LLE data ('lle' -> 'phase_eq').")
            lle_model = self.current_phase_eq["lle"]["phase_eq"]

            fs_order = self.current_indices
            feed_peq = units.transform_stream_fs_to_stream_phase_eq(
                molar_flowrates_flowsheet=input_flow,
                order_components_flowsheet=fs_order,
                phase_eq_order_components=peq_order
            )

            out_peq = units.decantation(
                transformed_feed_molar_flowrates=feed_peq,
                current_phase_eq_liq=lle_model
            )

            outputs = []
            for s in out_peq:
                s_fs = units.transform_stream_phase_eq_to_stream_fs(
                    molar_flowrates_phase_eq=s,
                    phase_eq_order_components=peq_order,
                    order_components_flowsheet=fs_order,
                    max_num_components_flowsheet=self.env_config.max_number_of_components
                )
                outputs.append(np.maximum(s_fs, 0.0))

            if len(self.current_indices) == 2:
                for k in range(len(outputs)):
                    outputs[k][-1] = 0.0

            self.graph.nodes[node_id]["phase_eq"] = self.current_phase_eq
            return outputs

        # unknown unit or passthrough
        return [input_flow]

    # ----- helpers -------

    def _new_node_id(self) -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _refresh_system_metadata(self) -> None:
        names = self.env_config.phase_eq_generator.names_components
        max_comp = self.env_config.max_number_of_components

        # --- pure critical data (TC, PC, ω) per present component ---
        pure_blocks: list[np.ndarray] = []
        for gidx in self.current_indices:
            name = names[gidx]
            crit = np.asarray(self.env_config.dict_pure_component_data[name]["critical_data"], dtype=float)
            pure_blocks.append(crit)

        # All critical blocks must have same length
        block_len = pure_blocks[0].size if pure_blocks else 3
        if any(b.size != block_len for b in pure_blocks):
            raise ValueError("Inconsistent critical_data vector length among components.")

        # pad to max components with zero-blocks
        while len(pure_blocks) < max_comp:
            pure_blocks.append(np.zeros(block_len, dtype=float))

        pure_flat = np.concatenate(pure_blocks, axis=None) if pure_blocks else np.zeros(0, dtype=float)

        # --- γ(∞) interaction pairs ---
        # Choose temperature from feed situation
        fs_idx = self.feed_stream_information.get("feed_situation_index")
        temps = self.env_config.phase_eq_generator.subsystems_temperatures
        if fs_idx is None or not (0 <= int(fs_idx) < len(temps)):
            raise ValueError("feed_situation_index is missing or out of range for subsystems_temperatures.")
        T = temps[int(fs_idx)]

        # only present components (flowsheet order)
        pair_list: list[np.ndarray] = []
        n_present = len(self.current_indices)
        for i in range(n_present):
            for j in range(i + 1, n_present):
                name_i = names[self.current_indices[i]]
                name_j = names[self.current_indices[j]]
                gij = np.asarray(
                    self.env_config.phase_eq_generator.compute_inf_dilution_act_coeffs(name_i, name_j, T),
                    dtype=float,
                )
                # Expect exactly [γ(i→j), γ(j→i)]
                if gij.size != 2:
                    raise ValueError("compute_inf_dilution_act_coeffs must return a 2-length vector.")
                pair_list.append(gij)

        # pad pairs to full size implied by max components
        full_pairs = (max_comp * (max_comp - 1)) // 2
        while len(pair_list) < full_pairs:
            pair_list.append(np.zeros(2, dtype=float))

        gamma_flat = np.concatenate(pair_list, axis=None) if pair_list else np.zeros(2 * full_pairs, dtype=float)

        # store
        self.system_metadata["indices"] = list(self.current_indices)
        self.system_metadata["pure_critical"] = pure_flat
        self.system_metadata["gamma_inf"] = gamma_flat

        # Optional: legacy aliases for tools/testers expecting old keys
        # self.system_metadata["present_component_indices"] = self.system_metadata["indices"]
        # self.system_metadata["critical_data_vector"] = self.system_metadata["pure_critical"]
        # self.system_metadata["interactions_vector"] = self.system_metadata["gamma_inf"]

    def _has_cycles(self) -> bool:
        try:
            nx.find_cycle(self.graph, orientation="original")
            return True
        except nx.exception.NetworkXNoCycle:
            return False

    def _list_recycle_edges(self) -> list[tuple[int, int, int, dict]]:
        return [(u, v, k, d) for u, v, k, d in self.graph.edges(keys=True, data=True)
                if d.get("is_recycle", False)]

    def _sum_inbound_streams(self, node_id: int) -> np.ndarray:
        num_comp = self.env_config.max_number_of_components
        acc = np.zeros(num_comp, dtype=float)

        for u, v, key, data in self.graph.in_edges(node_id, keys=True, data=True):
            if data.get("is_recycle", False):
                flow = data.get("stream", {}).get("flow")
                if flow is not None:
                    acc = acc + np.array(flow, dtype=float)
            else:
                lbl = data.get("output_label")
                src_out = self.graph.nodes[u]["output_flows"].get(lbl)
                if src_out is not None:
                    acc = acc + np.array(src_out, dtype=float)

        return np.maximum(acc, 0.0)

    def _net_fresh_solvent_added(self, factor_mol: float = 1e6):
        """
        Compute net fresh solvent makeup from all add_solvent nodes.

        Definition:
            net fresh solvent added = max(solvent_out - solvent_in, 0)

        This correctly handles recycle into an add_solvent node:
            target/source solvent = 9
            recycled solvent entering add_solvent = 1
            net fresh solvent makeup = 8

        Returns:
            added_vector_mol:
                vector in flowsheet order with net fresh solvent added
            total_added_mol:
                scalar molar amount of net fresh solvent makeup
            total_added_kg_per_hr:
                scalar kg/hr equivalent, using factor_mol
            by_node:
                debug dictionary per add_solvent node
        """
        num_comp = self.env_config.max_number_of_components
        enable_cost_debug = bool(
            getattr(self.env_config, "enable_cost_debug", False)
        )

        added_vector_mol = np.zeros(num_comp, dtype=float)
        total_added_mol = 0.0
        total_added_kg_per_hr = 0.0
        by_node = {}

        for nid, node_data in self.graph.nodes(data=True):
            if node_data.get("unit_type") != "add_solvent":
                continue

            params = node_data.get("params", {}) or {}
            if "index_new_component" not in params:
                continue

            solvent_global_index = int(params["index_new_component"])

            if solvent_global_index not in self.current_indices:
                # This should not happen after a successful simulation, but skip safely.
                continue

            solvent_fs_index = self.current_indices.index(solvent_global_index)

            out = node_data.get("output_flows", {}).get("out0")
            if out is None:
                continue

            inp = self._sum_inbound_streams(nid)
            out = np.asarray(out, dtype=float)

            input_solvent_mol = float(inp[solvent_fs_index])
            output_solvent_mol = float(out[solvent_fs_index])
            net_added_mol = max(output_solvent_mol - input_solvent_mol, 0.0)

            solvent_only_vector = np.zeros(num_comp, dtype=float)
            solvent_only_vector[solvent_fs_index] = net_added_mol

            net_added_kg_per_hr = float(
                np.sum(self._convert_mol_flow_to_kg(solvent_only_vector, factor_mol=factor_mol))
            )

            added_vector_mol[solvent_fs_index] += net_added_mol
            total_added_mol += net_added_mol
            total_added_kg_per_hr += net_added_kg_per_hr

            # Always keep the minimal information needed for solvent-cost accounting.
            # Only attach the heavier diagnostic payload when cost debug is enabled.
            by_node[nid] = {
                "solvent_global_index": solvent_global_index,
                "solvent_fs_index": solvent_fs_index,
                "net_added_mol": net_added_mol,
                "net_added_kg_per_hr": net_added_kg_per_hr,
            }

            if enable_cost_debug:
                by_node[nid].update({
                    "requested_solvent_amount": float(params.get("solvent_amount", 0.0)),
                    "input_solvent_mol": input_solvent_mol,
                    "output_solvent_mol": output_solvent_mol,
                    "input_flow": inp.copy(),
                    "output_flow": out.copy(),
                })

        self.solvent_makeup_debug = copy.deepcopy(by_node) if enable_cost_debug else None

        return added_vector_mol, total_added_mol, total_added_kg_per_hr, by_node

    def _consume_open_stream(self, node_id: int, label: str) -> None:
        """
        Sanity guard: ensure (node,label) exists as an output. We don't hard "close"
        here; simulate() considers an output "open" iff it has no consumer edge.
        """
        node = self.graph.nodes.get(node_id, {})
        of = node.get("output_flows", {})
        if label not in of:
            raise RuntimeError(f"Stream {label} not found on node {node_id}.")

        # ensure no existing consumer edge already uses this (node,label), commented out now for speed
        # for _, _, _, d in self.graph.out_edges(node_id, keys=True, data=True):
        #     if d.get("output_label") == label:
        #         raise RuntimeError(f"Stream ({node_id}, {label}) is already connected to a consumer.")

    def _peq_indices(self) -> list:
        """
        Return the component index order for the active phase-eq model.
        Normalizes across possible layouts:
          - self.current_phase_eq["indices"]
          - self.current_phase_eq["vle"]["indices_components"]
          - self.current_phase_eq["lle"]["indices_components"]
        """
        peq = getattr(self, "current_phase_eq", None)
        if not peq:
            return None
        if "indices" in peq and peq["indices"] is not None:
            return peq["indices"]
        # fallbacks to existing structure
        if "vle" in peq and isinstance(peq["vle"], dict):
            if "indices_components" in peq["vle"]:
                return peq["vle"]["indices_components"]
        if "lle" in peq and isinstance(peq["lle"], dict):
            if "indices_components" in peq["lle"]:
                return peq["lle"]["indices_components"]
        return None

    # ----- open streams / destinations ------

    def get_open_streams(self) -> list[Tuple[int, str]]:
        """
        Streams that currently leave a node and do not yet have *any* consumer edge.
        If a recycle edge is attached to an output, that output is NOT an open product.
        """
        opens: list[Tuple[int, str]] = []
        for nid in self.graph.nodes:
            of = self.graph.nodes[nid].get("output_flows", {})
            for lbl, arr in of.items():
                if arr is None:
                    continue
                # If there is any outgoing edge (recycle or not) from (nid,lbl), it's consumed
                consumed = False
                for _, v, _, d in self.graph.out_edges(nid, keys=True, data=True):
                    if d.get("output_label") == lbl:
                        consumed = True
                        break
                if not consumed:
                    opens.append((nid, lbl))
        return opens

    def get_units_with_available_input(self, exclude: Optional[int] = None, max_inputs: int = None) -> list:
        """
        Return units that currently already have a certain number of inputs (currently 3 (more should be impossible),
        excluding feeds and optionally excluding a particular node_id.
        """
        eligible = []
        for nid in self.graph.nodes:
            ut = self.graph.nodes[nid].get("unit_type")
            if ut is None or ut == "feed":
                continue
            if exclude is not None and nid == exclude:
                continue
            if self.graph.in_degree(nid) <= max_inputs:
                eligible.append(nid)
        return eligible

    def _set_active_phase_eq_from_current_indices(self) -> None:
        names = [self.env_config.phase_eq_generator.names_components[i] for i in self.current_indices]
        peq = self.env_config.phase_eq_generator.search_subsystem_phase_eq(names)
        # normalize indices once
        idx = (peq.get("indices")
               or (peq.get("vle") or {}).get("indices_components")
               or (peq.get("lle") or {}).get("indices_components"))
        peq["indices"] = idx
        self.current_phase_eq = peq

    # ----- mass balance & NPV ------

    def _mass_balance_check(self) -> None:
        """
        Compare base feed + solvent added vs sum of all *leaving* product streams.
        """

        num_comp = self.env_config.max_number_of_components

        # base feed (sum of feed nodes 'out0')
        base_feed = np.zeros(num_comp, dtype=float)
        for nid in self.graph.nodes:
            if self.graph.nodes[nid].get("unit_type") == "feed":
                f = self.graph.nodes[nid]["output_flows"].get("out0")
                if f is not None:
                    base_feed += np.array(f, dtype=float)

        # Net fresh solvent makeup, accounting for recycle into add_solvent nodes.
        added, _, _, _ = self._net_fresh_solvent_added(factor_mol=1e6)

        effective_feed = base_feed + added

        # total leaving products = open streams that are not consumed downstream (excluding recycle-only)
        total_products = np.zeros(num_comp, dtype=float)
        for nid, lbl in self.get_open_streams():
            flow = self.graph.nodes[nid]["output_flows"].get(lbl)
            if flow is not None:
                total_products += np.array(flow, dtype=float)

        diff = effective_feed - total_products
        total_feed_sum = float(np.sum(base_feed))
        atol = max(self.env_config.mb_atol,
                   self.env_config.mb_relative_percent * total_feed_sum)
        rtol = self.env_config.mb_rtol

        if not np.allclose(total_products, effective_feed, rtol=rtol, atol=atol):
            raise ValueError(
                f"Mass balance VIOLATION (severe): diff={diff}, threshold={atol}"
            )

    def compute_npv(self):
        """
        Graph-version of the generic/literature NPV with EnvConfig-driven
        prices and unit costs.

        Important accounting rule:
            credit_solvent_product = False
                Pure solvent leaving the process is tracked, but it does not
                receive economic credit and does not improve performance_ratio.

            credit_solvent_product = True
                Pure solvent leaving the process is treated as useful recovered
                material, closer to the previous behavior.

        Fresh solvent makeup is always charged. Recycle into add_solvent reduces
        fresh solvent makeup through _net_fresh_solvent_added().
        """
        # --- common knobs / helpers ---
        epsilon_for_flowrates = 1e-4
        maxC = self.env_config.max_number_of_components
        names = self.env_config.phase_eq_generator.names_components

        credit_solvent_product = bool(
            getattr(self.env_config, "credit_solvent_product", False)
        )
        enable_cost_debug = bool(
            getattr(self.env_config, "enable_cost_debug", False)
        )

        # Feed components before any solvent was added.
        feed_comp_global = list(self.feed_stream_information.get("indices_components_in_feeds", []))
        num_comps_in_feed = len(feed_comp_global)
        feed_global_indices = set(int(x) for x in feed_comp_global)

        # Solvent components are the components introduced by add_solvent nodes.
        solvent_global_indices = set()
        for _, node_data in self.graph.nodes(data=True):
            if node_data.get("unit_type") != "add_solvent":
                continue
            params = node_data.get("params", {}) or {}
            if "index_new_component" in params:
                solvent_global_indices.add(int(params["index_new_component"]))

        # Where the current solvent lives in flowsheet order.
        # Current environment allows one added solvent, so this is the last active slot.
        def _solvent_fs_index():
            if len(self.current_indices) <= num_comps_in_feed:
                return None
            return len(self.current_indices) - 1

        # Gather all open product/leaving streams.
        def _leaving_streams():
            leaving = []
            for nid, lab in self.get_open_streams():
                src_node = self.graph.nodes[nid]
                flow = src_node.get("output_flows", {}).get(lab, None)
                if flow is not None:
                    leaving.append((nid, lab, flow))
            return leaving

        # Sum base feed before any add_solvent.
        base_feed_total = np.zeros(maxC, dtype=float)
        for feed_id in getattr(self, "feed_nodes", []):
            f = self.graph.nodes[feed_id].get("output_flows", {}).get("out0")
            if f is not None:
                base_feed_total += np.asarray(f, dtype=float)

        # Shared net fresh solvent makeup calculation.
        (
            net_solvent_added_vector_mol,
            net_solvent_added_mol,
            net_solvent_added_kg_per_hr,
            solvent_makeup_by_node,
        ) = self._net_fresh_solvent_added(factor_mol=1e6)

        solv_fs_idx = _solvent_fs_index()

        def _unit_nodes():
            for nid, nd in self.graph.nodes(data=True):
                ut = nd.get("unit_type")
                if ut and ut != "feed":
                    yield nid, ut, nd

        version = getattr(self.env_config, "npv_version", "generic").lower()

        # Performance-ratio numerator components.
        # sum_n_leaving only includes material counted as useful recovered material.
        sum_n_leaving = 0.0
        sum_n_feed_products_leaving = 0.0
        sum_n_solvent_leaving = 0.0
        sum_n_solvent_credited = 0.0
        sum_n_solvent_credit_blocked = 0.0

        # Performance-ratio denominator.
        sum_n_solvent_added = net_solvent_added_mol

        if version == "generic":
            unit_costs = getattr(self.env_config, "unit_costs_generic", {}) or {}
            product_price = getattr(self.env_config, "product_price_per_component", {}) or {}
            solvent_cost_per_mol = getattr(self.env_config, "solvent_cost_per_component_mol", {}) or {}

            specification_pure = 0.99
            specification_solvent = 0.99
            weight_pure_component = 1000.0
            weight_solvent = 100.0

            gain_leaving_stream = 0.0
            cost_units = 0.0
            cost_solvent_added = 0.0
            gain_solvent_released = 0.0
            gain_solvent_blocked = 0.0

            # Debug kg terms for parity with literature branch.
            kg_solvent_released_per_hr = 0.0
            kg_solvent_credited_per_hr = 0.0
            kg_solvent_credit_blocked_per_hr = 0.0

            # Leaving streams.
            for nid, lab, flow in _leaving_streams():
                flow = np.asarray(flow, dtype=float)
                tot = float(np.sum(flow))
                if tot <= epsilon_for_flowrates:
                    continue

                y = flow / max(tot, 1e-12)

                # Pure feed-component product.
                rel_feed = y[:num_comps_in_feed]
                if rel_feed.size and float(np.max(rel_feed)) > specification_pure:
                    j_best = int(np.argmax(rel_feed))
                    best_global = self.current_indices[j_best] if j_best < len(self.current_indices) else None
                    price = float(product_price.get(best_global, weight_pure_component))
                    purity = float(np.max(rel_feed))

                    gain_leaving_stream += price * tot * purity
                    sum_n_feed_products_leaving += tot
                    sum_n_leaving += tot

                # Pure solvent outlet.
                if solv_fs_idx is not None and solv_fs_idx < len(y):
                    y_solv = float(y[solv_fs_idx])
                    if y_solv > specification_solvent:
                        solvent_mol = tot * y_solv
                        solvent_kg = float(
                            np.sum(self._convert_mol_flow_to_kg(
                                np.eye(1, maxC, solv_fs_idx, dtype=float).ravel() * solvent_mol,
                                factor_mol=1e6,
                            ))
                        )

                        sum_n_solvent_leaving += tot
                        kg_solvent_released_per_hr += solvent_kg

                        g = weight_solvent * solvent_mol
                        if credit_solvent_product:
                            gain_solvent_released += g
                            sum_n_solvent_credited += tot
                            sum_n_leaving += tot
                            kg_solvent_credited_per_hr += solvent_kg
                        else:
                            gain_solvent_blocked += g
                            sum_n_solvent_credit_blocked += tot
                            kg_solvent_credit_blocked_per_hr += solvent_kg

            # Unit capital costs.
            for _, ut, _ in _unit_nodes():
                cost_units += float(unit_costs.get(ut, 0.0))

            # Fresh solvent makeup cost.
            for info in solvent_makeup_by_node.values():
                gidx = int(info["solvent_global_index"])
                added_mol = float(info["net_added_mol"])
                mol_cost = float(solvent_cost_per_mol.get(gidx, 0.0))
                cost_solvent_added += mol_cost * added_mol

            # Performance ratio.
            sum_n_feed = float(np.sum(base_feed_total))
            self.performance_ratio = sum_n_leaving / max(sum_n_feed + sum_n_solvent_added, 1e-12)

            # Assemble NPV.
            npv = 0.0
            self.npv_without_app_cost = 0.0

            npv += gain_leaving_stream
            self.npv_without_app_cost += gain_leaving_stream

            npv += gain_solvent_released - cost_solvent_added
            self.npv_without_app_cost += gain_solvent_released - cost_solvent_added

            npv -= cost_units

            normed = max(0.0, npv) / 1000.0

            if enable_cost_debug:
                self.cost_debug = {
                    "version": version,
                    "credit_solvent_product": credit_solvent_product,
                    "feed_global_indices": sorted(feed_global_indices),
                    "solvent_global_indices": sorted(solvent_global_indices),

                    "sum_n_feed": sum_n_feed,
                    "sum_n_leaving": sum_n_leaving,
                    "sum_n_feed_products_leaving": sum_n_feed_products_leaving,
                    "sum_n_solvent_leaving": sum_n_solvent_leaving,
                    "sum_n_solvent_credited": sum_n_solvent_credited,
                    "sum_n_solvent_credit_blocked": sum_n_solvent_credit_blocked,
                    "sum_n_solvent_added": sum_n_solvent_added,
                    "performance_ratio_denominator": sum_n_feed + sum_n_solvent_added,
                    "performance_ratio": self.performance_ratio,

                    "net_solvent_added_vector_mol": net_solvent_added_vector_mol.copy(),
                    "net_solvent_added_mol": net_solvent_added_mol,
                    "net_solvent_added_kg_per_hr": net_solvent_added_kg_per_hr,
                    "solvent_makeup_by_node": copy.deepcopy(solvent_makeup_by_node),

                    "kg_solvent_released_per_hr": kg_solvent_released_per_hr,
                    "kg_solvent_credited_per_hr": kg_solvent_credited_per_hr,
                    "kg_solvent_credit_blocked_per_hr": kg_solvent_credit_blocked_per_hr,

                    "gain_leaving_stream": gain_leaving_stream,
                    "gain_solvent_released": gain_solvent_released,
                    "gain_solvent_blocked": gain_solvent_blocked,
                    "cost_solvent_added": cost_solvent_added,
                    "cost_units": cost_units,
                    "npv": npv,
                    "npv_normed": normed,
                }
            else:
                self.cost_debug = None

            return npv, normed

        else:  # "literature"
            years = 10
            hr_per_year = 8000

            unit_costs = getattr(self.env_config, "unit_costs_literature", {}) or {}
            price_pure_component_per_kg = float(getattr(self.env_config, "lit_product_value_per_kg", 0.5))
            solvent_cost_per_kg = getattr(self.env_config, "solvent_cost_per_component_kg", {}) or {}

            specification_pure = 0.99
            specification_solvent = 0.99

            gain_leaving_stream = 0.0
            cost_units_total = 0.0

            kg_solvent_added_per_hr = net_solvent_added_kg_per_hr
            kg_solvent_released_per_hr = 0.0
            kg_solvent_credited_per_hr = 0.0
            kg_solvent_credit_blocked_per_hr = 0.0

            # Leaving streams: compute economic terms on mass basis.
            for nid, lab, flow in _leaving_streams():
                flow = np.asarray(flow, dtype=float)
                tot = float(np.sum(flow))
                if tot <= epsilon_for_flowrates:
                    continue

                flow_kg = self._convert_mol_flow_to_kg(flow, factor_mol=1e6)
                mass_total = float(np.sum(flow_kg))
                if mass_total <= 0.0:
                    continue

                mass_frac = flow_kg / max(mass_total, 1e-12)

                # Pure feed-component product.
                rel_mass_feed = mass_frac[:num_comps_in_feed]
                if rel_mass_feed.size and float(np.max(rel_mass_feed)) > specification_pure:
                    purity = float(np.max(rel_mass_feed))
                    gain_leaving_stream += (
                        price_pure_component_per_kg
                        * mass_total
                        * purity
                        * years
                        * hr_per_year
                    )
                    sum_n_feed_products_leaving += tot
                    sum_n_leaving += tot

                # Pure solvent outlet.
                if solv_fs_idx is not None and solv_fs_idx < len(mass_frac):
                    y_solv_m = float(mass_frac[solv_fs_idx])
                    if y_solv_m > specification_solvent:
                        solvent_kg_per_hr = mass_total * y_solv_m

                        kg_solvent_released_per_hr += solvent_kg_per_hr
                        sum_n_solvent_leaving += tot

                        if credit_solvent_product:
                            kg_solvent_credited_per_hr += solvent_kg_per_hr
                            sum_n_solvent_credited += tot
                            sum_n_leaving += tot
                        else:
                            kg_solvent_credit_blocked_per_hr += solvent_kg_per_hr
                            sum_n_solvent_credit_blocked += tot

            # Unit capex/opex.
            for nid, ut, nd in _unit_nodes():
                cap = float(unit_costs.get(ut, 0.0))

                inp_mol = self._sum_inbound_streams(nid)
                inp_kg = float(np.sum(self._convert_mol_flow_to_kg(inp_mol, factor_mol=1e6)))
                if inp_kg > epsilon_for_flowrates:
                    cap *= np.power(inp_kg / 25000.0, 0.6)  # 6/10 rule
                cost_units_total += cap

                if ut == "distillation_column":
                    out0 = nd.get("output_flows", {}).get("out0")
                    if out0 is not None:
                        heat_per_hr = 0.0
                        for j, gidx in enumerate(self.current_indices):
                            if j >= len(out0):
                                break
                            name = names[gidx]
                            factor = float(
                                self.env_config.dict_pure_component_data[name].get(
                                    "factor_heat_estimation_J_per_mol", 0.0
                                )
                            )
                            heat_per_hr += 2.0 * factor * float(out0[j]) * 1e6

                        wfac = float(
                            self.env_config.dict_pure_component_data["water"].get(
                                "factor_heat_estimation_J_per_mol", 0.0
                            )
                        )
                        if wfac > 0:
                            mol_water_per_hr = heat_per_hr / wfac
                            kg_water_per_hr = (
                                mol_water_per_hr
                                * float(self.env_config.dict_pure_component_data["water"]["M"])
                                / 1000.0
                            )
                            steam_price = float(unit_costs.get("steam_cost_per_kg", 0.0))
                            cost_units_total += steam_price * kg_water_per_hr * years * hr_per_year

            # Solvent price/cost basis.
            if solv_fs_idx is not None and solv_fs_idx < len(self.current_indices):
                solv_g = self.current_indices[solv_fs_idx]
                solv_price_kg = float(solvent_cost_per_kg.get(solv_g, 0.0))
            else:
                solv_g = None
                solv_price_kg = 0.0

            solvent_credit_minus_cost = (
                kg_solvent_credited_per_hr - kg_solvent_added_per_hr
            ) * solv_price_kg * years * hr_per_year

            # Aggregate NPV.
            npv_unscaled = 0.0
            self.npv_without_app_cost = 0.0

            npv_unscaled += gain_leaving_stream
            self.npv_without_app_cost += gain_leaving_stream

            npv_unscaled += solvent_credit_minus_cost
            self.npv_without_app_cost += solvent_credit_minus_cost

            npv_unscaled -= cost_units_total

            # Theoretical max NPV for normalization: all feed sold as pure product.
            theoretical_max_npv = 0.0
            for feed_id in getattr(self, "feed_nodes", []):
                f = self.graph.nodes[feed_id].get("output_flows", {}).get("out0")
                if f is None:
                    continue
                kg = self._convert_mol_flow_to_kg(f, factor_mol=1e6)
                theoretical_max_npv += (
                    price_pure_component_per_kg
                    * float(np.sum(kg))
                    * years
                    * hr_per_year
                )

            normed = (max(npv_unscaled, 0.0) / theoretical_max_npv) if theoretical_max_npv > 0 else 0.0

            # Scale raw NPV and npv_without_app_cost to M€.
            npv = npv_unscaled / 1000000.0
            self.npv_without_app_cost /= 1000000.0

            # Performance ratio.
            sum_n_feed = float(np.sum(base_feed_total))
            self.performance_ratio = sum_n_leaving / max(sum_n_feed + sum_n_solvent_added, 1e-12)

            if enable_cost_debug:
                self.cost_debug = {
                    "version": version,
                    "credit_solvent_product": credit_solvent_product,
                    "feed_global_indices": sorted(feed_global_indices),
                    "solvent_global_indices": sorted(solvent_global_indices),
                    "solvent_global_index_used_for_price": solv_g,

                    "sum_n_feed": sum_n_feed,
                    "sum_n_leaving": sum_n_leaving,
                    "sum_n_feed_products_leaving": sum_n_feed_products_leaving,
                    "sum_n_solvent_leaving": sum_n_solvent_leaving,
                    "sum_n_solvent_credited": sum_n_solvent_credited,
                    "sum_n_solvent_credit_blocked": sum_n_solvent_credit_blocked,
                    "sum_n_solvent_added": sum_n_solvent_added,
                    "performance_ratio_denominator": sum_n_feed + sum_n_solvent_added,
                    "performance_ratio": self.performance_ratio,

                    "net_solvent_added_vector_mol": net_solvent_added_vector_mol.copy(),
                    "net_solvent_added_mol": net_solvent_added_mol,
                    "net_solvent_added_kg_per_hr": net_solvent_added_kg_per_hr,
                    "solvent_makeup_by_node": copy.deepcopy(solvent_makeup_by_node),

                    "kg_solvent_added_per_hr": kg_solvent_added_per_hr,
                    "kg_solvent_released_per_hr": kg_solvent_released_per_hr,
                    "kg_solvent_credited_per_hr": kg_solvent_credited_per_hr,
                    "kg_solvent_credit_blocked_per_hr": kg_solvent_credit_blocked_per_hr,

                    "gain_leaving_stream": gain_leaving_stream,
                    "solvent_credit_minus_cost": solvent_credit_minus_cost,
                    "cost_units_total": cost_units_total,
                    "npv_unscaled": npv_unscaled,
                    "npv": npv,
                    "npv_normed": normed,
                }
            else:
                self.cost_debug = None

            return npv, normed

    def _convert_mol_flow_to_kg(self, flowrates_mol, factor_mol):
        """
        Convert a flowsheet-order molar flow vector to kg/hr, using the *current* indices
        and pure component molar masses from env_config.dict_pure_component_data.

        - flowrates_mol: array in flowsheet order (length max_number_of_components)
        - factor_mol: scale (legacy used 1e6 for Mmol/hr)
        """
        flowrates_mol = np.asarray(flowrates_mol, dtype=float)
        kg = np.zeros_like(flowrates_mol, dtype=float)

        for j, gidx in enumerate(self.current_indices):
            if j >= len(flowrates_mol):
                break
            name = self.env_config.phase_eq_generator.names_components[gidx]
            M_g_per_mol = float(self.env_config.dict_pure_component_data[name]["M"])
            kg[j] = M_g_per_mol * flowrates_mol[j] * factor_mol / 1000.0  # g/mol * mol/hr -> kg/hr

        return kg

    # ----- system data helpers -----

    def _collect_current_component_indices(self) -> list[int]:
        return list(self.current_indices)

    def _build_pure_crit_vector(self, indices: list[int]) -> np.ndarray:
        """
        [Tc0, Pc0, ω0, Tc1, Pc1, ω1, Tc2, Pc2, ω2, ...] up to max_number_of_components*3.
        Zeros for missing slots.
        """
        maxC = self.env_config.max_number_of_components
        out = np.zeros(3 * maxC, dtype=float)

        names = self.env_config.phase_eq_generator.names_components
        for slot, gidx in enumerate(indices[:maxC]):
            name = names[gidx]
            crit = self.env_config.dict_pure_component_data[name]["critical_data"]  # shape (3,)
            start = 3 * slot
            out[start:start + 3] = crit
        return out

    def _build_gamma_inf_vector(self, indices: list[int]) -> np.ndarray:
        """
        γinf vector concatenating (γ_ij, γ_ji) for all i<j among *present* components,
        padded with zeros to fixed length 2 * C(maxC, 2) = maxC*(maxC-1).

        Order: for present i<j in slot-order, append [γ_ij, γ_ji].
        """
        maxC = self.env_config.max_number_of_components
        max_len = maxC * (maxC - 1)  # 2*C(maxC,2)

        names = self.env_config.phase_eq_generator.names_components
        T = self.env_config.phase_eq_generator.subsystems_temperatures[
            self.feed_stream_information["feed_situation_index"]
        ]

        gammas = []
        # use the flowsheet slot-order (same order as indices list)
        n_present = min(len(indices), maxC)
        for i in range(n_present):
            for j in range(i + 1, n_present):
                n1 = names[indices[i]]
                n2 = names[indices[j]]
                gij, gji = self.env_config.phase_eq_generator.compute_inf_dilution_act_coeffs(n1, n2, T)
                gammas.extend([gij, gji])

        gammas = np.asarray(gammas, dtype=float)
        if gammas.size < max_len:
            gammas = np.pad(gammas, (0, max_len - gammas.size), mode="constant", constant_values=0.0)
        return gammas

    def _update_system_metadata_on_feed(self) -> None:
        """
        Compute and store fixed-length component metadata on the FEED node.

        Top-level (for programmatic access):
          - node['system_indices']
          - node['system_pure_crit']   (np.ndarray)
          - node['system_gammas_inf']  (np.ndarray)

        Also mirror into node['params'] as JSON-safe lists so the feed's params
        block isn't empty in dumps.
        """
        if not getattr(self, "feed_nodes", None):
            return  # nothing to attach to yet

        # ensure internal dict is up-to-date
        self._refresh_system_metadata()

        # build vectors using existing helpers
        indices = self._collect_current_component_indices()
        pure_vec = self._build_pure_crit_vector(indices)
        gamma_vec = self._build_gamma_inf_vector(indices)

        # attach to the first feed node
        feed_id = self.feed_nodes[0]
        node = self.graph.nodes[feed_id]

        # top-level (np arrays are fine here)
        node["system_indices"] = list(self.system_metadata["indices"])
        node["system_pure_crit"] = pure_vec
        node["system_gammas_inf"] = gamma_vec

        # mirrored into params as lists (JSON-safe for pretty printing)
        params = node.setdefault("params", {})
        params["system_indices"] = list(self.system_metadata["indices"])
        params["system_pure_crit"] = pure_vec.tolist() if hasattr(pure_vec, "tolist") else list(pure_vec)
        params["system_gammas_inf"] = gamma_vec.tolist() if hasattr(gamma_vec, "tolist") else list(gamma_vec)

    def get_system_feature_vector(self) -> np.ndarray:
        """
        Convenience accessor (for agent): returns [pure_crit, gammas_inf].
        If not yet computed, returns zeros of the correct size.
        """
        maxC = self.env_config.max_number_of_components
        pure_len = 3 * maxC
        gamma_len = maxC * (maxC - 1)

        if not getattr(self, "feed_nodes", None):
            return np.zeros(pure_len + gamma_len, dtype=float)

        feed_id = self.feed_nodes[0]
        node = self.graph.nodes[feed_id]
        pure = node.get("system_pure_crit", np.zeros(pure_len, dtype=float))
        gam = node.get("system_gammas_inf", np.zeros(gamma_len, dtype=float))
        return np.concatenate([pure, gam], axis=0)