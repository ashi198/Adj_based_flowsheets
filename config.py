import numpy as np 
import os, datetime, copy
import environment.phase_equilibria.phase_eq_handling as phase_eq_generation
from environment import units 
import torch

class GeneralConfig:

    """
    General configurations for the policy archiecture, training, and sampling 

    """
        

    def __init__(self, args):
        self.seed = args.seed

        # Network and environment
        self.latent_dim = 512 #latent dimension for Core transformer 
        self.num_transformer_blocks = 10 # Number of layers in the stack of transformer blocks for the architecture
        self.num_heads = 16 # Number of heads in the multihead attention.
        self.dropout = 0. # Dropout for feedforward layer in a transformer block.
        self.num_trf_flow_blocks = 3 #num of transformer blocks for flow expert 
        self.flow_latent_dim = 64

        # Loading trained checkpoints to resume training or evaluate
        self.load_checkpoint_from_path =  None #'./DONOT_DELETE_pretrain_with_reward_conditioning/all/best_model.pt'  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = True  # If True, the optimizer state is also loaded.

        # Training
        self.num_dataloader_workers = 3  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1,2,3"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = args.device  # Device on which to perform the supervised training
        self.num_epochs = args.epoches # Number of epochs (i.e., passes through training set) to train
        self.batch_size_training = 64 #Batch size to use for the supervised training during finetuning. 
        self.num_batches_per_epoch = None # Can be None, then we just do one pass through generated dataset

        self.wall_clock_limit = None
        self.mlflow_experiment = 'pretrain_wo_embeddings'
        self.steps = 50 #how many instances per systems should be generated within the test set
        self.eval_steps = args.eval_steps #how many instances per systems should be generated within the evaluation set
        self.balanced = False if args.subsystem != 'all' else True
        self.num_instances_per_batch = args.num_instances_per_batch
        self.if_pretrain = None
        self.evaluation_epoch = 10

        # Optimizer
        self.optimizer = {
            "lr": 1e-5,  # learning rate
            "weight_decay":  0.0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        # Self-improvement sequence decoding
        self.gumbeldore_config = {
            # Number of trajectories with the the highest objective function evaluation to keep for training
            "num_trajectories_to_keep": args.beam_search, # num_trajectories_to_keep for training PER instance 
            "keep_intermediate_trajectories": True,  # if True, we consider all intermediate, terminable trajectories
            "devices_for_workers": f"{args.device}" * 1,
            "destination_path": "./data",
            "batch_size_per_worker": 1, 
            "batch_size_per_cpu_worker": 1,
            "search_type": args.beam_search_type,
            "beam_width": args.beam_search,
            "replan_steps": 12,
            "num_rounds": 1,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # Only use for gumbeldore_eval=True below, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False, 
            "expert_min_obj": 0.0, 
            "replay_min_obj": 0.8, 
            "replay_top_k_per_instance": 5,
            "replay_every": 10

        }
        
        self.results_path = os.path.join(f"{args.results}", f"{args.subsystem}")
        os.makedirs(self.results_path, exist_ok=True)

        self.log_to_file = True


class EnvConfig:

    """
    Graph-first configuration for the flowsheet environment.
    - Loads phase equilibrium data (VLE/LLE) for allowed systems
    - Defines unit catalog, discretization of continuous specs, and action limits
    - Provides economics knobs for NPV calculation
    - Generates random feed problem instances
    """

    def __init__(self, args):

        # ----- Core dimensional settings -----

        # maximum number of components present simultaneously in a flowsheet
        self.max_number_of_components = 3
        self.max_simulator_tries = 50

        # ----- Phase equilibrium / property data -----
        all_subsystems = ['acetone_chloroform', 'ethanol_water', 'water_pyridine', 'n-butanol_water']
        self.systems_allowed = {}
        for sub in all_subsystems: 
            self.systems_allowed[sub] = False
            if sub == args.subsystem or args.subsystem == 'all':
                self.systems_allowed [sub] = True
                #self.systems_allowed ['n-butanol_water'] = False
        
        self.dicretization_parameter_lle = 5       # LLE simplex discretization
        self.curvature_parameter_vle = 0.001       # VLE curvature fitting

        self.phase_eq_generator = phase_eq_generation.PhaseEqHandling(
            directory=os.path.join(os.getcwd(), "environment", "phase_equilibria"),
            systems_allowed=self.systems_allowed
        )
        self.phase_eq_generator.load_phase_eqs(
            num_comp_lle=self.max_number_of_components,
            disc_para_lle=self.dicretization_parameter_lle,
            curvature_parameter=self.curvature_parameter_vle)

        # dict with pure component data (e.g., molar masses "M") used by literature NPV path
        self.dict_pure_component_data = self.phase_eq_generator.load_pure_component_data()
        self.component_names = list(self.dict_pure_component_data.keys())
        #self.component_names = ['acetone', 'benzene', 'chloroform', 'ethanol', 'n-butanol', 'pyridine', 'toluene', 'water', 'tetrahydrofuran']

        # make a components tensor for Add solvent 
        #self.component_names = list(self.dict_pure_component_data.keys())
        self.components_tensor = torch.tensor([self.dict_pure_component_data[name]["critical_data"] for name in self.component_names], dtype=torch.float32)
        self.gamma_mean = None # mean value of the entire training-set gamma values
        self.gamma_std = None #std of the entire training-set gamma values

        # Shuffle option for feed component order (usually keep False for stable tests)
        self.shuffle_order_of_components = False

        # ----- NPV & PRICING/COSTS -----
        # Choose NPV variant: "generic" (per-mole pricing) or "literature" (per-kg pricing)
        self.npv_version = "generic"  # or "literature"
        self.norm_npv = True  # also compute a normalized NPV
        self.credit_solvent_product = False  # if False, pure solvent leaving gets no product/performance credit
        self.enable_cost_debug = False  # keep False for training; set True only in manual/debug scripts
        self.allow_forward_recycles = False  # if False, recycle destinations must be upstream/lower node id

        # Build dynamic price/cost maps that always match the global component list
        names = self.phase_eq_generator.names_components
        self.num_components = len(names)

        # Uniform defaults (edit these two numbers to tune all components at once)
        _uniform_product_price_per_mol = 1000.0  # value for pure product streams (per mole) in "generic" mode
        _uniform_solvent_cost_per_mol = 100.0  # cost when a component is used as solvent (per mole) in "generic" mode

        _uniform_product_value_per_kg = 0.5  # value per kg in "literature" mode (set >0 to activate economics there)
        _uniform_solvent_cost_per_kg = 0.05  # cost per kg of solvent in "literature" mode

        # Generic (per-mole) pricing/costs: component-indexed dicts
        self.product_price_per_component = {idx: _uniform_product_price_per_mol for idx in range(self.num_components)}
        self.solvent_cost_per_component_mol = {idx: _uniform_solvent_cost_per_mol for idx in range(self.num_components)}

        # Literature (per-kg) pricing/costs
        self.lit_product_value_per_kg = _uniform_product_value_per_kg
        self.solvent_cost_per_component_kg = {idx: _uniform_solvent_cost_per_kg for idx in range(self.num_components)}

        self.steam_cost_per_kg = 0.04  # €/kg steam, used in literature NPV calc

        # Per-unit costs (used by compute_npv in the simulator)
        self.unit_costs_generic = {
            "add_solvent": 0.2,
            "distillation_column": 10.0,
            "decanter": 2.0,
            "split": 0.2,
            "mixer": 0.2
        }
        self.unit_costs_literature = {
            "add_solvent": 200000,
            "distillation_column": 1000000,
            "decanter": 200000,
            "split": 40000,
            "mixer": 40000,
            "steam_cost_per_kg": 0.04 # Energy/steam cost for distillation operating expenses
        }

        # Optional: override normalization scale for NPV (otherwise simulator uses total effective feed)
        # self.npv_normalization_scale = 1.0

        # ----- Unit catalog + discretization (RL action semantics) -----
        # Unit definitions: how many outputs, whether they need a continuous spec (range),
        # and which level to go to next.
        self.unit_types = {
            "distillation_column": {"num": 1, "output_streams": 2, "cont_range": [0.1, 0.95]},
            "decanter":            {"num": 1, "output_streams": 2, "cont_range": None},
            "split":               {"num": 1, "output_streams": 2, "cont_range": [0.1, 0.95]},
            "mixer":               {"num": 1, "output_streams": 1, "cont_range": None},
            "recycle":             {"num": 1, "output_streams": 1, "cont_range": None},
            "add_solvent":         {"num": 1,"output_streams": 1, "cont_range": [0.1, 1.95]},
        }

        self.outlet_to_idx = {"out0": 0, "out1": 1}
        self.max_outlets = 2

        self.distillation_column = units.distillation_column()

        # Stable index -> unit type mapping (flat catalog)
        self.units_map_indices_type = []
        for key in self.unit_types.keys():
            for _ in range(self.unit_types[key]["num"]):
                self.units_map_indices_type.append(key)

        self.num_units = len(self.unit_types)

        # index where "add_solvent" block starts
        self.add_solvent_start_index = None
        for i, key in enumerate(self.units_map_indices_type):
            if key == "add_solvent":
                self.add_solvent_start_index = i
                break

        # Action limits depending on the subsystem
        '''self.action_limits = {'acetone_chloroform': {'max_total_units': 7, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 3, 'max_solvent': 1}, 
                    'ethanol_water': {'max_total_units': 7, 'min_total_units': 3, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 3, 'max_solvent': 1},
                    'n-butanol_water': {'max_total_units': 5, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 1, 'max_split': 0, 'max_mixer': 0, 'max_recycle': 2, 'max_solvent': 1}, 
                    'water_pyridine': {'max_total_units': 5, 'min_total_units': 3, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 0, 'max_mixer': 0, 'max_recycle': 2, 'max_solvent': 1},
                    }'''
        

        '''self.action_limits = {'acetone_chloroform': {'max_total_units': 10, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 3, 'max_solvent': 2}, 
                    'ethanol_water': {'max_total_units': 10, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 3, 'max_solvent': 2},
                    'n-butanol_water': {'max_total_units': 10, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 3, 'max_solvent': 2}, 
                    'water_pyridine': {'max_total_units': 10, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 2, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 3, 'max_solvent': 2},
                    }'''
        

        self.action_limits = {'acetone_chloroform': {'max_total_units': 7, 'min_total_units': 3, 'max_distillation_columns': 3, 'max_decanters': 1, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 2, 'max_solvent': 1}, 
                    'ethanol_water': {'max_total_units': 7, 'min_total_units': 3, 'max_distillation_columns': 3, 'max_decanters': 1, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 2, 'max_solvent': 1},
                    'n-butanol_water': {'max_total_units': 7, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 1, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 2, 'max_solvent': 1}, 
                    'water_pyridine': {'max_total_units': 7, 'min_total_units': 2, 'max_distillation_columns': 3, 'max_decanters': 1, 'max_split': 1, 'max_mixer': 1, 'max_recycle': 2, 'max_solvent': 1},
                    }

        # ----- Recycle solver config -----
        # recycle guesses; see original env for semantics
        self.min_obj_for_terminate = 0.0
        self.random_guesses_root_iteration = 0
        self.max_num_root_finding_interactions = 50
        self.use_wegstein = False
        self.wegstein_constant = 0.5
        self.wegstein_steps = 500
        self.epsilon = 0.001

        # ----- Mass-balance tolerances (graph MB check) -----
        # feed-scaled tolerance (1% of total feed), with floors
        self.mb_relative_percent = 0.01
        self.mb_atol = 1e-6 # absolute tolerance
        self.mb_severe_atol = 1e-3
        self.mb_rtol = 1e-8 #relative tolerabce

        # List of mappings for params for distillation, split, add_solvent
        self.DF_distillation_map = self.get_continuous_grid([0.01, 0.99])
        self.split_ratio_map = self.get_continuous_grid([0.01, 0.99])
        self._amount_grid = self.get_continuous_grid([0.01, 4.99])
        self.add_solvent_comp_map = {
            name: self._amount_grid.copy()
            for name in self.component_names
        }

    def get_continuous_grid(self, cont_range, n_coarse=7, n_fine=7):
        alpha, beta = cont_range
        n = n_coarse * n_fine

        return alpha + np.arange(n) * (beta - alpha) / n

    def create_random_problem_instance(self, index):
        """
        sample a feed situation of format: [[indices from self.names_components for feed],
        [indices from self.names_components for add_component unit], number of feed streams]
        and return the situation index, feed streams, restrictions for add component unit and
        the names and order of the components in the streams
        """
        feed_streams = []

        # sample a feed situation
        #sampled_index = np.random.randint(len(self.phase_eq_generator.feed_situations))
        sampled_index = index
        sampled_situation = copy.deepcopy(self.phase_eq_generator.feed_situations[sampled_index])

        # shuffle order if specified
        if self.shuffle_order_of_components:
            np.random.shuffle(sampled_situation[0])

        # get names in feed streams
        names_in_streams = []
        for i in sampled_situation[0]:
            names_in_streams.append(self.phase_eq_generator.names_components[i])

        for i in range(sampled_situation[-1]):
            sampled_flowrates = np.random.rand(len(sampled_situation[0]))
            # normalize to 1 total flowrate
            sampled_flowrates = sampled_flowrates / (
                    sampled_situation[-1] * sum(sampled_flowrates))

            stream = np.zeros(self.max_number_of_components)
            stream[:len(sampled_flowrates)] = sampled_flowrates
            feed_streams.append(stream)

        return {"feed_situation_index": sampled_index,
                "indices_components_in_feeds": sampled_situation[0],
                "list_feed_streams": feed_streams,
                "possible_ind_add_comp": sampled_situation[1],
                "comp_order_feeds": names_in_streams,
                "system_name": "_".join(names_in_streams), 
                "lle_for_start": None,
                "vle_for_start": None}