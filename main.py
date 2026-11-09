import argparse, copy, os, time, ray, torch

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

from logger import Logger
from flowsheet_dataset import RandomDataset

import numpy as np
from config import GeneralConfig, EnvConfig

from core.gumbeldore_dataset import GumbeldoreDataset
from model.policy_arch import FlowsheetNetwork, dict_to_cpu
from utils import set_mlflow_connection, build_logit_tensors_per_level, generate_test_sets, batch_instances_for_dataset, set_seed, compute_gamma_normalization, batch_instances_for_dataset_eval


os.environ["RAY_DEDUP_LOGS"]="0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]="1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsystem",
                        type=str,
                        choices=['acetone_chloroform', 'ethanol_water', 'n-butanol_water', 'water_pyridine', 'all'],
                        default = 'all',
                        help="Specify benchmark task.")
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed")
    
    parser.add_argument("--epoches",
                        type=int,
                        default=100,
                        help="num of epoches to train")
    
    parser.add_argument("--device",
                        type=str,
                        default='cuda:2',
                        help="device")
    
    parser.add_argument("--beam_search",
                        type=int,
                        default=32,
                        help="64 or 128")
    
    parser.add_argument("--beam_search_type",
                        type=str,
                        default="tasar",
                        help="wor or tasar")
    
    parser.add_argument("--results",
                        type=str,
                        help="specify directory for storing results", 
                        default="./min_obj_0.7_WITHOUT_reward_conditioning_32_future_rewards")
    
    parser.add_argument("--num_instances_per_batch",
                        type=int,
                        help="specify directory for storing results", 
                        default=4)
    
    parser.add_argument("--if_leave_one_out",
                        type=bool,
                        help="whether run leave one out for unseen chemical system generalization", 
                        default=False)
    
    
    args = parser.parse_args()
    return args
    

def save_checkpoint(checkpoint: dict, filename: str, gen_config):
    os.makedirs(gen_config.results_path, exist_ok=True)
    path = os.path.join(gen_config.results_path, filename)
    torch.save(checkpoint, path)


def train_for_one_epoch(epoch: int, gen_config, env_config, network: FlowsheetNetwork, network_weights: dict,
                        optimizer: torch.optim.Optimizer, best_objective: float, train_instances: dict, destination_path: str):

    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config, train_instances)
    metrics, destination_full_path = gumbeldore_dataset.generate_dataset(
        network_weights,
        best_objective=best_objective,
        memory_aggressive=False, 
        if_test = False, epoch= epoch
    )

    print("Generated Flowsheets")
    print(f"Mean obj. over fresh best flowsheets: {metrics['mean_gen_obj']:.3f}")
    print(f"Best / worst obj. over fresh best flowsheets: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")

    torch.cuda.empty_cache()
    time.sleep(1)

    print("---- Loading dataset")
    dataset = RandomDataset(epoch, gen_config=gen_config, env_config=env_config, path_to_pickle=destination_full_path, batch_size=gen_config.batch_size_training,
                                    custom_num_batches=gen_config.num_batches_per_epoch, no_random= False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=False, persistent_workers=False)

    # Train for one epoch
    network.train()  

    # freeze layers except the last
    '''for parameter in network.parameters():
        parameter.requires_grad = False

    for parameter in network.state_encoder.parameters():
        parameter.requires_grad = False

    # unfreeze last 3 transformer block, terminate, open_stream, and unit experts heads
    for parameter in network.core_transformer[-3:].parameters():
        parameter.requires_grad = True
    
    for parameter in network.terminate_head.parameters():
        parameter.requires_grad = True

    for parameter in network.open_stream_head.parameters():
        parameter.requires_grad = True

    for expert in network.state_encoder.unit_experts.values():
        for parameter in expert.logit_linear.parameters():
            parameter.requires_grad = True'''

    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    accumulated_loss_lvl_three = 0
    accumulated_loss = 0

    count_lvl_zero = 0
    count_lvl_one = 0
    count_lvl_two = 0
    count_lvl_three = 0

    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    scaler = torch.amp.GradScaler()
    #criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
    criterion = CrossEntropyLoss(reduction="none", ignore_index=-1)

    if num_batches == 0:
        print(f"Epoch {epoch + 1}: No training batches available. Skipping training.")

        metrics.update({
            "loss_level_zero": float("nan"),
            "loss_level_one": float("nan"),
            "loss_level_two": float("nan"),
            "loss_level_three": float("nan"),
            "total_loss": float("nan"),
        })

        return metrics

    for _ in progress_bar:
        data = next(data_iter)
        indices_for_tracking = data.pop('indices_for_tracking')
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        
        # targets for the logit levels
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)
        target_three = data["target_three"][0].to(network.device)
        advantages = data["advantages"][0].to(network.device)

        # Convert advantages to positive weights
        weights = torch.exp(advantages.clamp(-2.0, 2.0) / 2.0)
        # Normalize to keep the loss scale stable
        weights = weights / weights.mean().detach()

        mask_zero = target_zero != -1
        mask_one = target_one != -1
        mask_two = target_two != -1
        mask_three = target_three != -1

        terminate_or_open_streams_logits, unit_predictions = network(input_data)
        
        #build logits based on selected units and streams 
        lvl_zero_logits, lvl_one_logits, lvl_two_logits, lvl_three_logits = build_logit_tensors_per_level(dataset= dataset, 
                                                                            input_data=input_data, terminate_or_open_streams_logits=terminate_or_open_streams_logits,
                                                                            unit_predictions=unit_predictions, indices_for_tracking = indices_for_tracking)
        
        lvl_zero_logits[input_data["feasibility_mask_level_zero"]] = float("-inf")
        lvl_one_logits[input_data["feasibility_mask_level_one"]] = float("-inf")
        lvl_two_logits[input_data["feasibility_mask_level_two"]] = float("-inf")
        lvl_three_logits[input_data["feasibility_mask_level_three"]] = float("-inf")

        loss_zero = criterion(lvl_zero_logits, target_zero) 
        loss_one = criterion(lvl_one_logits, target_one)
        loss_two = criterion(lvl_two_logits, target_two)
        loss_three = criterion(lvl_three_logits, target_three)
        loss_per_batch = loss_zero + loss_one + loss_two + loss_three
        policy_loss = (weights * loss_per_batch).mean()

        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(policy_loss).backward()
        scaler.unscale_(optimizer)

        if gen_config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=gen_config.optimizer["gradient_clipping"])

        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        batch_loss = policy_loss.item()
        accumulated_loss += policy_loss.item()

        accumulated_loss_lvl_zero += loss_zero[mask_zero].sum().item()
        accumulated_loss_lvl_one += loss_one[mask_one].sum().item()
        accumulated_loss_lvl_two += loss_two[mask_two].sum().item()
        accumulated_loss_lvl_three += loss_three[mask_three].sum().item()

        count_lvl_zero += mask_zero.sum().item()
        count_lvl_one += mask_one.sum().item()
        count_lvl_two += mask_two.sum().item()
        count_lvl_three += mask_three.sum().item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data 

    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / max(count_lvl_zero, 1)
    metrics["loss_level_one"] = accumulated_loss_lvl_one / max(count_lvl_one, 1)
    metrics["loss_level_two"] = accumulated_loss_lvl_two / max(count_lvl_two, 1)
    metrics["loss_level_three"] = accumulated_loss_lvl_three / max(count_lvl_three, 1)
    metrics["total_loss"] = accumulated_loss/ num_batches

    print(f"Epoch {epoch+1}/{gen_config.num_epochs} |" f"Level 0={accumulated_loss_lvl_zero /  max(count_lvl_zero, 1):.4f} |" f"Level 1={accumulated_loss_lvl_one /  max(count_lvl_one, 1):.4f} |" 
          f"Level 2={accumulated_loss_lvl_two /  max(count_lvl_two, 1):.4f} |" f"Level 3 ={accumulated_loss_lvl_three / max(count_lvl_three, 1):.4f} |" 
          f"Total loss this round ={accumulated_loss/ num_batches:.4f}")

    return metrics

def test(eval_type: str, gen_config, env_config, model_weights, test_instances: dict, destination_path):

    gen_config.gumbeldore_config["search_type"] == "beam_search" #Set the beam search type to deterministic beam search during testing
    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config, train_instances=test_instances)
    gumbeldore_dataset.generate_dataset(network_weights=model_weights, memory_aggressive=False, 
                                                     if_test = True, epoch=0)
    
def evaluate(epoch, eval_type: str, gen_config, env_config, model_weights, test_instances: dict, destination_path):

    gen_config.gumbeldore_config["search_type"] == "beam_search" #set the beam search type to deterministic beam search during testing
    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config, train_instances=test_instances)
    metrics_return, destination_full_path = gumbeldore_dataset.generate_dataset(network_weights=model_weights, memory_aggressive=False, 
                                                     if_eval = True, epoch=epoch)
    
    return metrics_return


def main(args):
    print(">> Flowsheet Design using Gumbeldore")

    gen_config = GeneralConfig(args)
    env_config = EnvConfig(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = gen_config.CUDA_VISIBLE_DEVICES
    num_gpus = len(gen_config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, log_to_driver=False, logging_level="info", num_cpus=16) 
    print(ray.available_resources())

    logger = Logger(args, gen_config.results_path, gen_config.log_to_file)
    logger.log_hyperparams(gen_config)

    # set up mlflow connection 
    '''set_mlflow_connection() 
    model_start_time = f'{args.subsystem}' + '_' + 'generic'+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if gen_config.mlflow_experiment is None:
        mlflow.set_experiment(f'{args.subsystem}' + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))                        
    else:
        mlflow.set_experiment(gen_config.mlflow_experiment)'''

    # Fix random number generator seed for better reproducibility
    set_seed(gen_config.seed)

    # Setup the neural network for training
    network = FlowsheetNetwork(gen_config, env_config, gen_config.training_device)

    # Load checkpoint if needed
    if gen_config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {gen_config.load_checkpoint_from_path}")
        checkpoint = torch.load(gen_config.load_checkpoint_from_path, weights_only=False)
        print(f"{checkpoint['pretrain_epochs']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "best_optimizer_state": None,
            "pretrain_epochs": None,
            "epochs_trained": 0,
            "validation_metric": float("-inf"),   # objective of the best sequence designed during validation.
            "best_validation_metric": float("-inf")  # corresponding to best model weights
        }

    # Start always with the best model checkpoint 
    best_model_weights =  copy.deepcopy(checkpoint.get("model_weights")) 
    best_optimizer_state = copy.deepcopy(checkpoint.get("optimizer_state"))
    best_validation_metric = float("-inf")
    
    print(f"Policy network is on device {gen_config.training_device}")
    network.to(network.device)
    network.eval()
    
    # generate train and test instances for all four sub-systems 
    train_instances, validation_instances, test_instances = generate_test_sets(gen_config.num_epochs, gen_config, env_config, gen_config.results_path)
    env_config.gamma_mean, env_config.gamma_std = compute_gamma_normalization(train_instances)
    batches = batch_instances_for_dataset_eval(gen_config, train_instances)
    eval_batches = batch_instances_for_dataset_eval(gen_config, validation_instances)
    test_batches = batch_instances_for_dataset(gen_config, test_instances)
    print(f"Number of train_instances: {len(train_instances)}, Number of val_instances: {len(validation_instances)}, Number of test_instances: {len(test_instances)}")
    print(f"Number of train_batches: {len(batches)}, Number of val_batches: {len(eval_batches)}, Number of test_batches: {len(test_batches)}")


    if gen_config.num_epochs > 0:

        # Training loop
        print(f"Starting training for {gen_config.num_epochs} epochs.")

        current_model_weights = best_model_weights
        current_optimizer_state = best_optimizer_state

        # Continue training from the latest model.
        if current_model_weights is not None:
            network.load_state_dict(current_model_weights)

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=gen_config.optimizer["lr"],
            weight_decay=gen_config.optimizer["weight_decay"]
        )

        # Restore the optimizer corresponding to the loaded best model.
        if current_optimizer_state is not None:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(copy.deepcopy(current_optimizer_state))

        print("Setting up LR scheduler")
        _lambda = lambda epoch: gen_config.optimizer["schedule"]["decay_factor"] ** (
                    checkpoint["epochs_trained"] // gen_config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, threshold=1e-3, min_lr=1e-6)

        start_time_counter = None
        if gen_config.wall_clock_limit is not None:
            print(f"Wall clock limit of training set to {gen_config.wall_clock_limit / 3600} hours")
            start_time_counter = time.perf_counter()

        #with mlflow.start_run(run_name = model_start_time):
        for epoch in range(gen_config.num_epochs):

            gen_config.gumbeldore_config["search_type"] == args.beam_search_type #set the beam search type to either TASAR or wor during training
            print("------")
            print(f"Generating dataset.")

            # model used for sampling BEFORE doing updates
            sampling_model_weights = copy.deepcopy(network.get_weights())
            sampling_optimizer_state = copy.deepcopy(
                dict_to_cpu(optimizer.state_dict())
            )

            #mlflow.log_params({k: v for k, v in vars(gen_config).items() if isinstance(v, (int, float, str, bool))})

            generated_loggable_dict = train_for_one_epoch(
                epoch=epoch, gen_config=gen_config, env_config=env_config, network=network, network_weights=sampling_model_weights, optimizer=optimizer, 
                best_objective=best_validation_metric, train_instances=batches[epoch], destination_path=gen_config.gumbeldore_config['destination_path']
            )

            # Save model
            current_model_weights = copy.deepcopy(network.get_weights())
            current_optimizer_state = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))

            checkpoint["model_weights"] = current_model_weights
            checkpoint["optimizer_state"] = current_optimizer_state
            checkpoint["epochs_trained"] += 1
            save_checkpoint(checkpoint, "last_model.pt", gen_config)

            # log metrics per epoch 
            '''for key, val in generated_loggable_dict.items():
                mlflow.log_metric(key, val, step=epoch)'''

            if (epoch) % gen_config.evaluation_epoch == 0:
                total_val_mean = []
                print(">> Running validation.")
                for num, eval_batch in enumerate(eval_batches): 
                    with torch.no_grad():
                        print(f"Processing batch {num}/{len(eval_batches)}")
                        metrics = evaluate(epoch=epoch, eval_type='eval', gen_config=gen_config, env_config=env_config, model_weights=current_model_weights, test_instances=eval_batch, destination_path=gen_config.results_path) 
                        total_val_mean.append(metrics["worst_gen_obj"])  

                val_metric = np.mean(total_val_mean)     
                if val_metric >= best_validation_metric:
                    print(">> Got new best model.")
                    print(f"Previous validation best mean score {best_validation_metric}, New validation best mean score {val_metric},")
                    best_model_weights = copy.deepcopy(sampling_model_weights)
                    best_optimizer_state = copy.deepcopy(sampling_optimizer_state)
                    best_validation_metric = val_metric

                    checkpoint["best_model_weights"] = best_model_weights
                    checkpoint["best_validation_metric"] = best_validation_metric
                    checkpoint["best_optimizer_state"] = copy.deepcopy(best_optimizer_state)

                    save_checkpoint(checkpoint, "best_model.pt", gen_config)
                else:
                    print(f"No improvmement. Previous validation best mean score {best_validation_metric}, New validation mean score {val_metric},")

            if start_time_counter is not None and time.perf_counter() - start_time_counter > gen_config.wall_clock_limit:
                print("Time exceeded. Stopping training.")
                break

    if gen_config.num_epochs == 0:
        print(f"Testing with loaded model.")
        if (checkpoint.get("model_weights") is None and checkpoint.get("best_model_weights") is None):
            print("WARNING! No training was performed, but also no checkpoint to load was given. "
              "Evaluating with random model.")
        
        if best_model_weights is not None:
            test_model_weights = copy.deepcopy(best_model_weights)
        elif checkpoint.get("model_weights") is not None:
            test_model_weights = copy.deepcopy(checkpoint["model_weights"])
        else:
            test_model_weights = copy.deepcopy(network.get_weights())
            
    else:
        print(f'Testing with best model trained for {checkpoint["epochs_trained"]} epoches.')
        if best_model_weights is not None:
            #test_model_weights = checkpoint["best_model_weights"]
            test_model_weights = current_model_weights
        else:
            # Defensive fallback if no finite best metric was produced.
            test_model_weights = current_model_weights
            
    
    network.load_state_dict(test_model_weights)
    torch.cuda.empty_cache()
    print(">> TEST")

    for num, test_bat in enumerate(test_batches): 
        with torch.no_grad():
            print(f"Processing batch {num}/{len(test_batches)}")
            test('test', gen_config, env_config, model_weights=test_model_weights, test_instances=test_bat, destination_path=gen_config.results_path)
            
    print("Finished. Shutting down ray.") 
    ray.shutdown()

if __name__=='__main__':
    args = parse_args()
    main(args)
