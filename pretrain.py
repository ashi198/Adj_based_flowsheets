import argparse, copy, os, ray, torch

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime, pickle

from logger import Logger
from flowsheet_dataset import RandomDataset

import numpy as np
from config import GeneralConfig, EnvConfig
from model.policy_arch import FlowsheetNetwork, dict_to_cpu
from utils import set_mlflow_connection, build_logit_tensors_per_level, set_seed, compute_gamma_normalization, generate_test_sets


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
                        default=200,
                        help="num of epoches to train")
    
    parser.add_argument("--device",
                        type=str,
                        default='cuda:1',
                        help="device")
    
    parser.add_argument("--beam_search",
                        type=int,
                        default=64,
                        help="64 or 128")
    
    parser.add_argument("--beam_search_type",
                        type=str,
                        default="tasar",
                        help="wor or tasar")
    
    parser.add_argument("--results",
                        type=str,
                        help="specify directory for storing results", 
                        default="./pretrain_with_reward_conditioning/")
    
    parser.add_argument("--train_pickle",
                        type=str,
                        help="train_pickle file for checking training", 
                        default="./train_actions_new.pickle")
    
    parser.add_argument("--val_pickle",
                        type=str,
                        help="train_pickle file for checking training", 
                        default="./val_actions_new.pickle")

    parser.add_argument("--if_pretrain",
                        type=bool,
                        help="True or False", 
                        default=True)
        
    parser.add_argument("--num_instances_per_batch",
                        type=int,
                        help="specify directory for storing results", 
                        default=4)
    
    
    args = parser.parse_args()
    return args
    

def save_checkpoint(checkpoint: dict, filename: str, gen_config):
    os.makedirs(gen_config.results_path, exist_ok=True)
    path = os.path.join(gen_config.results_path, filename)
    torch.save(checkpoint, path)


def train_for_one_epoch(epoch: int, gen_config, env_config, network: FlowsheetNetwork, network_weights: dict,
                        optimizer: torch.optim.Optimizer, train_dataset= None, is_validation=False):

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0,
                            pin_memory=False, persistent_workers=False)
    
    metrics = dict()

    # Train for one epoch
    network.train()  

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
    criterion = CrossEntropyLoss(reduction="none", ignore_index=-1)

    for _ in progress_bar:
        data = next(data_iter)
        indices_for_tracking = data.pop('indices_for_tracking')
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        
        # targets for the logit levels
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)
        target_three = data["target_three"][0].to(network.device)

        terminate_or_open_streams_logits, unit_predictions = network(input_data)
        
        #build logits based on selected units and streams 
        lvl_zero_logits, lvl_one_logits, lvl_two_logits, lvl_three_logits = build_logit_tensors_per_level(dataset= train_dataset, 
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
        loss = loss_per_batch.mean()

        mask_zero = target_zero != -1
        mask_one = target_one != -1
        mask_two = target_two != -1
        mask_three = target_three != -1

        # Optimization step
        if not is_validation:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if gen_config.optimizer["gradient_clipping"] > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=gen_config.optimizer["gradient_clipping"])

            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

        batch_loss = loss.item()
        accumulated_loss += loss.item()
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
    
    metric_prefix = "" if not is_validation else "val_"

    metrics[f"{metric_prefix}loss_level_zero"] = accumulated_loss_lvl_zero /  max(count_lvl_zero, 1)
    metrics[f"{metric_prefix}loss_level_one"] = accumulated_loss_lvl_one /  max(count_lvl_one, 1)
    metrics[f"{metric_prefix}loss_level_two"] = accumulated_loss_lvl_two /  max(count_lvl_two, 1)
    metrics[f"{metric_prefix}loss_level_three"] = accumulated_loss_lvl_three /  max(count_lvl_three, 1)
    metrics[f"{metric_prefix}total_loss"] = accumulated_loss/ num_batches

    if not is_validation:
        print(f"Epoch {epoch+1}/{gen_config.num_epochs} |" f"Level 0={accumulated_loss_lvl_zero / max(count_lvl_zero, 1):.4f} |" f"Level 1={accumulated_loss_lvl_one / max(count_lvl_one, 1):.4f} |" 
            f"Level 2={accumulated_loss_lvl_two / max(count_lvl_two, 1):.4f} |" f"Level 3 ={accumulated_loss_lvl_three /  max(count_lvl_three, 1):.4f} |" 
            f"Total loss this round ={accumulated_loss/ num_batches:.4f}")
    else:
        print(f"Validation |" f"Level 0={accumulated_loss_lvl_zero / max(count_lvl_zero, 1):.4f} |" f"Level 1={accumulated_loss_lvl_one / max(count_lvl_one, 1):.4f} |" 
            f"Level 2={accumulated_loss_lvl_two / max(count_lvl_two, 1):.4f} |" f"Level 3 ={accumulated_loss_lvl_three / max(count_lvl_three, 1):.4f} |" 
            f"Total loss this round ={accumulated_loss/ num_batches:.4f}")


    return metrics

def calculate_gamma_normalization(gen_config, env_config):
    train_instances, validation_instances, test_instances = generate_test_sets(gen_config.num_epochs, gen_config, env_config, gen_config.results_path)
    env_config.gamma_mean, env_config.gamma_std = compute_gamma_normalization(train_instances)


def main(args):
    print(">> Flowsheet Design using Gumbeldore")

    gen_config = GeneralConfig(args)
    env_config = EnvConfig(args)
    gen_config.if_pretrain = args.if_pretrain

    os.environ["CUDA_VISIBLE_DEVICES"] = gen_config.CUDA_VISIBLE_DEVICES
    num_gpus = len(gen_config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, log_to_driver=False, logging_level="info", num_cpus=16) 
    print(ray.available_resources())

    logger = Logger(args, gen_config.results_path, gen_config.log_to_file)
    logger.log_hyperparams(gen_config)

    # set up mlflow connection 
    '''set_mlflow_connection() 
    model_start_time = f'{args.subsystem}' + '_' + 'pretrain_check_'+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            "optimizer_state": None,
            "pretrain_epochs": 0,
            "epochs_trained": 0,
            "best_validation_loss": float("inf")  # corresponding to best model weights
        }

    # Start always with the best model checkpoint 
    best_model_weights = copy.deepcopy(checkpoint.get("model_weights")) 
    best_optimizer_state = copy.deepcopy(checkpoint.get("optimizer_state"))

    print(f"Policy network is on device {gen_config.training_device}")
    network.to(network.device)
    network.eval()

    calculate_gamma_normalization(gen_config, env_config)

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
        
        print("---- Loading dataset")
        train_dataset = RandomDataset(0, gen_config=gen_config, env_config=env_config, path_to_pickle=args.train_pickle, batch_size=gen_config.batch_size_training,
                                    custom_num_batches=gen_config.num_batches_per_epoch, no_random=False)
        
        val_dataset = RandomDataset(0, gen_config=gen_config, env_config=env_config, path_to_pickle=args.val_pickle, batch_size=gen_config.batch_size_training,
                                    custom_num_batches=gen_config.num_batches_per_epoch, no_random=True)

        patience = gen_config.optimizer.get("early_stopping_patience", 25)
        min_delta = gen_config.optimizer.get("early_stopping_min_delta", 1e-2)
        num_bad_epochs = 0

        #with mlflow.start_run(run_name = model_start_time):
        for epoch in range(gen_config.num_epochs):
            print("------")
            print(f"Training on fixed dataset.")

            # model used for sampling BEFORE doing updates
            sampling_model_weights = copy.deepcopy(network.get_weights())

            #mlflow.log_params({k: v for k, v in vars(gen_config).items() if isinstance(v, (int, float, str, bool))})

            generated_loggable_dict = train_for_one_epoch(
                epoch=epoch, gen_config=gen_config, env_config=env_config, network=network, network_weights=sampling_model_weights, optimizer=optimizer, 
                train_dataset=train_dataset)

            # Save model
            current_model_weights = copy.deepcopy(network.get_weights())
            current_optimizer_state = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))
            checkpoint["model_weights"] = current_model_weights
            checkpoint["optimizer_state"] = current_optimizer_state
            checkpoint["pretrain_epochs"] += 1
            save_checkpoint(checkpoint, "last_model.pt", gen_config)
            
            print("Validating...")
            torch.cuda.empty_cache()
            with torch.no_grad():
                validation_metrics = train_for_one_epoch(
                    epoch=None, gen_config=gen_config, env_config=env_config, network=network,network_weights= current_model_weights, optimizer=None, train_dataset=val_dataset, is_validation=True
                )
                scheduler.step(validation_metrics["val_total_loss"])

                # log metrics per epoch 
                '''for key, val in generated_loggable_dict.items():
                    mlflow.log_metric(key, val, step=epoch)

                for key, val in validation_metrics.items():
                    mlflow.log_metric(key, val, step=epoch)'''

                if validation_metrics["val_total_loss"] < checkpoint["best_validation_loss"]:
                    print(">> Got new best model.")
                    best_model_weights = copy.deepcopy(current_model_weights)
                    best_optimizer_state = copy.deepcopy(current_optimizer_state)
                    checkpoint["best_validation_loss"] = validation_metrics["val_total_loss"] 
                    num_bad_epochs = 0
                    checkpoint["best_model_weights"] = best_model_weights
                    checkpoint["best_optimizer_state"] = copy.deepcopy(best_optimizer_state)
                    save_checkpoint(checkpoint, "best_model.pt", gen_config)

                elif validation_metrics["val_total_loss"] <= checkpoint["best_validation_loss"] + min_delta:
                    print(f"Okayish improvement")
                    pass

                else:
                    num_bad_epochs += 1
                    print(f">> Validation. Avg loss: {validation_metrics['val_total_loss']}. Best validation loss: {checkpoint['best_validation_loss']}")
                    print(f"No improvement for {num_bad_epochs}/{patience} epochs.")

                # early stopping trigger
                if num_bad_epochs >= patience:
                    print("Early stopping triggered.")
                    break

            
    print("Finished. Shutting down ray.")
    ray.shutdown()

if __name__=='__main__':
    args = parse_args()
    main(args)
