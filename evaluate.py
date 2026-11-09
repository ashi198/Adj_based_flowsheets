import argparse, copy, os, time, ray, torch, mlflow

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime, pickle

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
                        default='cuda:0',
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
                        default="./FINAL_evaluation_again/test_with_complete_training_and_last_model")
    
    parser.add_argument("--load_checkpoint_from_path",
                        type=str,
                        help="specify directory for storing results", 
                        default="/home/akhanna/GumbelFlowsheets/FINAL_evaluation_again/all/last_model.pt")
    
    parser.add_argument("--load_test_pickle",
                        type=str,
                        help="specify directory for storing results", 
                        default="/home/akhanna/GumbelFlowsheets/FINAL_evaluation_again/all/test_instances.pickle")
    
    parser.add_argument("--load_train_pickle",
                        type=str,
                        help="specify directory for storing results", 
                        default="/home/akhanna/GumbelFlowsheets/FINAL_evaluation_again/all/train_instances.pickle")
    
    args = parser.parse_args()
    return args
    
def load_pickle_file (path):
    
    with open(path, "rb") as file: 
        loaded_data = pickle.load(file)
    
    return loaded_data


def save_checkpoint(checkpoint: dict, filename: str, gen_config):
    os.makedirs(gen_config.results_path, exist_ok=True)
    path = os.path.join(gen_config.results_path, filename)
    torch.save(checkpoint, path)

def test(eval_type: str, gen_config, env_config, model_weights, test_instances: dict, destination_path):

    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config, train_instances=test_instances)
    gumbeldore_dataset.generate_dataset(network_weights=model_weights, memory_aggressive=False, 
                                                     if_test = True, epoch=0)
    
def evaluate(epoch, eval_type: str, gen_config, env_config, model_weights, test_instances: dict, destination_path):

    gumbeldore_dataset = GumbeldoreDataset(gen_config, env_config, train_instances=test_instances)
    metrics_return, destination_full_path = gumbeldore_dataset.generate_dataset(network_weights=model_weights, memory_aggressive=False, 
                                                     if_eval = True, epoch=epoch)
    
    return metrics_return


def evaluate(args):
    print(">> Flowsheet Design using Gumbeldore")

    gen_config = GeneralConfig(args)
    env_config = EnvConfig(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = gen_config.CUDA_VISIBLE_DEVICES
    num_gpus = len(gen_config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, log_to_driver=False, logging_level="info", num_cpus=16) 
    print(ray.available_resources())

    logger = Logger(args, gen_config.results_path, gen_config.log_to_file)
    logger.log_hyperparams(gen_config)

    # Fix random number generator seed for better reproducibility
    set_seed(gen_config.seed)

    # Setup the neural network for training
    network = FlowsheetNetwork(gen_config, env_config, gen_config.training_device)

    # Load checkpoint if needed
    if args.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {args.load_checkpoint_from_path}")
        checkpoint = torch.load(args.load_checkpoint_from_path, weights_only=False)
        print(f"{checkpoint['epochs_trained']} episodes have been trained in the loaded checkpoint.")

    # Start always with the best model checkpoint 
    best_model_weights = copy.deepcopy(checkpoint.get("model_weights")) 
    
    print(f"Policy network is on device {gen_config.training_device}")
    network.to(network.device)
    network.eval()
    
    # generate train and test instances for all four sub-systems 
    train_instances = load_pickle_file(args.load_train_pickle)
    env_config.gamma_mean, env_config.gamma_std = compute_gamma_normalization(train_instances)
        
    if best_model_weights is not None:
        test_model_weights = copy.deepcopy(best_model_weights)
    elif checkpoint.get("model_weights") is not None:
        test_model_weights = copy.deepcopy(checkpoint["model_weights"])
            
    network.load_state_dict(test_model_weights)
    torch.cuda.empty_cache()
    test_instances = load_pickle_file(args.load_test_pickle)

    test_batches = batch_instances_for_dataset(gen_config, test_instances)
    print(">> TEST")

    for num, test_bat in enumerate(test_batches): 
        with torch.no_grad():
            print(f"Processing batch {num}/{len(test_batches)}")
            test('test', gen_config, env_config, model_weights=test_model_weights, test_instances=test_bat, destination_path=gen_config.results_path)
            
    print("Finished. Shutting down ray.") 
    ray.shutdown()

if __name__=='__main__':
    args = parse_args()
    evaluate(args)
