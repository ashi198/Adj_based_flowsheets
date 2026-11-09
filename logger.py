import os
import inspect
import json, pickle
from typing import Optional
import numpy as np


class Logger:
    def __init__(self, args, results_path: str, log_to_file: bool):
        self.results_path = results_path
        self.log_to_file = log_to_file

        self.file_log_path = os.path.join(self.results_path, "log.txt")
        if self.log_to_file:
            os.makedirs(self.results_path, exist_ok=True)

    def log_hyperparams(self, config_object):
        attributes = inspect.getmembers(config_object, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attribute_dict = {}

        def add_to_attribute_dict(a):
            for key, value in a:
                key = key.replace("+", "_plus")
                key = key.replace("@", "_at")
                if isinstance(value, dict):
                    add_to_attribute_dict([(f"{key}.{k}", v) for k, v in value.items()])
                else:
                    if key not in ["devices_for_eval_workers"] and len(str(value)) <= 500:
                        attribute_dict[key] = value

        add_to_attribute_dict(attributes)

        if self.log_to_file:
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps({"hyperparameters": attribute_dict}))
                f.write("\n")

    def log_metrics(self, metrics: dict, step: Optional[int] = None, step_desc: Optional[str] = "epoch"):
        if self.log_to_file:
            if step is not None:
                metrics[step_desc] = step
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics, default=str))
                f.write("\n")