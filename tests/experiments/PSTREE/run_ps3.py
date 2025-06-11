import numpy as np 
import mlflow 
from tests.experiments.PSTREE.config_ps3 import *
from tests.experiments.tracking import get_tasks

from tests.experiments.tunner import Tuner
from tests.experiments.PSTREE.tune_ps3 import gp_tune

from tests.experiments.tester import Tester 
from tests.experiments.PSTREE.tune_ps3 import gp_test
from tests.experiments.github import periodic_commit

from joblib import Parallel, delayed, parallel_config
import threading


def run_experiment(config, task):
    name, selector, split_id = task['gen_params']['dataset_name'], task['gen_params']['selector'], task['split_id']
    print(f"Running task: {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}")
    mlflow.set_experiment(f"{EXPERIMENT_NAME}_{name}_{selector}_split{split_id}")

    tuner = Tuner(config=config, 
                  objective_fn = gp_tune,
                  **task)
    params = tuner.tune()    

    print(f'Tunning completed for {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}')
    print(f"Running testing for task: {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}")

    tester = Tester(config=config, 
                    test_fn=gp_test,
                    best_params=params, 
                    **task)
    tester.run()

    print(f'Testing completed for {EXPERIMENT_NAME} / {name} / selector {selector} / split {split_id}')



def run_gp(args):
    np.random.seed(SEED)

    mlflow.set_tracking_uri("file:../data/mlruns")

    tasks = get_tasks(args, config)

    # commit_thread = threading.Thread(
    #     target=periodic_commit, args=(config,), daemon=True
    # )
    # commit_thread.start()
    
    with parallel_config(n_jobs=args.workers, prefer='processes', verbose=10):
        Parallel()(delayed(run_experiment)(config, task) for task in tasks)
