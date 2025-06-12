from tests.experiments.PSTREE.run_ps3 import run_ps3
from tests.experiments.PSTREE.config_ps3 import config

from tests.experiments.parse import parse_args
from tests.experiments.github import init_or_update_repo
from tests.experiments.mlflow import cleanup_running_runs

import os 
from pathlib import Path
os.environ['MLFLOW_TRACKING_URI'] = config['MLFLOW_TRACKING_URI']


if __name__ == "__main__":
    try: 
        data_dir = Path("..") / config['DATA_DIR']
        if not data_dir.exists():
            os.mkdir(data_dir)
        print(f"Created directory: {data_dir}")

        args = parse_args(config)

        # init_or_update_repo(config)

        cleanup_running_runs()
        
        run_ps3(args)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
