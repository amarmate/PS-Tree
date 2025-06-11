from tests.experiments.PSTREE.run_ps3 import run_ps3
from tests.experiments.PSTREE.config_ps3 import config

from tests.experiments.parse import parse_args
from tests.experiments.github import init_or_update_repo
from tests.experiments.mlflow import cleanup_running_runs

if __name__ == "__main__":
    try: 
        args = parse_args(config)

        # init_or_update_repo(config)

        cleanup_running_runs()
        
        run_ps3(args)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
