from skopt.space import Integer, Real
from pstree.datasets.synthetic_datasets import (
    load_synthetic1, load_synthetic2, load_synthetic3, load_synthetic4, load_synthetic5, load_synthetic6, 
    load_synthetic7, load_synthetic8, load_synthetic9, load_synthetic10, load_synthetic11, load_synthetic12,
)
from pstree.datasets.data_loader import ( 
    load_airfoil, load_boston, 
    load_concrete_strength, 
    load_diabetes, load_efficiency_heating, load_forest_fires,
    load_istanbul, load_ld50, load_bioav, load_parkinson_updrs, load_ppb, load_resid_build_sale_price,
)

from pstree.cluster_gp_sklearn import GPRegressor
from sklearn.tree import DecisionTreeRegressor

# --------------------------- # 
#    General Configuration    #
# --------------------------- #

datasets = {name.split('load_')[1] : loader for name, loader in globals().items() if name.startswith('load_') and callable(loader)}

N_SPLITS = 4                
N_CV = 4                     # 4      

N_SEARCHES_HYPER = 20      
N_RANDOM_STARTS = 10       

NOISE_SKOPT = 1e-3
N_TESTS = 20              
P_TEST = 0.25 
SEED = 0
N_TIME_BINS = 300
SUFFIX_SAVE = '1'
PREFIX_SAVE = 'PS3'  
EXPERIMENT_NAME = 'PS3'
TEST_DIR = 'test'
TUNE_DIR = 'train'
MLFLOW_TRACKING_URI = 'file:../data/mlruns' 

DATA_DIR = 'data'
REPO_URL = 'git@github.com:amarmate/data_transfer.git'
AUTO_COMMIT_INTERVAL = 0.25 * 3600 # every 15 min  


# ------------------------------- # 
#  PS3 Experiment Configuration   # 
# ------------------------------- # 

BASIC_PRIMITIVE         = True
CONSTANT_RANGE          = 2
N_POP                   = 25
N_GEN                   = 500
ADAPTIVE_TREE           = True
NORMALIZE               = False
FINAL_PRUNE             = True
VERBOSE                 = False  # False
SIZE_OBJECTIVE          = True
SOFT_TREE               = False
REGR_CLASS              = GPRegressor
TREE_CLASS              = DecisionTreeRegressor



SPACE_PS3 = [
    Integer(2, 8, name='max_leaf_nodes'),
    Integer(3, 8, name='height_limit'),     
]

ps3_params = {
    "regr_class": REGR_CLASS,
    "tree_class": TREE_CLASS,
    "n_pop": N_POP,
    "n_gen": N_GEN,
    "basic_primitive": BASIC_PRIMITIVE,
    "size_objective": SIZE_OBJECTIVE,
    "constant_range": CONSTANT_RANGE,
    "normalize": NORMALIZE,
    "verbose": VERBOSE,
    "adaptive_tree": ADAPTIVE_TREE,
    "final_prune": FINAL_PRUNE,
    "soft_tree": SOFT_TREE,
}


# --------------------------- #
#    Save Configuration       #
# --------------------------- #



config = {
    'N_SPLITS': N_SPLITS,
    'N_CV': N_CV,

    'N_SEARCHES_HYPER': N_SEARCHES_HYPER,
    'N_RANDOM_STARTS': N_RANDOM_STARTS,

    'NOISE_SKOPT': NOISE_SKOPT,
    'N_TESTS': N_TESTS,
    'P_TEST': P_TEST,
    'SEED': SEED,
    'N_TIME_BINS': N_TIME_BINS,
    'SUFFIX_SAVE': SUFFIX_SAVE,
    'PREFIX_SAVE': PREFIX_SAVE,
    'EXPERIMENT_NAME': EXPERIMENT_NAME,
    'AUTO_COMMIT_INTERVAL': AUTO_COMMIT_INTERVAL,
    'PI' : None,
    'multi_run': False,
    'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI,

    'DATA_DIR': DATA_DIR,
    'TEST_DIR': TEST_DIR,
    'TUNE_DIR': TUNE_DIR,
    'REPO_URL': REPO_URL,
        
    'datasets' : datasets,
    'SELECTORS': ['no'],

    'SPACE_PARAMETERS': SPACE_PS3, 
    'gen_params' : ps3_params,
}