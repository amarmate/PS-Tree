from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor
from pstree.datasets.data_loader import *
from pstree.datasets.synthetic_datasets import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (r2_score, 
                             root_mean_squared_error as rmse)
from tests.misc_functions import get_classification_summary
from tests.metrics_test import (train_test_split, 
                                calc_scores_from_summary as calc_scores)

from matplotlib import pyplot as plt
import pickle

if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(0)
    X = np.linspace(0, 10, 200).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(scale=0.2, size=X.shape[0])

    r = PSTreeRegressor(regr_class=GPRegressor, 
                        tree_class=DecisionTreeRegressor,
                        height_limit=6,  # 6 
                        n_pop=25,  # 25
                        n_gen=200,  # 50 
                        basic_primitive=True,
                        size_objective=True,
                        min_samples_leaf=10, 
                        max_leaf_nodes=5,  # 4 
                        constant_range=2,  # 2 
                        max_depth=2,
                        random_seed=0, 
                        # random_state=0,
                        normalize=False,
                        verbose=True,
                        adaptive_tree=True,
                        final_prune=True,
                        soft_tree=False,
    )

    r.fit(X, y)
    y_pred = r.predict(X)
    with open('pickle_preds_PS3.pkl', 'wb') as f:
        pickle.dump(y_pred, f)
    

    
    
    
# | **Parameter**            | **Value**                          | **Description**                             |
# | ------------------------ | ---------------------------------- | ------------------------------------------- |
# | Evaluation Metric        | R²                                 | Used across all experiments                 |
# | Max Number of Partitions | 4                                  | To keep the model explainable               | DONE 
# | Population Size          | 25                                 | Evolutionary strategy setting               | DONE 
# | Max Generations          | 50                                 | Evolutionary strategy setting               | DONE 
# | Max Feature Tree Height  | 6                                  | Controls model complexity                   | DONE 
# | Random Constant Range    | \[−2, 2]                           | Bounds for constant values                  | DONE 
# | Height of New Subtrees   | \[1, 3]                            | Mutation depth constraint                   | 
# | Crossover Probability    | 0.9                                | Evolution operator                          | DONE 
# | Mutation Probability     | 0.1                                | Evolution operator                          | DONE 
# | Primitive Functions      | {+, −, ∗, AQ}                      | Limited to interpretable functions          | DONE 
# | Train/Test Split         | 80% training / 20% testing         | To assess generalization                    | 
# | Data Standardization     | Standardized (mean=0, var=1)       | Preprocessing before training               | 
# | Ablation Studies         | 50 independent runs                | Medians used for statistical comparison     |
# | Statistical Comparison   | Wilcoxon test (95% significance)   | Symbols: + (better), ∼ (similar), - (worse) |
