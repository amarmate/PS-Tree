from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

if __name__ == "__main__":
    X,y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    r = PSTreeRegressor(regr_class=GPRegressor, 
                        tree_class=DecisionTreeRegressor,
                        height_limit=6, 
                        n_pop=25, 
                        n_gen=50,
                        basic_primitive=True,
                        size_objective=True,
                        max_leaf_nodes=4,
                        constant_range=2,
                        random_seed=0, 
                        random_state=0,
                        normalize=True,
    )

    r.fit(X_train, y_train)
    print(r2_score(y_test, r.predict(X_test)))
    print(r.model())




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
