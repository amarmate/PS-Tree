from sklearn.model_selection import train_test_split

from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor
from pstree.datasets.data_loader import *
from pstree.datasets.synthetic_datasets import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, root_mean_squared_error as rmse

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # X,y = load_diabetes()
    X, y, _, _ = load_synthetic12()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    r = PSTreeRegressor(regr_class=GPRegressor, 
                        tree_class=DecisionTreeRegressor,
                        height_limit=6,  # 6 
                        n_pop=25,  # 25
                        n_gen=5,  # 50 
                        basic_primitive=True,
                        size_objective=True,
                        max_leaf_nodes=4,  # 4 
                        constant_range=2,  # 2 
                        random_seed=0, 
                        # random_state=0,
                        normalize=False,
                        verbose=True,
                        adaptive_tree=True,
                        test_data=(X_test, y_test),
                        final_prune=True,
                        soft_tree=False,
    )

    r.fit(X_train, y_train)
    print(r2_score(y_test, r.predict(X_test)))
    print(rmse(y_test, r.predict(X_test)))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, r.predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('PSTreeRegressor Predictions vs True Values')
    plt.savefig('pstree_regressor_predictions.png')
    
    print(type(r.regr.log_book))
    print(r.labels)
        




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
