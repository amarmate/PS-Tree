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


import sympy
from sympy.functions.elementary.piecewise import ExprCondPair

def parse_piecewise(expr: str):
    namespace = {
        "Piecewise": sympy.Piecewise,
        "ExprCondPair": ExprCondPair,
        "Add": sympy.Add,
        "And": sympy.And,
        "Or": sympy.Or,
        "Not": sympy.Not,
        "Mul": sympy.Mul,
        "Pow": sympy.Pow,
        "Float": sympy.Float,
        "Rational": sympy.Rational,
        "Integer": sympy.Integer,
        "Symbol": sympy.Symbol,
        "LessThan": sympy.LessThan,
        "true": True  # SymPy uses `true` in srepr sometimes
    }
    expr = eval(expr, namespace)
    if not isinstance(expr, sympy.Piecewise):
        raise ValueError("Input is not a Piecewise expression")
    result = []
    for expr_, cond in expr.args:
        cond_str = str(cond) if cond is not True else "otherwise"
        result.append((str(expr_), cond_str))
    return result


if __name__ == "__main__":
    X,y = load_airfoil()
    # X, y, mask, mask = load_synthetic12()

    train, test = train_test_split(X, y, p_test=0.2, seed=0, indices_only=True)
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    # mask_train = [m[train] for m in mask] if mask is not None else None

    r = PSTreeRegressor(regr_class=GPRegressor, 
                        tree_class=DecisionTreeRegressor,
                        height_limit=6,  # 6 
                        n_pop=5,  # 25
                        n_gen=200,  # 50 
                        basic_primitive=True,
                        size_objective=True,
                        min_samples_leaf=10, 
                        max_leaf_nodes=2,  # 4 
                        constant_range=2,  # 2 
                        max_depth=2,
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
    
    # class_summary = get_classification_summary(
    #     X_data    = X_train,
    #     mask      = mask_train,
    #     spec_masks= r.labels
    # )
    # print("Confusion summary:\n", class_summary)

    # acc, macro_f1, weighted_f1 = calc_scores(class_summary)
    # print(f"Accuracy: {acc:.3f}")
    # print(f"Macro‑F1: {macro_f1:.3f}")
    # print(f"Weighted‑F1: {weighted_f1:.3f}")
    
    raw = f"{r.model()}"
    print(raw)
    parsed = parse_piecewise(raw)
    for expr, cond in parsed:
        print(f"If {cond}, then {expr}")
    



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
