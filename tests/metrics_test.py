import numpy as np
import math 
from sklearn.metrics import precision_recall_fscore_support, f1_score


def mape(y_true, y_pred):
    """"Mean Absolute Percentage Error."""
    return np.mean(np.array((np.abs((y_true - y_pred) / y_true)))) * 100

def nrmse(y_true, y_pred):
    """Normalized RMSE."""
    range_y = y_true.max() - y_true.min()
    return (np.sqrt(np.mean(np.array((y_true - y_pred) ** 2))) / range_y)

def r_squared(y_true, y_pred):
    """R-squared."""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array((y_true - y_pred))))

def standardized_rmse(y_true, y_pred):
    """Standardized RMSE."""
    std_y = np.std(np.array(y_true))
    return np.sqrt(np.mean(np.array((y_true - y_pred) ** 2))) / std_y

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean(np.array((y_true - y_pred) ** 2)))

def calc_scores_from_summary(class_summary):
    """
    class_summary: np.array, shape=(n_predicted_classes, n_true_classes)
    Gibt accuracy, macro_f1 und weighted_f1 zurück.
    """
    
    # 1) Pad matrix auf quadratisch (n_true_classes × n_true_classes)
    n_true = class_summary.shape[1]
    n_pred = class_summary.shape[0]
    if n_pred < n_true:
        padding = np.zeros((n_true - n_pred, n_true), dtype=int)
        cm = np.vstack([class_summary, padding])
    else:
        cm = class_summary[:, :n_true]

    # 2) Rekonstruiere y_true, y_pred
    y_true, y_pred = [], []
    for true_label in range(n_true):
        for pred_label in range(n_true):
            cnt = cm[pred_label, true_label]
            y_true.extend([true_label] * cnt)
            y_pred.extend([pred_label] * cnt)

    # 3) Berechne Scores
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    macro_f1    = f1_score(y_true, y_pred, average='macro',    labels=list(range(n_true)), zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=list(range(n_true)), zero_division=0)

    return accuracy, macro_f1, weighted_f1



# ------------------------------------------------------------

def train_test_split(X, y, p_test=0.3, shuffle=True, indices_only=False, seed=0):
    """Splits X and y arrays into train and test subsets.

    This method replicates the behaviour of Sklearn's 'train_test_split'.

    Parameters
    ----------
    X : np.ndarray
        Input data instances.
    y : np.ndarray
        Target vector.
    p_test : float (default=0.3)
        The proportion of the dataset to include in the test split.
    shuffle : bool (default=True)
        Whether to shuffle the data before splitting.
    indices_only : bool (default=False)
        Whether to return only the indices representing training and test partition.
    seed : int (default=0)
        The seed for random number generators.

    Returns
    -------
    X_train : np.ndarray
        Training data instances.
    X_test : np.ndarray
        Test data instances.
    y_train : np.ndarray
        Training target vector.
    y_test : np.ndarray
        Test target vector.
    train_indices : np.ndarray
        Indices representing the training partition.
    test_indices : np.ndarray
        Indices representing the test partition.
    """
    np.random.seed(seed)
    
    if shuffle:
        indices = np.random.permutation(X.shape[0])
    else:
        indices = np.arange(0, X.shape[0], 1)
    
    split = int(math.floor(p_test * X.shape[0]))
    
    train_indices, test_indices = indices[split:], indices[:split]

    if indices_only:
        return train_indices, test_indices
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test

