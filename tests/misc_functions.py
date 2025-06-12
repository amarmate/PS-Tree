import numpy as np 
from collections import defaultdict
from itertools import chain


# ------------------------------------------------ PF FUNCTIONS ---------------------------------------------------
def pf_rmse_comp_extended(points, index=False, error=0):
    """
    Identifies the Pareto front from a list of points.
    Each point is a tuple where the first element is RMSE, the second is complexity,
    and any subsequent elements are other metrics.
    Dominance is determined based on RMSE and complexity:
    A point A dominates point B if A's RMSE <= B's RMSE and A's complexity <= B's complexity,
    and at least one of these is strictly smaller.
    The function returns the full original points that are non-dominated.
    """
    err = 1 - error 
    
    pareto = []
    if not points:
        return pareto

    for i, point1 in enumerate(points):
        rmse1 = point1[0]
        comp1 = point1[1]
        dominated = False
        for j, point2 in enumerate(points):
            if i == j:
                continue 

            rmse2 = point2[0]
            comp2 = point2[1]
            if (rmse2 <= rmse1*err and comp2 <= comp1*err) and \
               (rmse2 < rmse1*err or comp2 < comp1*err):
                dominated = True
                break 

        if not dominated:
            pareto.append((i, point1)) if index else pareto.append(point1)
    if index: 
        pareto.sort(key=lambda x: (x[1][0], x[1][1])) 
    else: 
        pareto.sort(key=lambda x: (x[0], x[1]))
    return pareto

def pf_rmse_comp_time(points): 
    """
    Generate a Pareto front considering RMSE, complexity, and time.

    Parameters
    ----------
    points : list of tuples (rmse, comp, time)
        A list of individuals from the Pareto front. Each individual is represented as 
        (RMSE, complexity, time)

    Returns
    -------
    list
        A Pareto front containing the selected individuals based on the criteria.
    """

    pareto = []
    for i, (rmse1, comp1, time1) in enumerate(points):
        dominated = False
        for j, (rmse2, comp2, time2) in enumerate(points):
            if j != i and (rmse2 <= rmse1 and comp2 <= comp1 and time2 <= time1) and (rmse2 < rmse1 or comp2 < comp1 or time2 < time1):
                dominated = True
                break
        if not dominated:
            pareto.append((rmse1, comp1, time1))

    pareto.sort(key=lambda x: (x[0], x[1], x[2]))
    return pareto




# ------------------------------------------------ SPECIALIST MASKS ---------------------------------------------------
def get_specialist_masks(tree_node, X_data, current_mask=None, indices=True):

    """
    Function used to get the masks for each specialist to be used for specialist tunning

    Write the rest of the documentation 

    """
    def recursion(tree_node, X_data, current_mask=None):
        if current_mask is None:
            current_mask = np.ones(X_data.shape[0], dtype=bool)
        
        if isinstance(tree_node, tuple):  # Condition node
            condition = tree_node[0]
            left_child = tree_node[1]
            right_child = tree_node[2]
            
            # Evaluate condition
            cond_results = condition.predict(X_data) > 0
            true_mask = current_mask & cond_results
            false_mask = current_mask & ~cond_results
            
            # Process both branches
            left_masks = recursion(left_child, X_data, true_mask)
            right_masks = recursion(right_child, X_data, false_mask)
            
            # Merge results while preserving all masks
            merged = defaultdict(list)
            for sp, masks in chain(left_masks.items(), right_masks.items()):
                merged[sp].extend(masks)
                
            return merged
            
        else:
            return {tree_node: [current_mask]}
    
    result = recursion(tree_node, X_data)
    merged = defaultdict(list)

    for ind, mask in result.items(): 
        merged[ind] = np.sum(mask, axis=0).astype(bool)

    if not indices: 
        return merged  
    
    for ind, mask in merged.items():
        merged[ind] = np.where(mask)[0].tolist()
    return merged


# def get_classification_summary(X_data,
#                                mask,
#                                spec_masks=None,
#                                tree_node=None):
#     """
#     Liefert eine quadratische Kontingenzmatrix mit Shape (n_true_classes, n_true_classes).
#     Zeilen = vorhergesagte Klassen, Spalten = wahre Klassen.
#     """
#     assert (tree_node is not None) or (spec_masks is not None), \
#         "Entweder spec_masks oder tree_node muss gesetzt sein."

#     if spec_masks is None:
#         spec_dict = get_specialist_masks(tree_node, X_data, indices=True)
#     elif isinstance(spec_masks, dict):
#         spec_dict = spec_masks
#     else:
#         arr = np.array(spec_masks)
#         if arr.dtype == bool:
#             spec_dict = {i: np.where(arr[i])[0].tolist()
#                          for i in range(len(arr))}
#         else:
#             labels = np.unique(arr)
#             spec_dict = {lab: np.where(arr == lab)[0].tolist()
#                          for lab in labels}

#     id_masks = []
#     for m in mask:
#         m_arr = np.array(m)
#         if m_arr.dtype == bool:
#             id_masks.append(np.where(m_arr)[0].tolist())
#         else:
#             id_masks.append(m_arr.tolist())

#     class_summary = []
#     for _, pred_indices in spec_dict.items():
#         pred_set = set(pred_indices)
#         row = [len(pred_set.intersection(m_indices))
#                for m_indices in id_masks]
#         class_summary.append(row)
#     class_summary = np.array(class_summary, dtype=int)

#     n_true = class_summary.shape[1]
#     n_pred = class_summary.shape[0]
#     if n_pred < n_true:
#         padding = np.zeros((n_true - n_pred, n_true), dtype=int)
#         class_summary = np.vstack([class_summary, padding])
#     elif n_pred > n_true:
#         class_summary = class_summary[:n_true, :n_true]

#     return class_summary


def get_classification_summary(X_data,
                               mask,
                               spec_masks=None,
                               tree_node=None):
    """
    Liefert eine transponierte Kontingenzmatrix mit Shape (n_true_classes, n_pred_classes).
    Zeilen = wahre Klassen, Spalten = vorhergesagte Klassen.
    """
    assert (tree_node is not None) or (spec_masks is not None), \
        "Entweder spec_masks oder tree_node muss gesetzt sein."
                               
    if spec_masks is None:
        spec_dict = get_specialist_masks(tree_node, X_data, indices=True)
    elif isinstance(spec_masks, dict):
        spec_dict = spec_masks
    else:
        arr = np.array(spec_masks)
        assert spec_masks.shape[0] == len(mask[0]), "spec_masks must have the same length as mask[0]"
        
        if arr.dtype == bool:
            spec_dict = {i: np.where(arr[i])[0].tolist()
                         for i in range(len(arr))}
        else:
            labels = np.unique(arr)
            spec_dict = {lab: np.where(arr == lab)[0].tolist()
                         for lab in labels}
        
    id_masks = []
    for m in mask:
        m_arr = np.array(m)
        if m_arr.dtype == bool:
            id_masks.append(np.where(m_arr)[0].tolist())
        else:
            id_masks.append(m_arr.tolist())

    # Matrix mit shape (n_true_classes, n_pred_classes) erzeugen
    class_summary = []
    for m_indices in id_masks:
        true_set = set(m_indices)
        row = [len(true_set.intersection(pred_indices))
               for _, pred_indices in spec_dict.items()]
        class_summary.append(row)

    class_summary = np.array(class_summary, dtype=int)

    n_true = class_summary.shape[0]
    n_pred = class_summary.shape[1]
    if n_pred < n_true:
        padding = np.zeros((n_true, n_true - n_pred), dtype=int)
        class_summary = np.hstack([class_summary, padding])
    elif n_pred > n_true:
        padding = np.zeros((n_pred - n_true, n_pred), dtype=int)
        class_summary = np.vstack([class_summary, padding])
        class_summary = class_summary[:n_pred, :]  # just in case

    return class_summary
