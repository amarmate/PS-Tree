
import numpy as np
import time
from sklearn.model_selection import KFold
from tests.metrics_test import *
from tests.experiments.PSTREE.config_ps3 import *
from pstree.cluster_gp_sklearn import PSTreeRegressor
from tests.misc_functions import get_classification_summary
from tests.metrics_test import calc_scores_from_summary as calc_scores

def ps3_tune(gen_params, 
               dataset, 
               split_id,
               n_splits=5,
               **kwargs):
    
    params = gen_params.copy()
    data_all = dataset.copy()
    mask = data_all.pop('mask')
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses_tr, rmses_te, nodes, overfit, times = [], [], [], [], []
    r2_tr, r2_te, classes_n = [], [], []
    accs, macro_f1s, weighted_f1s = [], [], []

    for i, (idx_tr, idx_te) in enumerate(kf.split(data_all['X_train'])):
        X_tr, y_tr = data_all['X_train'][idx_tr], data_all['y_train'][idx_tr]
        X_te, y_te = data_all['X_train'][idx_te], data_all['y_train'][idx_te]
        mask_kf = [marr[idx_tr] for marr in mask] if mask is not None else None
        
        reg = PSTreeRegressor(
            **params,
            random_seed  = split_id + i,
            random_state = split_id + i, 
        )
        t0 = time.time()
        reg.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        
        labels = reg.predict_labels(X_tr)
        rmse_train      = rmse(reg.predict(X_tr), y_tr)
        rmse_test       = rmse(reg.predict(X_te), y_te)
        r2_train        = r_squared(y_tr, reg.predict(X_tr))
        r2_test         = r_squared(y_te, reg.predict(X_te))
        class_n         = len(np.unique(labels))
        
        rmses_tr.append(rmse_train)
        rmses_te.append(rmse_test)
        r2_tr.append(r2_train)
        r2_te.append(r2_test)
        nodes.append(reg.regr.nodes_count)
        classes_n.append(class_n)
        overfit.append(100 * (rmse_train - rmse_test) / rmse_train)
        times.append(elapsed)
        
        if mask is not None:
             class_summary = get_classification_summary(
                 X_data = X_tr, 
                 mask = mask_kf,
                 spec_masks = labels,
                 )
             
             acc, macro_f1, weighted_f1 = calc_scores(class_summary)    
             accs.append(acc)
             macro_f1s.append(macro_f1)
             weighted_f1s.append(weighted_f1)      
             
    stats_general = {
        'rmse_tr'        : float(np.mean(rmses_tr)),
        'r2_tr'          : float(np.mean(r2_tr)),
        'r2_te'          : float(np.mean(r2_te)),
        'overfit'        : float(np.mean(overfit)),
        'nodes'          : float(np.mean(nodes)),
        'classes_n'      : float(np.mean(classes_n)),
        'time'           : float(np.mean(times)),
    }

    if mask is not None: 
        stats_general.update({
            'acc'           : float(np.mean(accs)),
            'macro_f1'      : float(np.mean(macro_f1s)),
            'weighted_f1'   : float(np.mean(weighted_f1s)),
        })

    return float(np.mean(rmses_te)), stats_general, {
        'std_rmse_elite'    : float(np.std(rmses_te)),
        'std_nodes'         : float(np.std(nodes)),
    }