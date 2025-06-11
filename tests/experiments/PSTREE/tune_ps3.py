
import numpy as np
import hashlib
from sklearn.model_selection import KFold
from tests.metrics_test import *
from tests.metrics
from tests.experiments.PSTREE.config_ps3 import *
from pstree.cluster_gp_sklearn import PSTreeRegressor

def ps3_tune(gen_params, 
               dataset, 
               split_id,
               n_splits=5):
    
    params = gen_params.copy()
    data_all = dataset.copy()
    mask = data_all.pop('mask')
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_id)
    rmses_tr, rmses_te, nodes, tnodes = [], [], [], []
    best_ensemble_rmses, best_ensemble_sizes, norm_errs = [], [], []

    for i, (idx_tr, idx_te) in enumerate(kf.split(data_all['X_train'])):
        X_tr, y_tr = data_all['X_train'][idx_tr], data_all['y_train'][idx_tr]
        X_te, y_te = data_all['X_train'][idx_te], data_all['y_train'][idx_te]
        mask_kf = [marr[idx_tr] for marr in mask] if mask is not None else None

        reg = PSTreeRegressor(
            **params,
            random_seed  = split_id + i,
            random_state = split_id + i, 
        )
        reg.fit(X_tr, y_tr)
        
        rmse_train      = rmse(reg.predict(X_tr), y_tr)
        rmse_test       = rmse(reg.predict(X_te), y_te)
        r2_train        = r2_score(y_tr, r.predict(X_tr))
        r2_test         = r2_score(y_te, r.predict(X_te))
        class_n         = np.nunique(reg.labels)
        class_summary   = []
        
        rmses_tr.append(rmse_train)
        rmses_te.append(rmse_test)
        tnodes.append(elite.total_nodes)
        nodes.append(elite.nodes_count)

    stats_general = {
        'mean_tnodes_elite' : float(np.mean(tnodes)),
        'mean_nodes'        : float(np.mean(nodes)),
        'rmse_train'        : float(np.mean(rmses_tr)),
    }

    if mask is not None: 
        stats_general.update({
            'sizes_spec_ens'    : float(np.mean(best_ensemble_sizes)),
            'norm_errs_ens'     : float(np.mean(norm_errs)),
            'ensemble_rmse'     : float(np.mean(best_ensemble_rmses)),
            'divergence_tr'     : float(np.mean(rmses_tr) / np.mean(best_ensemble_rmses)),
        })

    return float(np.mean(rmses_te)), stats_general, {
        'std_rmse_elite'    : float(np.std(rmses_te)),
        'std_nodes_elite'   : float(np.std(tnodes)),
        'std_nodes'         : float(np.std(nodes)),
    }