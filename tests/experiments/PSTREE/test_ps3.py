import numpy as np
import time 
from tests.metrics_test import *
from tests.experiments.PSTREE.config_ps3 import *
from tests.misc_functions import get_classification_summary
from tests.metrics_test import calc_scores_from_summary as calc_scores
from pstree.cluster_gp_sklearn import PSTreeRegressor


def ps3_test(best_params, 
               dataset, 
               split_id, 
               seed,
               **kwargs):
    
    dataset = dataset.copy()
    params = best_params.copy()

    mask = dataset.pop('mask', None) 
    bcv_rmse = params.pop('bcv_rmse')
    X_tr, y_tr = dataset['X_train'], dataset['y_train']
    X_te, y_te = dataset['X_test'], dataset['y_test']

    reg = PSTreeRegressor(
        **params,
        test_data = (X_te, y_te),
        random_seed  = seed,
        random_state = seed,
    )
    
    t0 = time.time()
    reg.fit(X_tr, y_tr)
    
    elapsed         = time.time() - t0
    labels          = reg.predict_labels(X_tr)
    rmse_train      = rmse(reg.predict(X_tr), y_tr)
    rmse_test       = rmse(reg.predict(X_te), y_te)
    r2_train        = r_squared(y_tr, reg.predict(X_tr))
    r2_test         = r_squared(y_te, reg.predict(X_te))
    class_n         = len(np.unique(labels))
    overfit         = (100 * (rmse_train - rmse_test) / rmse_train)
    gen_gap         = 100 * abs(rmse_test - bcv_rmse) / bcv_rmse
    logs            = reg.regr.log_book 
    pop_stats       = []
    
    extra_records = {}
    if mask is not None:
        class_summary = get_classification_summary(
                X_data      = X_tr, 
                mask        = mask,
                spec_masks  = labels
                )
        
        acc, macro_f1, weighted_f1 = calc_scores(class_summary)   
        extra_records = {
            'acc'           : acc,
            'macro_f1'      : macro_f1,
            'weighted_f1'   : weighted_f1,
            'class_summary' : str(class_summary)
        }     
        
    records = {
        'dataset_name'          : best_params['dataset_name'],
        'split_id'              : split_id,
        'trial_id'              : seed,
        'seed'                  : seed,
        'rmse_train'            : rmse_train,
        'rmse_test'             : rmse_test,
        'r2_train'              : r2_train,
        'r2_test'               : r2_test,
        'nodes'                 : reg.regr.nodes_count,
        'classes_n'             : class_n,
        'overfit'               : overfit,
        'gen_gap'               : gen_gap,
        'time'                  : elapsed,
        **extra_records
    }

    return records, pop_stats, logs 