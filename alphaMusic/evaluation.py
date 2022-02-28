import numpy as np


def best_match_sorted(doas_est, doas_ref):
    # sort the values
    doas_est = np.sort(doas_est)
    doas_ref = np.sort(doas_ref)

    best_match = []
    for ref in doas_ref:
        idx = np.argmin(np.abs(doas_est - ref))
        est = doas_est[idx]
        doas_est = np.delete(doas_est, idx)
        best_match.append(est)

    return best_match


def compute_ssl_metrics(doas_est, doas_ref):

    # sort the values
    doas_est = np.sort(doas_est)
    doas_ref = np.sort(doas_ref)

    Nest = len(doas_est)
    Nref = len(doas_ref)

    results = np.zeros((Nest, Nref))
    for i in range(Nest):
        sq_error_val = []
        for j in range(Nref):
            values =   [np.abs(doas_est[i] - doas_ref[j]),
                        np.abs(doas_est[i] - doas_ref[j] - 180),
                        np.abs(doas_est[i] - doas_ref[j] + 180)]
            results[i, j] = min(values)

    scores = np.min(results, axis=1)

    ACC5 = 100 * (np.sum(scores < 5.) / Nest)
    ACC10 = 100 * (np.sum(scores < 10.) / Nest)
    RMSE = np.sqrt((scores**2).mean())
    MAE = scores.mean()
    MISS = 100 * ((Nref - Nest) / Nref)

    metrics = {
        'MAE' : MAE,
        'RMSE' : RMSE,
        'ACC5' : ACC5,
        'ACC10': ACC10,
        'MISS' : MISS,
    }
    
    return metrics