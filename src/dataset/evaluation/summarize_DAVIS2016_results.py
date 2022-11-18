import h5py
import numpy as np

fn = 'K:/results-test/output/dino_vit_b8_baseline_davis2016val_unsupervised_v4.h5'
# fn = 'K:/results-test/output/CIS.h5'

with h5py.File(fn, 'r') as f:
    j = f['J']

    total = 0
    total_count = 0
    means = []
    for k in j.keys():
        dat = np.array(j[k])
        select = ~np.isnan(dat)
        total_count += np.count_nonzero(select)
        total += dat[select].sum()

        means.append(np.nanmean(dat))

    J_mean_v1 = total / total_count
    J_mean_v2 = np.mean(means)
    print(round(J_mean_v1 * 100, 1))
    print(round(J_mean_v2 * 100, 1))
