import numpy
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from sampler import DTWSampler

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

ref_dim = 0
s = DTWSampler(scaling_col_idx=ref_dim, reference_idx=0, interp_kind="linear")
km = KMeans(n_clusters=3)

data = []
data.append(numpy.loadtxt("data/Xi_ref.txt"))
data.append(numpy.loadtxt("data/Xi_0.txt"))
data.append(numpy.loadtxt("data/Xi_1.txt"))

d = data[0].shape[1]

max_sz = max([ts.shape[0] for ts in data])
n_rep = 5

npy_arr = numpy.zeros((len(data) * n_rep, max_sz, d)) + numpy.nan
std_per_d = None
for idx_rep in range(n_rep):
    for idx, ts in enumerate(data):
        sz = ts.shape[0]
        npy_arr[idx + idx_rep * len(data), :sz] = ts + 0.1 * numpy.random.randn(sz, d) * ts.std(axis=0)

dtw_kmeans = Pipeline([('dtw_sampler', s), ('l2-kmeans', km)])

print(dtw_kmeans.fit_predict(npy_arr.reshape(-1, max_sz * d)))