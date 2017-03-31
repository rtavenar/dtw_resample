import numpy
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.interpolate import interp1d

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class DTWSampler(BaseEstimator, TransformerMixin):
    def __init__(self, scaling_col_idx=0, reference_idx=0, n_sampling=100):
        self.scaling_col_idx = scaling_col_idx
        self.reference_idx = reference_idx
        self.n_sampling = n_sampling
        self.reference_series = None

    def fit(self, X):
        xnew = numpy.linspace(0, 1, self.n_sampling)
        ref_dim0 = X[self.reference_idx, :, self.scaling_col_idx]
        f = interp1d(numpy.linspace(0, 1, ref_dim0.shape[0]), ref_dim0, kind='slinear')
        self.reference_series = f(xnew)

    def transform(self, X):
        X_resampled = numpy.zeros((X.shape[0], self.n_sampling, X.shape[2]))
        xnew = numpy.linspace(0, 1, self.n_sampling)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                if j == self.scaling_col_idx:
                    X_resampled[i, :, j] = xnew
                else:
                    indices_xy = [[] for _ in range(self.n_sampling)]
                    for t_current, t_ref in dtw_path(X[i, :, self.scaling_col_idx], self.reference_series):
                        indices_xy[t_ref].append(t_current)
                    ynew = numpy.array([numpy.mean(X[i, indices, j]) for indices in indices_xy])
                    f = interp1d(numpy.linspace(0, 1, self.n_sampling), ynew, kind='slinear')
                    X_resampled[i, :, j] = f(xnew)
        return X_resampled

    def dump(self, fname):
        numpy.savetxt(fname, self.reference_series)


def empty_mask(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    mask = numpy.zeros((l1 + 1, l2 + 1))
    mask[1:, 0] = numpy.inf
    mask[0, 1:] = numpy.inf
    return mask


def dtw_path(s1, s2):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    cum_sum = numpy.zeros((l1 + 1, l2 + 1))
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf
    predecessors = [([None] * l2) for i in range(l1)]

    for i in range(l1):
        for j in range(l2):
            if numpy.isfinite(cum_sum[i + 1, j + 1]):
                dij = numpy.linalg.norm(s1[i] - s2[j]) ** 2
                pred_list = [cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]]
                argmin_pred = numpy.argmin(pred_list)
                cum_sum[i + 1, j + 1] = pred_list[argmin_pred] + dij
                if i + j > 0:
                    if argmin_pred == 0:
                        predecessors[i][j] = (i - 1, j)
                    elif argmin_pred == 1:
                        predecessors[i][j] = (i, j - 1)
                    else:
                        predecessors[i][j] = (i - 1, j - 1)

    i = l1 - 1
    j = l2 - 1
    best_path = [(i, j)]
    while predecessors[i][j] is not None:
        i, j = predecessors[i][j]
        best_path.insert(0, (i, j))

    return best_path