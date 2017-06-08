import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d

try:
    from cydtw import dtw_path
except:
    print("Could not import efficient DTW computation from cydtw package, using pure Python alternative (slower)")
    print("\'pip install cydtw\' should help")
    from utils import dtw_path

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class DTWSampler(BaseEstimator, TransformerMixin):
    def __init__(self, scaling_col_idx=0, reference_idx=0, d=1, n_samples=100,interp_kind="slinear", save_path=False):
        self.scaling_col_idx = scaling_col_idx
        self.reference_idx = reference_idx
        self.n_samples = n_samples
        self.d = d
        self.interp_kind = interp_kind
        self.reference_series = None
        
        # if saving dtw_Path
        self.save_path = save_path
        self.saved_dtw_path = None
        
    def fit(self, X):
        _X = X.reshape((X.shape[0], -1, self.d))
        end = last_index(_X[self.reference_idx])
        self.reference_series = resampled(_X[self.reference_idx, :end, self.scaling_col_idx], n_samples=self.n_samples,
                                          kind=self.interp_kind)
        return self

    def transform_3d(self, X):
        X_resampled = numpy.zeros((X.shape[0], self.n_samples, X.shape[2]))
        for i in range(X.shape[0]):
            end = last_index(X[i])
            for j in range(X.shape[2]):
                X_resampled[i, :, j] = resampled(X[i, :end, j], n_samples=self.n_samples, kind=self.interp_kind)
            # Compute indices based on alignment of dimension self.scaling_col_idx with the reference
            indices_xy = [[] for _ in range(self.n_samples)]

            if i == self.reference_idx:
                path = [(idx, idx) for idx in range(self.reference_series.shape[0])]
            elif self.saved_dtw_path is None:
                path, _ = dtw_path(X_resampled[i, :, self.scaling_col_idx].reshape((-1, 1)),
                                self.reference_series.reshape((-1, 1)))
                if self.save_path:
                    self.saved_dtw_path = path
            else:
                path = self.saved_dtw_path

            for t_current, t_ref in path:
                indices_xy[t_ref].append(t_current)
            for j in range(X.shape[2]):
                ynew = numpy.array([numpy.mean(X_resampled[i, indices, j]) for indices in indices_xy])
                X_resampled[i, :, j] = ynew
        return X_resampled

    def transform(self, X):
        _X = X.reshape((X.shape[0], -1, self.d))
        return self.transform_3d(_X).reshape((X.shape[0], -1))

    def dump(self, fname):
        numpy.savetxt(fname, self.reference_series)


def resampled(X, n_samples=100, kind="linear"):
    xnew = numpy.linspace(0, 1, n_samples)
    f = interp1d(numpy.linspace(0, 1, X.shape[0]), X, kind=kind)
    return f(xnew)


def last_index(X):
    timestamps_infinite = numpy.all(~numpy.isfinite(X), axis=1)  # Are there NaNs padded after the TS?
    if numpy.alltrue(~timestamps_infinite):
        idx = X.shape[0]
    else:  # Yes? then remove them
        idx = numpy.nonzero(timestamps_infinite)[0][0]
    return idx
    
    
if __name__=='__main__':
    import time
    npy_arr = numpy.load('data/sample.npy')
    s = DTWSampler(scaling_col_idx=0, reference_idx=0, d=1, n_samples=20, save_path=True)
    t0 = time.time()
    for i in range(1000):
        transformed_array = s.fit_transform(npy_arr)
    print(time.time()-t0)
