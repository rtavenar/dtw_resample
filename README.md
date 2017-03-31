# Resample data based on DTW alignment

## Purpose

The goal of this code is to allow irregular resampling of timestamped data. 
We assume we have a variable that informs about how advanced we are in the process for each timestamp in a time series.
This variable is called **base modality** in the following.
This variable could be, for example, the amount of discharge if we study water quality during a flood (_i.e._ in
this case, aligning discharge corresponds to aligning floods in a plausible way).
 
We will then use this variable to align other modalities of our dataset.
To do so, we record DTW path obtained when aligning base modality of each time series with that of a 
**reference time series**.
This path is then used to perform irregular resampling of time series in our dataset w.r.t. alignment of base 
modalities.

We refer the interested reader to the following publication for more details:
```
@article{dupas:halshs-01228397,
  TITLE = {{Identifying seasonal patterns of phosphorus storm dynamics with dynamic time warping}},
  AUTHOR = {Dupas, R{\'e}mi and Tavenard, Romain and Fovet, Oph{\'e}lie and Gilliet, Nicolas and Grimaldi, Catherine and Gascuel-Odoux, Chantal},
  JOURNAL = {{Water Resources Research}},
  PUBLISHER = {{American Geophysical Union}},
  VOLUME = {51},
  NUMBER = {11},
  PAGES = {8868--8882},
  YEAR = {2015},
  DOI = {10.1002/2015WR017338},
  PDF = {https://halshs.archives-ouvertes.fr/halshs-01228397/file/article_WRR_accepte_avec_fig.pdf}
}
```

Also, if you use our code in a scientific publication, it would be nice to cite us using the above-mentionned
reference :)

## Code details

Example tests are provided in files `test_sampling.py` and `test_clustering.py` [**TODO**].
In a few words, data should be resampled using the class `DTWSampler` that is almost a standard `sklearn` 
transformer.
Hence, fitting the sampler can be performed via:
```python
from sampler import DTWSampler

s = DTWSampler(scaling_col_idx=0, reference_idx=0)
s.fit(data)
```

Here, `data` is a 3-dimensional array of dimension $(n_ts, l, d)$ where $n_ts$ is the number of time series in the 
dataset, $l$ is the length of a time series and $d$ is the number of modalities provided for each time-stamp 
(including base modality).
`scaling_col_idx` is the index of the base modality and `reference_idx` is the index of the time series to be used as 
reference.


And applying the transformation to some new data is done via:
```python
s.transform(newdata)
```

If one wants to do both fit and transform on the same data, the following should do it:
```python
s.fit_transform(data)
```
