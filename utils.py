import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


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

    return best_path, cum_sum[-1, -1]