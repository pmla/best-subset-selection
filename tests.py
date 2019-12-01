import pytest
from numpy.testing import assert_allclose
import numpy as np
import bestsubset


TOL = 1E-10


@pytest.mark.parametrize("k", range(1, 5))
def test_rmse_overdetermined(k):
    m, n = 20, 10
    rng = np.random.RandomState(0)
    A = rng.uniform(size=(m, n))
    b = rng.uniform(size=m)
    params = {'OutputFlag': 0, 'Threads': 1, 'TimeLimit': 3}

    status, objective, result = bestsubset.solve_rmse(A, b, k, params=params)
    columns, values = zip(*sorted(result.items()))
    res = np.linalg.lstsq(A[:, list(columns)], b, rcond=-1)
    xcheck, objective_check, _, _ = res
    assert_allclose(xcheck, values, atol=TOL)
    assert_allclose(objective, objective_check, atol=TOL)


@pytest.mark.parametrize("k", range(1, 5))
def test_rmse_underdetermined(k):
    m, n = 10, 20
    rng = np.random.RandomState(0)
    A = rng.uniform(size=(m, n))
    b = rng.uniform(size=m)
    params = {'OutputFlag': 0, 'Threads': 1, 'TimeLimit': 3}

    status, objective, result = bestsubset.solve_rmse(A, b, k, params=params)
    columns, values = zip(*sorted(result.items()))
    res = np.linalg.lstsq(A[:, list(columns)], b, rcond=-1)
    xcheck, objective_check, _, _ = res
    assert_allclose(xcheck, values, atol=TOL)
    assert_allclose(objective, objective_check, atol=TOL)


@pytest.mark.parametrize("k", range(1, 5))
def test_mae(k):
    m, n = 10, 20
    rng = np.random.RandomState(0)
    A = rng.uniform(size=(m, n))
    b = rng.uniform(size=m)
    params = {'OutputFlag': 0, 'Threads': 1, 'TimeLimit': 3}

    status, objective, result = bestsubset.solve_mae(A, b, k, params=params)
    columns, values = zip(*sorted(result.items()))
    deltas = A[:, list(columns)] @ list(values) - b
    objective_check = np.sum(np.abs(deltas))
    assert_allclose(objective, objective_check, atol=TOL)


@pytest.mark.parametrize("optfunc", [bestsubset.solve_rmse,
                                     bestsubset.solve_mae])
def test_initial_features(optfunc):
    m, n = 20, 10
    rng = np.random.RandomState(0)
    A = rng.uniform(size=(m, n))
    b = rng.uniform(size=m)
    params = {'OutputFlag': 0, 'Threads': 1, 'TimeLimit': 3}
    tparams = {'OutputFlag': 0, 'Threads': 1, 'NodeLimit': 0}

    initial_features = {}
    for k in range(1, 5):
        status, objective, result = optfunc(A, b, k, initial_features, params)
        initial_features = result
        _, objective1, _ = optfunc(A, b, k + 1, initial_features,
                                   params=tparams)
        assert objective1 <= objective + TOL
