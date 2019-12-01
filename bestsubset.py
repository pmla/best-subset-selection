import numpy as np
from gurobipy import Model, GRB


def build_skeleton_model(model_type, n, num_features, params):

    # Create model
    model = Model(model_type)

    # Set user parameters
    for k, v in sorted(params.items(),
                       key=lambda x: x[0].lower() == 'outputflag',
                       reverse=True):
        model.setParam(k, v)

    # Create common variables
    x = {j: model.addVar(vtype=GRB.CONTINUOUS,
                         lb=-float("inf"),
                         ub=float("inf")) for j in range(n)}
    y = {j: model.addVar(vtype=GRB.BINARY) for j in range(n)}

    # Set indicator constraints
    for j in range(n):
        model.addConstr((y[j] == 0) >> (x[j] == 0),
                        'indicator_{}'.format(j))

    # Set column constraints
    model.addConstr(sum([y[j] for j in range(n)]) == num_features,
                    'num_features')
    return model, x, y


def solve_model(model, x, y):
    # Solve model and read out variables
    model.optimize()
    objective = model.getAttr("ObjVal")
    status = model.getAttr("Status")
    result = {j: x[j].x for j in sorted(x.keys()) if y[j].x > 0.5}
    return status, objective, result


def solve_rmse(A, b, num_features, user_params={}):

    # Gurobi has some numerical problems with MIQP if presolve is used and
    # needs tight tolerances to stop constraint violations
    params = {'Presolve': 0,
              'FeasibilityTol': 1E-9,
              'IntFeasTol': 1E-9,
              'MIPFocus': 2}    # Focus on proving optimality
    params.update(user_params)
    m, n = A.shape
    model, x, y = build_skeleton_model("miqp", n, num_features, params)

    if n < m:
        # Over-determined case
        Q = A.T @ A
        c = -2 * A.T @ b
        constant = np.sum(b**2)

        # Set objective
        model.setObjective(sum([sum([Q[j, k] * x[k] for k in range(n)]) * x[j]
                                for j in range(n)]) +
                           sum([c[j] * x[j] for j in range(n)]) +
                           constant, GRB.MINIMIZE)
    else:
        # Under-determined (or high-dimensional) case
        error = {i: model.addVar(vtype=GRB.CONTINUOUS,
                                 lb=-float("inf"),
                                 ub=+float("inf")) for i in range(m)}

        # Define errors
        for i in range(m):
            model.addConstr(error[i] == sum([A[i, j] * x[j]
                                            for j in range(n)]) - b[i])

        # Set objective
        model.setObjective(sum([error[i] * error[i]
                                for i in range(m)]), GRB.MINIMIZE)
    return solve_model(model, x, y)


def solve_mae(A, b, num_features, user_params={}):

    params = {'Presolve': 2,    # Apply maximum presolve
              'MIPGap': 0,      # Prevent early termination
              'MIPFocus': 2}    # Focus on proving optimality
    params.update(user_params)
    m, n = A.shape
    model, x, y = build_skeleton_model("mip", n, num_features, params)

    # Under-determined (or high-dimensional) case
    error = {i: model.addVar(vtype=GRB.CONTINUOUS,
                             lb=0,
                             ub=+float("inf")) for i in range(m)}

    # Define errors
    for i in range(m):
        model.addConstr(error[i] >= sum([A[i, j] * x[j]
                                        for j in range(n)]) - b[i])
        model.addConstr(error[i] >= -(sum([A[i, j] * x[j]
                                           for j in range(n)]) - b[i]))

    # Set objective
    model.setObjective(sum(error.values()), GRB.MINIMIZE)
    return solve_model(model, x, y)
