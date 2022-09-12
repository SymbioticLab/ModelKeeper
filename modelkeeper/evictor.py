from ortools.linear_solver import pywraplp


def mip(weight_threshold, weight_list, util_list):
    """
    A Integer Programming model that solves the 0-1 Knapsack problem.
    Args:
        weight_threshold: Weight threshold
        weight_list: List of weights for each item in item set I
        util_list: utility score of each item i
    Returns:
        The optimal utility score of the knapsack problem
    """
    n = len(weight_list)

    # initialize the integer programming model with the open source CBC solver
    solver = pywraplp.Solver('simple_mip_program',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Declare binary variable x for each item from 1 to n
    x_dict = []
    for i in range(n):
        x_dict.append(solver.IntVar(0, 1, f'x_{i}'))
    # Add constraint on total weight of items selected cannot exceed   weight
    # threshold
    solver.Add(solver.Sum([weight_list[i] * x_dict[i]
                           for i in range(n)]) <= weight_threshold)
    # Maximize total utility score
    solver.Maximize(solver.Sum([util_list[i] * x_dict[i] for i in range(n)]))

    status = solver.Solve()

    # Uncomment the section below to print solution details
    # if status == pywraplp.Solver.OPTIMAL:
    # print('Problem solved in %f milliseconds' % solver.wall_time())
    res_x = [x_dict[i].solution_value() for i in range(n)]

    return solver.Objective().Value(), res_x

# model_size = 100
# weight_threshold = model_size-1
# weight_list = [1] * model_size
# value_list = [1] * model_size #np.random.random(model_size)
# mip(weight_threshold, weight_list, value_list)
