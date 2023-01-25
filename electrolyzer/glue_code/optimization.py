import copy

from electrolyzer import Stack


def calc_rated_stack(modeling_options, step=0.01, tol=0.001, limit=10e3, in_place=True):
    """
    For a given model specification, determines a configuration that meets the
    desired stack rating (kW). Only modifies `n_cells` and `cell_area`.

    NOTE: This is a naive approach: it is only concerned with achieving the desired
        power rating. Any checks on the validity of the resulting design must be
        performed by the user.

    Args:
        modeling_options (dict): An options Dict compatible with the modeling schema
        limit (int, optional): Iteration limit for `cell_area` optimization.
        step (float, optional): Step size when changing `cell_area` (cm^2)
        tol (float, optional): Tolerance for the rated power residual (kW)

    Returns:
        A new modeling options Dict containing adjusted stack parameters.
    """
    stack = Stack.from_dict(modeling_options["electrolyzer"]["stack"])

    n_cells = stack.n_cells

    # start with an initial calculation of stack power to compare with desired
    stack_p = stack.calc_stack_power(stack.max_current)
    desired_rating = stack.stack_rating_kW

    # nudge cell count up or down until it overshoots
    if stack_p > desired_rating:
        while stack_p > desired_rating:
            n_cells -= 1
            stack.n_cells = n_cells
            stack_p = stack.calc_stack_power(stack.max_current)
    elif stack_p < desired_rating:
        while stack_p < desired_rating:
            n_cells += 1
            stack.n_cells = n_cells
            stack_p = stack.calc_stack_power(stack.max_current)

    cell_area = stack.cell_area

    count = 0

    # nudge cell area up or down to minimize the residual
    while abs(stack_p - desired_rating) > tol:
        if count == limit:
            print(f"Iteration limit reached: {limit}")
            break

        if stack_p < desired_rating:
            cell_area -= step
        else:
            cell_area += step

        stack.cell_area = cell_area
        stack.cell.cell_area = cell_area
        stack_p = stack.calc_stack_power(stack.max_current)
        count += 1

    if in_place:
        modeling_options["electrolyzer"]["stack"]["cell_area"] = cell_area
        modeling_options["electrolyzer"]["stack"]["n_cells"] = n_cells
        modeling_options["electrolyzer"]["stack"]["stack_rating_kW"] = stack_p
    else:
        result = copy.deepcopy(modeling_options)
        result["electrolyzer"]["stack"]["cell_area"] = cell_area
        result["electrolyzer"]["stack"]["n_cells"] = n_cells
        result["electrolyzer"]["stack"]["stack_rating_kW"] = stack_p

        return result
