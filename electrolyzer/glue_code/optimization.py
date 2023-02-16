import copy

from scipy.optimize import fsolve

from electrolyzer import Stack


def calc_rated_system(modeling_options: dict):
    """
    Calculates number of stacks and stack power rating (kW) to match a desired
    system rating (MW).

    Args:
        modeling_options (dict): An options Dict compatible with the modeling schema
    """
    options = copy.deepcopy(modeling_options)

    system_rating_kW = options["electrolyzer"]["control"]["system_rating_MW"] * 1e3
    stack_rating_kW = options["electrolyzer"]["stack"]["stack_rating_kW"]

    # determine number of stacks (int) closest to stack rating (float)
    n_stacks = round(system_rating_kW / stack_rating_kW)
    options["electrolyzer"]["control"]["n_stacks"] = n_stacks

    # determine new desired rating to adjust parameters for
    new_rating = system_rating_kW / n_stacks
    options["electrolyzer"]["stack"]["stack_rating_kW"] = new_rating

    # solve for new stack rating (modifies dict)
    calc_rated_stack(options)

    return options


def _solve_rated_stack(desired_rating: float, stack: Stack):
    cell_area = stack.cell.cell_area

    # root finding function
    def calc_rated_power_diff(cell_area: float):
        stack.cell.cell_area = cell_area
        p_rated = stack.calc_stack_power(stack.max_current)

        return p_rated - desired_rating

    return fsolve(calc_rated_power_diff, cell_area)


def calc_rated_stack(modeling_options: dict):
    """
    For a given model specification, determines a configuration that meets the
    desired stack rating (kW). Only modifies `n_cells` and `cell_area`.

    NOTE: This is a naive approach: it is only concerned with achieving the desired
        power rating. Any checks on the validity of the resulting design must be
        performed by the user.

    Args:
        modeling_options (dict): An options Dict compatible with the modeling schema
    """
    stack = Stack.from_dict(modeling_options["electrolyzer"]["stack"])

    n_cells = stack.n_cells

    # start with an initial calculation of stack power to compare with desired
    stack_p = stack.calc_stack_power(stack.max_current)
    desired_rating = stack.stack_rating_kW

    stack_p_prev = stack_p
    # nudge cell count up or down until it overshoots
    if stack_p > desired_rating:
        while stack_p > desired_rating:
            n_cells -= 1
            stack.n_cells = n_cells
            stack_p_prev = stack_p
            stack_p = stack.calc_stack_power(stack.max_current)

        # choose n_cells with closest resulting power rating
        if abs(desired_rating - stack_p_prev) < abs(desired_rating - stack_p):
            stack.n_cells += 1

    elif stack_p < desired_rating:
        while stack_p < desired_rating:
            n_cells += 1
            stack.n_cells = n_cells
            stack_p_prev = stack_p
            stack_p = stack.calc_stack_power(stack.max_current)

        # choose n_cells with closest resulting power rating
        if abs(desired_rating - stack_p_prev) < abs(desired_rating - stack_p):
            stack.n_cells -= 1

    # solve for optimal stack
    cell_area = _solve_rated_stack(desired_rating, stack)

    # recalc stack power
    stack.cell.cell_area = cell_area[0]
    stack_p = stack.calc_stack_power(stack.max_current)

    modeling_options["electrolyzer"]["stack"]["cell_area"] = cell_area[0]
    modeling_options["electrolyzer"]["stack"]["n_cells"] = n_cells
    modeling_options["electrolyzer"]["stack"]["stack_rating_kW"] = stack_p
