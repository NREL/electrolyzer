import os

from numpy.testing import assert_almost_equal

from electrolyzer.inputs import validation as val
from electrolyzer.glue_code.optimization import calc_rated_stack


input_modeling = "./test_modeling_options.yaml"
fname_input_modeling = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "test_modeling_options.yaml"
)


def test_calc_rated_stack():
    """
    Should be able to handle a case where the desired rating is lower than
    the calculated rating.
    """
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    # Update modeling dict in place
    calc_rated_stack(modeling_options)

    assert modeling_options["electrolyzer"]["stack"]["n_cells"] == 108
    assert_almost_equal(
        modeling_options["electrolyzer"]["stack"]["cell_area"], 1034.059, decimal=3
    )
    assert_almost_equal(
        modeling_options["electrolyzer"]["stack"]["stack_rating_kW"], 500.000, decimal=3
    )


def test_calc_rated_stack_new_copy():
    """
    Perform the same function as `test_calc_rated_stack`, but return a new dict.
    """
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    # Update modeling dict in place
    updated_model = calc_rated_stack(modeling_options, in_place=False)

    # make sure we didn't overwrite the original
    assert modeling_options["electrolyzer"]["stack"]["n_cells"] == 100

    assert updated_model["electrolyzer"]["stack"]["n_cells"] == 108
    assert_almost_equal(
        updated_model["electrolyzer"]["stack"]["cell_area"], 1034.059, decimal=3
    )
    assert_almost_equal(
        updated_model["electrolyzer"]["stack"]["stack_rating_kW"], 500.000, decimal=3
    )


def test_calc_rated_stack_lower():
    """
    Should be able to handle a case where the desired rating is higher than
    the calculated rating.
    """
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    # set the desired stack rating, this time lower than baseline
    modeling_options["electrolyzer"]["stack"]["stack_rating_kW"] = 750

    # set the maximum number of cells
    # modeling_options["electrolyzer"]["stack"]["n_cells"] = 120

    calc_rated_stack(modeling_options)

    assert modeling_options["electrolyzer"]["stack"]["n_cells"] == 161
    assert_almost_equal(
        modeling_options["electrolyzer"]["stack"]["cell_area"], 1007.019, decimal=3
    )
    assert_almost_equal(
        modeling_options["electrolyzer"]["stack"]["stack_rating_kW"], 750.000, decimal=3
    )
