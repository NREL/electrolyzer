"""This module provides unit tests for `PEM_cell`"""

import pytest

from electrolyzer import PEM_Cell


# from numpy.testing import assert_almost_equal


@pytest.fixture
def cell():
    return PEM_Cell.from_dict()
