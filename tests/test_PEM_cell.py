"""This module provides unit tests for `PEMCell`"""

import pytest

from electrolyzer import PEMCell


# from numpy.testing import assert_almost_equal


@pytest.fixture
def cell():
    return PEMCell.from_dict()
