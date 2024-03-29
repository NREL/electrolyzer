"""
This module provides utilities for serializing/deserializing classes using `attrs`.
"""
from typing import Any, Tuple, Union, Callable, Iterable

import attrs
import numpy as np
import numpy.typing as npt
from attrs import Attribute, define


float_type = np.float64

NDArrayFloat = npt.NDArray[float_type]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFilter = Union[npt.NDArray[np.int_], npt.NDArray[np.bool_]]
NDArrayObject = npt.NDArray[np.object_]


def array_converter(data: Iterable) -> np.ndarray:
    try:
        a = np.array(data, dtype=float_type)
    except TypeError as e:
        raise TypeError(e.args[0] + f". Data given: {data}")
    return a


def attr_serializer(inst: type, field: Attribute, value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def attr_filter(inst: Attribute, value: Any) -> bool:
    if inst.init is False:
        return False
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
    return True


def iter_validator(iter_type, item_types: Union[Any, Tuple[Any]]) -> Callable:
    """Helper function to generate iterable validators that will reduce the amount of
    boilerplate code.

    Parameters
    ----------
    iter_type : any iterable
        The type of iterable object that should be validated.
    item_types : Union[Any, Tuple[Any]]
        The type or types of acceptable item types.

    Returns
    -------
    Callable
        The attr.validators.deep_iterable iterable and instance validator.
    """
    validator = attrs.validators.deep_iterable(
        member_validator=attrs.validators.instance_of(item_types),
        iterable_validator=attrs.validators.instance_of(iter_type),
    )
    return validator


@define
class FromDictMixin:
    """
    A Mixin class to allow for kwargs overloading when a data class doesn't
    have a specific parameter definied. This allows passing of larger dictionaries
    to a data class without throwing an error.
    """

    @classmethod
    def from_dict(cls, data: dict):
        """Maps a data dictionary to an `attr`-defined class.

        TODO: Add an error to ensure that either none or all the parameters
            are passed in

        Args:
            data : dict
                The data dictionary to be mapped.
        Returns:
            cls
                The `attr`-defined class.
        """
        # Check for any inputs that aren't part of the class definition
        class_attr_names = [a.name for a in cls.__attrs_attrs__]
        extra_args = [d for d in data if d not in class_attr_names]
        if len(extra_args):
            raise AttributeError(
                f"""The initialization for {cls.__name__}
                was given extraneous inputs: {extra_args}"""
            )

        kwargs = {
            a.name: data[a.name]
            for a in cls.__attrs_attrs__
            if a.name in data and a.init
        }

        # Map the inputs must be provided:
        # 1) must be initialized, 2) no default value defined
        required_inputs = [
            a.name for a in cls.__attrs_attrs__ if a.init and a.default is attrs.NOTHING
        ]
        undefined = sorted(set(required_inputs) - set(kwargs))

        if undefined:
            raise AttributeError(
                f"""The class definition for {cls.__name__}
                is missing the following inputs: {undefined}"""
            )
        return cls(**kwargs)

    def as_dict(self) -> dict:
        """Creates a JSON and YAML friendly dictionary that can be saved for future
        reloading.  This dictionary will contain only Python types that can later
        be converted to their proper formats.

        Returns:
            dict: All key, vaue pais required for class recreation.
        """
        return attrs.asdict(self, filter=attr_filter, value_serializer=attr_serializer)
