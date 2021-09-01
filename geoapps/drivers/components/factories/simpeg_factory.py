#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.io.params import Params


class SimPEGFactory:
    """
    Build SimPEG objects based on inversion type.

    Parameters
    ----------
    params :
        Driver parameters object.
    factory_type :
        Concrete factory type.
    simpeg_object :
        Abstract SimPEG object.

    Methods
    -------
    assemble_arguments():
        Assemble arguments for SimPEG object instantiation.
    assemble_keyword_arguments():
        Assemble keyword arguments for SimPEG object instantiation.
    build():
        Generate SimPEG object with assembled arguments and keyword arguments.
    """

    valid_factory_types = ["gravity", "magnetic", "mvi", "direct_current"]

    def __init__(self, params: Params):
        """
        :param params: Driver parameters object.
        :param factory_type: Concrete factory type.
        :param simpeg_object: Abstract SimPEG object.

        """
        self.params = params
        self.factory_type: str = params.inversion_type
        self.simpeg_object = None

    @property
    def factory_type(self):
        return self._factory_type

    @factory_type.setter
    def factory_type(self, p):
        if p not in self.valid_factory_types:
            msg = f"Factory type: {self.factory_type} not implemented yet."
            raise NotImplementedError(msg)
        else:
            self._factory_type = p

    def concrete_object(self):
        """To be over-ridden in factory implementations."""

    def assemble_arguments(self, **kwargs):
        """To be over-ridden in factory implementations."""
        return []

    def assemble_keyword_arguments(self, **kwargs):
        """To be over-ridden in factory implementations."""
        return {}

    def build(self, **kwargs):
        """To be over-ridden in factory implementations."""

        class_args = self.assemble_arguments(**kwargs)
        class_kwargs = self.assemble_keyword_arguments(**kwargs)
        return self.simpeg_object(*class_args, **class_kwargs)
