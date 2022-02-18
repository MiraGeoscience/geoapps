#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from abc import ABC, abstractmethod


class AbstractFactory(ABC):
    def __init__(self, params):
        self.params = params
        super().__init__()

    @property
    @abstractmethod
    def factory_type(self):
        """Returns type used to switch concrete build methods."""

    @property
    @abstractmethod
    def concrete_object(self):
        """Returns a class to be constructed by the build method."""

    @abstractmethod
    def build(self):
        """Constructs concrete object for provided factory type."""
