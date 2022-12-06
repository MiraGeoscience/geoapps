#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.driver_base.params import BaseParams

import numpy as np

from geoapps.shared_utils.utils import rotate_xyz

from .simpeg_factory import SimPEGFactory


class SourcesFactory(SimPEGFactory):
    """Build SimPEG sources objects based on factory type."""

    def __init__(self, params: BaseParams):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):

        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import sources

            return sources.SourceField

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import sources

            return sources.SourceField

        elif "direct current" in self.factory_type:
            from SimPEG.electromagnetics.static.resistivity import sources

            return sources.Dipole

        elif "induced polarization" in self.factory_type:
            from SimPEG.electromagnetics.static.induced_polarization import sources

            return sources.Dipole

        elif self.factory_type in ["magnetotellurics", "tipper"]:
            from SimPEG.electromagnetics.natural_source import sources

            return sources.Planewave_xy_1Dprimary

    def assemble_arguments(
        self,
        receivers=None,
        locations=None,
        frequency=None,
    ):  # pylint: disable=arguments-differ
        """Provides implementations to assemble arguments for sources object."""

        args = []

        if locations is not None and getattr(self.params.mesh, "rotation", None):
            locations = rotate_xyz(
                locations,
                self.params.mesh.origin.tolist(),
                -1 * self.params.mesh.rotation[0],
            )

        if self.factory_type in [
            "direct current 3d",
            "direct current 2d",
            "induced polarization 3d",
            "induced polarization 2d",
        ]:
            args += self._dcip_arguments(
                receivers=receivers,
                locations=locations,
            )

        elif self.factory_type in ["magnetotellurics", "tipper"]:
            args.append(receivers)
            args.append(frequency)

        else:
            args.append([receivers])

        return args

    def assemble_keyword_arguments(  # pylint: disable=arguments-differ
        self, receivers=None, locations=None, frequency=None
    ):
        """Provides implementations to assemble keyword arguments for receivers object."""
        _ = (receivers, locations, frequency)
        kwargs = {}
        if self.factory_type in ["magnetic scalar", "magnetic vector"]:
            kwargs["parameters"] = self.params.inducing_field_aid()
        if self.factory_type in ["magnetotellurics", "tipper"]:
            kwargs["sigma_primary"] = [self.params.background_conductivity]

        return kwargs

    def build(
        self, receivers=None, locations=None, frequency=None
    ):  #  pylint: disable=arguments-differ
        return super().build(
            receivers=receivers,
            locations=locations,
            frequency=frequency,
        )

    def _dcip_arguments(self, receivers=None, locations=None):

        args = []

        locations_a = locations[0]
        locations_b = locations[1]
        args.append([receivers])
        args.append(locations_a)

        if np.all(locations_a == locations_b):
            if "direct current" in self.factory_type:
                from SimPEG.electromagnetics.static.resistivity import sources
            else:
                from SimPEG.electromagnetics.static.induced_polarization import sources
            self.simpeg_object = sources.Pole
        else:
            args.append(locations_b)

        return args
