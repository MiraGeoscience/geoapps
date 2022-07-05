#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import numpy as np

from geoapps.shared_utils.utils import rotate_xyz


class RectangularBlock:
    """
    Define a rotated rectangular block in 3D space
    :param length: U-size of the block
    :param width:  V-size of the block
    :param depth:  W-size of the block
    :param center: Position of the prism center
    :param dip: Orientation of the u-axis in degree from horizontal
    :param azimuth: Orientation of the u axis in degree from north
    :param reference: Point of rotation to be 'center' or 'top'
    """

    def __init__(self, **kwargs):
        self._center: list[float] = [0.0, 0.0, 0.0]
        self._length: float = 1.0
        self._width: float = 1.0
        self._depth: float = 1.0
        self._dip: float = 0.0
        self._azimuth: float = 0.0
        self._vertices: np.ndarray = None
        self._reference: str = "center"
        self.triangles: np.ndarray = np.vstack(
            [
                [0, 2, 1],
                [1, 2, 3],
                [0, 1, 4],
                [4, 1, 5],
                [1, 3, 5],
                [5, 3, 7],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [4, 5, 6],
                [6, 5, 7],
            ]
        )

        for attr, item in kwargs.items():
            try:
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def center(self) -> list[float]:
        """Prism center"""
        return self._center

    @center.setter
    def center(self, value: list[float]):
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(
                "Input value for 'center' must be a list of floats len(3)."
            )
        self._center = value
        self._vertices = None

    @property
    def length(self) -> float:
        """U-size of the block"""
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError("Input value for 'length' must be a float >0.")

        self._length = value
        self._vertices = None

    @property
    def width(self) -> float:
        """V-size of the block"""
        return self._width

    @width.setter
    def width(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError("Input value for 'width' must be a float >0.")

        self._width = value
        self._vertices = None

    @property
    def depth(self) -> float:
        """W-size of the block"""
        return self._depth

    @depth.setter
    def depth(self, value):
        if not isinstance(value, float) or value < 0:
            raise ValueError("Input value for 'depth' must be a float >0.")

        self._depth = value
        self._vertices = None

    @property
    def dip(self) -> float:
        """Orientation of the u-axis in degree from horizontal"""
        return self._dip

    @dip.setter
    def dip(self, value):
        if not isinstance(value, float) or value < -90.0 or value > 90.0:
            raise ValueError(
                "Input value for 'dip' must be a float on the interval [-90, 90] degrees."
            )

        self._dip = value
        self._vertices = None

    @property
    def azimuth(self) -> float:
        """Orientation of the u axis in degree from north"""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        if not isinstance(value, float) or value < -360.0 or value > 360.0:
            raise ValueError(
                "Input value for 'azimuth' must be a float on the interval [-360, 360] degrees."
            )

        self._azimuth = value
        self._vertices = None

    @property
    def reference(self) -> str:
        """Point of rotation to be 'center' or 'top'"""
        return self._reference

    @reference.setter
    def reference(self, value: str):
        if not isinstance(value, str) or value not in ["center", "top"]:
            raise ValueError(
                "Input value for 'reference' point should be a str from ['center', 'top']."
            )
        self._reference = value
        self._vertices = None

    @property
    def vertices(self) -> np.ndarray | None:
        """
        Prism eight corners in 3D space
        """

        if getattr(self, "_vertices", None) is None:
            x1, x2 = [
                -self.length / 2.0 + self.center[0],
                self.length / 2.0 + self.center[0],
            ]
            y1, y2 = [
                -self.width / 2.0 + self.center[1],
                self.width / 2.0 + self.center[1],
            ]
            z1, z2 = [
                -self.depth / 2.0 + self.center[2],
                self.depth / 2.0 + self.center[2],
            ]

            block_xyz = np.asarray(
                [
                    [x1, x2, x1, x2, x1, x2, x1, x2],
                    [y1, y1, y2, y2, y1, y1, y2, y2],
                    [z1, z1, z1, z1, z2, z2, z2, z2],
                ]
            )

            theta = (450.0 - np.asarray(self.azimuth)) % 360.0
            phi = -self.dip
            xyz = rotate_xyz(block_xyz.T, self.center, theta, phi)

            if self.reference == "top":
                offset = np.mean(xyz[4:, :], axis=0) - self._center
                self._center = self._center - offset
                xyz -= offset

            self._vertices = xyz

        return self._vertices
