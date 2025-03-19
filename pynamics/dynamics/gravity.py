import typing

import numpy as np

from pynamics.dynamics import BodyDynamicsParent
from pynamics.kinematics.topology import Topology


class Gravity(BodyDynamicsParent):
    def __init__(
        self,
        g: float,
        direction: np.matrix = np.matrix([0, 0, 1]).T,
        body_names: typing.List[str] = [],
    ):
        """Initialize gravity on a set of bodies.

        Args:
            g (float): Acceleration due to gravity.
            direction (np.matrix): "Up" axis.
            body_names (typing.List[str], optional): List of body names on which \
            to apply gravity. Defaults to [].
        """
        super().__init__(body_names=body_names)
        self.g = g
        self.direction = direction / np.linalg.norm(direction)

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:
        force = self.g * self.direction * topology.bodies[body_name].mass

        point_of_application = topology.get_transform(
            "World", "Identity", body_name, "Center of Mass"
        )[:3, 3]
        return [
            (force, point_of_application),
        ]
