import typing

import numpy as np

from pynamics.dynamics import BodyDynamicsParent
from pynamics.kinematics.topology import Topology


class Spring(BodyDynamicsParent):
    def __init__(
        self,
        body1: str,
        frame1: str,
        body2: str,
        frame2: str,
        stiffness: float,
    ):
        """Initialize a spring between two bodies.

        Args:
            body1 (str): The first body to which the spring is attached.
            frame1 (str): The frame on `body1` to which the spring is attached.
            body2 (str): The second body to which the spring is attached.
            frame2 (str): The frame on `body2` to which the spring is attached.
            stiffness (float): The stiffness of the spring.
        """
        super().__init__(body_names=[body1, body2])
        self.stiffness = stiffness
        self.body1 = body1
        self.frame1 = frame1
        self.body2 = body2
        self.frame2 = frame2

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:
        this_body, this_frame = ((self.body2, self.frame2), (self.body1, self.frame1))[
            body_name == self.body1
        ]
        that_body, that_frame = ((self.body2, self.frame2), (self.body1, self.frame1))[
            body_name == self.body2
        ]
        r_this = topology.get_transform(
            from_body_name="World",
            from_frame_name="Identity",
            to_body_name=this_body,
            to_frame_name=this_frame,
        )[:3, 3]
        r_that = topology.get_transform(
            from_body_name="World",
            from_frame_name="Identity",
            to_body_name=that_body,
            to_frame_name=that_frame,
        )[:3, 3]

        r_this_that = r_that - r_this
        force = self.stiffness * r_this_that
        point_of_application = topology.get_transform(
            "World", "Identity", this_body, this_frame
        )[:3, 3]
        return [
            (force, point_of_application),
        ]
