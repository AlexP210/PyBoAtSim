import typing

import numpy as np

from pynamics.dynamics import JointDynamicsParent
from pynamics.kinematics.topology import Topology


class JointDamping(JointDynamicsParent):

    def __init__(self, damping_factor: float, joint_names: typing.List[str] = []):
        """Initialize damping on a set of joints.

        Args:
            damping_factor (float): The factor relating the joint velocity to \
            damping force.
            joint_names (typing.List[str], optional): List of joint names on \
            which to apply damping. Defaults to [].
        """
        super().__init__(joint_names=joint_names)
        self.damping_factor = damping_factor

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.List[np.matrix]:
        return [
            -self.damping_factor * topology.joints[body_name].get_configuration_d(),
        ]
