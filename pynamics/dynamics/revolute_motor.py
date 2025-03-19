import typing

import numpy as np

from pynamics.dynamics import JointDynamicsParent
from pynamics.kinematics.topology import Topology


class RevoluteDCMotor(JointDynamicsParent):
    def __init__(
        self,
        joint_name: str,
        electromotive_constant: float,
        resistance: float,
        inductance: float,
        voltage: float,
        initial_current: float = 0,
    ):
        """Initialize a DC motor on a revolute joint.

        Args:
            joint_name (str): Name of the joint on which to place the motor.
            electromotive_constant (float): Constant relating the motor current \
            to motor torque.
            resistance (float): Armature resistance of the motor.
            inductance (float): Inductance of the motor coil.
            voltage (float): Potential difference applied across the `resistance` \
            and `inductance`
            initial_current (float, optional): Initial state of the current. \
            Defaults to 0.
        """
        super().__init__(
            joint_names=[
                joint_name,
            ]
        )
        self.electromotive_constant = electromotive_constant
        self.resistance = resistance
        self.inductance = inductance
        self.voltage = voltage
        self.current = initial_current
        self.emf = 0

    def compute_dynamics(
        self, topology: Topology, joint_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:
        joint = topology.joints[joint_name]
        S = joint.get_motion_subspace()
        if not (sum(S[:3, 0]) == 1 and sum(S[3:, 0]) == 0):
            raise ValueError(
                f'RevoluteMotor "{self.name}" cannot operate on non-revolute joint "{joint_name}".'
            )
        return [
            self.electromotive_constant
            * self.current
            * np.matrix(np.ones(shape=joint.get_configuration().shape)),
        ]

    def update(self, topology: Topology, dt: float):
        joint = topology.joints[self.joint_names[0]]
        self.emf = self.electromotive_constant * joint.get_configuration_d()[0, 0]
        di_dt = (
            (self.voltage - self.emf) - self.resistance * self.current
        ) / self.inductance
        self.current += di_dt * dt

    def get_data(self):
        return {
            "Current": self.current,
            "EMF": self.emf,
            "Voltage": self.voltage,
        }
