import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

from pyboatsim.constants import HOME, AXES
from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, SimpleBodyDrag, ConstantForce, MeshBuoyancy, MeshGravity, MeshBodyDrag
from pyboatsim.math import linalg
from pyboatsim.topology import Topology, Frame, Body

class BoAtSim:
    def __init__(
            self,
            state: State,
            dynamics: typing.List[DynamicsParent]
        ) -> None:
        """
        Initializer
        """
        self.state = state
        self.history = []
        self.dynamics = dynamics
        self.dynamics_names = [
            dynamics_module.name 
            for dynamics_module in self.dynamics
        ]
        self.required_labels = [
            f"r_{axis}__boat" for axis in AXES
        ] + [
            f"v_{axis}__boat" for axis in AXES
        ] + [
            f"a_{axis}__boat" for axis in AXES
        ] + [
            f"theta_{axis}__boat" for axis in AXES
        ] + [
            f"omega_{axis}__boat" for axis in AXES
        ] + [
            f"alpha_{axis}__boat" for axis in AXES
        ] + [
            f"m__boat"
        ] + [
            f"I_{axis}{axis}__boat" for axis in AXES
        ] + [
            f"c_{axis}__boat" for axis in AXES
        ] + [
            "t"
        ]

    def _compute_accelerations(self):
        # Force and Moment
        F = np.matrix([self.state[f"f_{axis}__total"] for axis in AXES]).T
        T = np.matrix([self.state[f"tau_{axis}__total"] for axis in AXES]).T
        FT = np.block([
            [F,],
            [T,],
        ])
        # Angular velocity
        w = np.matrix([self.state[f"omega_{axis}__boat"] for axis in AXES]).T
        w_x = linalg.cross_product_matrix(w)
        # Mass properties
        I_cm = np.array([
            [self.state["I_xx__boat"], self.state["I_xy__boat"],  self.state["I_xz__boat"]],
            [self.state["I_yx__boat"], self.state["I_yy__boat"],  self.state["I_yz__boat"]],
            [self.state["I_zx__boat"], self.state["I_zy__boat"],  self.state["I_zz__boat"]],
        ])
        m = self.state["m__boat"]
        c = np.matrix([self.state[f"c_{axis}__boat"] for axis in AXES]).T
        c_x = linalg.cross_product_matrix(c)
        # Identity matrix
        I3 = np.eye(3)

        # Assemble the matrices from https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations#Any_reference_frame
        # A = np.block([
        #     [m*I3, -m*c_x],
        #     [m*c_x, I_cm - m*c_x@c_x]
        # ])
        A = np.block([
            [m*I3, np.zeros((3,3))],
            [np.zeros((3,3)), I_cm]
        ])
        # B = np.block([
        #     [m*w_x@w_x@c,],
        #     [w_x@(I_cm - m*c_x@c_x)@w]
        # ])
        B = np.block([
            [np.zeros((3,1)),],
            [w_x@I_cm@w,],
        ])

        accelerations = np.linalg.inv(A)*(FT - B)
        
        for idx, axis in enumerate(AXES):
            self.state[f"a_{axis}__boat"] = accelerations[idx, 0]
            self.state[f"alpha_{axis}__boat"] = accelerations[3+idx, 0]

        return 

    def step(self, dt):
        """
        Steps the simulation by `self._state["dt"]`.
        """
        # Apply the dynamics on the state, adds forces, torques, and other
        # intermediate values calculated by dynamics modules based on the
        # current state.
        for dynamics_module in self.dynamics:
            self.state = dynamics_module(self.state, dt)
        for axis in AXES:
            # Calculate the total force & moment by adding all the "f_"  and
            # "tau_" labels in the state dictionary
            self.state.set(
                partial_state_dictionary={
                    f"f_{axis}__total": sum([
                        self.state[f"f_{axis}__{name}"] for name in self.dynamics_names
                    ]),
                    f"tau_{axis}__total": sum([
                        self.state[f"tau_{axis}__{name}"] for name in self.dynamics_names
                    ])
                }
            )

        # Solve Newton-Euler equations to calculate the linear and angular accelerations
        # in the state dictionary
        self._compute_accelerations()
        
        # Add the state to the history
        self.history.append(self.state.get())
        # Create a new state to store the next update
        next_state = self.state.copy()

        # For each axis, update the position and velocities to be used in the
        # next state
        if len(self.history) < 1: raise ValueError("Why is self.history empty? For shame.")
        for axis in AXES:
            if len(self.history) == 1:
                # Update the positions
                next_state[f"r_{axis}__boat"] += self.state[f"v_{axis}__boat"]*dt + 0.5*next_state[f"a_{axis}__boat"]*dt**2
                next_state[f"theta_{axis}__boat"] += self.state[f"omega_{axis}__boat"]*dt + 0.5*next_state[f"alpha_{axis}__boat"]*dt**2
                next_state[f"v_{axis}__boat"] += self.state[f"a_{axis}__boat"] * dt
                next_state[f"omega_{axis}__boat"] += self.state[f"alpha_{axis}__boat"] * dt
            else:
                # Update the positions
                x_prev = self.history[-2]
                lin = f"r_{axis}__boat"
                ang = f"theta_{axis}__boat"
                next_state[lin] = 2*self.state[lin] - x_prev[lin] + self.state[f"a_{axis}__boat"]*dt**2
                next_state[ang] = 2*self.state[ang] - x_prev[ang] + self.state[f"alpha_{axis}__boat"]*dt**2
                next_state[f"v_{axis}__boat"] = (next_state[lin] - x_prev[lin]) / (2*dt)
                next_state[f"omega_{axis}__boat"] = (next_state[ang] - x_prev[ang]) / (2*dt)


        # Set the next state
        self.state = next_state
        self.state["t"] += dt

    def simulate(self, delta_t:float, dt:float, verbose=False):
        """
        Runs the simulation for delta_t more seconds.
        """
        # Ensure that the state contains the basic required labels to run
        # a simulation
        missing_labels = [
            label 
            for label in self.required_labels if not label in self.state._state_dictionary
        ]
        if len(missing_labels) != 0:
            raise ValueError(
                f"Cannot compute dynamics, missing the following"
                f" labels: {', '.join(missing_labels)}"
            )
        if verbose:
            for _ in tqdm.tqdm(range(int(delta_t//dt+1))):
                self.step(dt=dt)
        else:
            for _ in range(int(delta_t//dt+1)):
                self.step(dt=dt)
    
    def save_history(self, file_path:str):
        self.get_history_as_dataframe().to_csv(file_path)

    def get_history_as_dataframe(self):
        return pd.DataFrame.from_dict(self.history)
