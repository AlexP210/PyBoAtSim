import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

from constants import HOME, AXES
from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, BodyDrag, ConstantForce

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
            "t"
        ]
              
    def step(self, dt):
        """
        Steps the simulation by `self._state["dt"]` using forward euler.
        """
        # Apply the dynamics on the state
        for dynamics_module in self.dynamics:
            self.state = dynamics_module(self.state)

        # For each axis, apply the dynamics & update the state
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
            # Calculate the linear & angular acceleration from the total
            # force and torque (TODO: Generalize this beyond principal co-
            # ordinate system)
            self.state.set(
                partial_state_dictionary={
                    f"a_{axis}__boat": self.state[f"f_{axis}__total"] / self.state["m__boat"],
                    f"alpha_{axis}__boat": self.state[f"tau_{axis}__total"] / self.state[f"I_{axis}{axis}__boat"]
                }
            )
        
        # Add the state to the history
        self.history.append(self.state.get())
        
        # Create a new state to store the next update
        next_state = self.state.copy()

        # For each axis, update the position and velocities to be used in the
        # next state
        for axis in AXES:
            # Update the velocities
            next_state[f"v_{axis}__boat"] += self.state[f"a_{axis}__boat"]*dt
            next_state[f"omega_{axis}__boat"] += self.state[f"alpha_{axis}__boat"]*dt
            # Update the positions
            next_state[f"r_{axis}__boat"] += 0.5*(self.state[f"v_{axis}__boat"] + next_state[f"v_{axis}__boat"])*dt
            next_state[f"theta_{axis}__boat"] += 0.5*(self.state[f"omega_{axis}__boat"] + next_state[f"omega_{axis}__boat"])*dt
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
        pd.DataFrame.from_dict(self.history).to_csv(file_path)

if __name__ == "__main__":

    # Assemble the sim
    sim = BoAtSim(
        state=State(
            state_dictionary={
            "t": 0,
            "r_x__boat": 0, 
            "r_y__boat": 0,
            "r_z__boat": 0,
            "v_x__boat": 0,
            "v_y__boat": 0, 
            "v_z__boat": 0,
            "a_x__boat": 0, 
            "a_y__boat": 0, 
            "a_z__boat": 0, 
            "theta_x__boat": 0, 
            "theta_y__boat": 0, 
            "theta_z__boat": 0, 
            "omega_x__boat": 0, 
            "omega_y__boat": 0, 
            "omega_z__boat": 0,
            "alpha_x__boat": 0, 
            "alpha_y__boat": 0, 
            "alpha_z__boat": 0,
            "m__boat": 1,
            "I_xx__boat": 1,
            "I_yy__boat": 1,
            "I_zz__boat": 1,
            "rho": 1000,
            "v_x__water": 0,
            "v_y__water": 0, 
            "v_z__water": 0,
            "gamma__waterwheel": 0,
            "gammadot__waterwheel": 0.01,
        }), 
        dynamics=[
            WaterWheel("waterwheel", 1, 1, 0.1, 2, 1, 1),
            BodyDrag("bodydrag", 1, 1),
        ]
    )

    # Run the sim
    sim.simulate(delta_t=3, dt=0.001, verbose=True)
    data = pd.DataFrame.from_dict(sim.history)

    # Plot the results
    plt.plot(data["t"], data["f_x__waterwheel"], label="f_x__waterwheel")
    plt.plot(data["t"], data["f_x__bodydrag"], label="f_x__bodydrag")
    plt.plot(data["t"], data["f_x__total"], label="f_x__total")
    plt.title("Forces During Basic Bo-At Sim")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.show()