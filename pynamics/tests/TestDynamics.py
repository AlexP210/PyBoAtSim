import os
import unittest
import warnings

import numpy as np
import matplotlib.pyplot as plt
import trimesh

import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
import pynamics.dynamics as dynamics
from pynamics.sim import Sim
import pynamics.constants as const
from pynamics.visualizer import Visualizer


class TestDynamics(unittest.TestCase):

    def save_plot_artifact(self, time, value, expected_value, value_name, test_name):
        fig, ax = plt.subplots(1, 2)

        ax[0].scatter(time, value, s=5, label=f"Simulated {value_name}")
        ax[0].scatter(time, expected_value, s=5, label=f"Expected {value_name}")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel(value_name)
        ax[0].legend()

        ax[1].scatter(
            time, value - expected_value, c="k", s=5, label=f"{value_name} Error"
        )
        ax[1].hlines(
            y=[-const.EPSILON, const.EPSILON],
            xmin=time[0],
            xmax=time[-1],
            colors="r",
            linestyles="--",
            label="Tolerance",
        )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel(value_name)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(const.HOME, "tests", f"{test_name}.png"))
        plt.clf()
        plt.cla()
        plt.close()

    def make_free_cube_topology(self):
        cube = topo.Body(
            mass_properties_model=trimesh.load(
                file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                file_type="obj",
                force="mesh",
            ),
            density=990,  # A little bit less dense than water
        )
        corner = topo.Frame(translation=np.matrix([1, 1, 1]).T)
        cube.add_frame(corner, "Corner")

        topology = topo.Topology()
        topology.add_connection(
            parent_body_name="World",
            parent_frame_name="Identity",
            child_body=cube.copy(),
            child_body_name="Cube",
            joint=joint.FreeJoint(),
        )

        visualizer = Visualizer(
            topology=topology,
            visualization_models={
                ("Cube", "Identity"): trimesh.load(
                    file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                    file_type="obj",
                    force="mesh",
                )
            },
        )

        return topology, visualizer

    def test_buoyancy(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

        topology, visualizer = self.make_free_cube_topology()
        topology.joints["Cube"].set_configuration(np.matrix([1, 0, 0, 0, 0, 0, -500]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "buoyancy": dynamics.Buoyancy(
                    buoyancy_models={
                        "Cube": trimesh.load(
                            file_obj=os.path.join(
                                const.HOME, "models", "common", "Cube.obj"
                            ),
                            file_type="obj",
                            force="mesh",
                        )
                    },
                    fluid_density=1000,
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.05)
        visualizer.add_sim_data(sim)
        visualizer.animate(
            save_path=os.path.join(const.HOME, "tests", "test_buoyancy.mp4")
        )

        t = np.array(sim.data_history["Time"])
        x = sim.data_history["Cube / Position 6"]
        f = 9.81 * 1000 * (2**3)
        a = f / sim.topology.bodies["Cube"].mass
        x_expected = -500 + 0.5 * (a * t**2)
        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Z Position",
            test_name="test_buoyancy",
        )
        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Buoyancy simulates correctly.",
        )

    def test_gravity(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_free_cube_topology()
        # topology.joints["Cube"].set_configuration(np.matrix([1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),5,0,0]).T)
        topology.joints["Cube"].set_configuration(np.matrix([1, 0, 0, 0, 5, 0, 0]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "gravity": dynamics.Gravity(g=-9.81, body_names="Cube")
            },
        )
        sim.simulate(delta_t=5, dt=0.05)
        visualizer.add_sim_data(sim)
        visualizer.animate(
            save_path=os.path.join(const.HOME, "tests", "test_gravity.mp4")
        )

        t = np.array(sim.data_history["Time"])
        a = -9.81
        x_expected = 0.5 * a * (t**2)
        x = sim.data_history["Cube / Position 6"]

        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Z Position",
            test_name="test_gravity",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Gravity simulates correctly on a free body.",
        )

    def test_drag(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_free_cube_topology()
        topology.joints["Cube"].set_configuration_d(np.matrix([0, 0, 0, 0.1, 0, 0]).T)
        topology.joints["Cube"].set_configuration(np.matrix([1, 0, 0, 0, 0, 0, -10]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "drag": dynamics.QuadraticDrag(
                    drag_models={
                        "Cube": trimesh.load(
                            file_obj=os.path.join(
                                const.HOME, "models", "common", "Cube.obj"
                            ),
                            file_type="obj",
                            force="mesh",
                        )
                    },
                    drag_coefficient=1,
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.05)
        visualizer.add_sim_data(sim)
        visualizer.animate(save_path=os.path.join(const.HOME, "tests", "test_drag.mp4"))

        t = np.array(sim.data_history["Time"])
        # v_d = (-0.5 * density * C_d * A) / m * v^2
        # => v_d(t) = 1 / [(0.5*density*C_d*A)*t + c_1]
        # => c_1 = 1/v_d(0)
        m = sim.topology.bodies["Cube"].mass
        v_expected = 1 / (0.5 * 1000 * 1 * (2**2) / m * t + 1 / 0.1)
        v = sim.data_history["Cube / Velocity 3"]

        self.save_plot_artifact(
            time=t,
            value=v,
            expected_value=v_expected,
            value_name="X Velocity",
            test_name="test_drag",
        )

        self.assertTrue(
            (abs(v - v_expected) < const.EPSILON).all(),
            msg="Drag simulates correctly on a free body.",
        )

    def test_spring(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_free_cube_topology()
        topology.joints["Cube"].set_configuration_d(np.matrix([0, 0, 0, 0, 0, -0.1]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "spring": dynamics.Spring(
                    body1="World",
                    frame1="Identity",
                    body2="Cube",
                    frame2="Identity",
                    stiffness=topology.bodies["Cube"].mass,
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        visualizer.animate(
            save_path=os.path.join(const.HOME, "tests", "test_spring.mp4")
        )

        t = np.array(sim.data_history["Time"])
        # x_dd = -(k/m)*(|x|-l)
        # x(t) = A*sin(sqrt(k/m)*t) + B*cos(sqrt(k/m)*t) + l
        # x_d(t) = A*sqrt(k/m)*cos(sqrt(k/m)*t) - B*sqrt(k/m)*sin(sqrt(k/m)*t)
        # x_dd(t) = -A*(k/m)*sin(sqrt(k/m)*t) - B*(k/m)*cos(sqrt(k/m)*t)
        # x_dd(t) = -(k/m) * ( A*sin(sqrt(k/m)*t) + B*cos(sqrt(k/m)*t) ) = -(k/m)*x(t) - l
        # => x(0) = B + l => B = x(0) - l
        # => x_d(0) = A*sqrt(k/m) => A = x_d(0)/omega * x_d(0)
        m = sim.topology.bodies["Cube"].mass
        k = m
        omega = np.sqrt(k / m)
        x_expected = (-0.1 / omega) * np.sin(omega * t)
        x = sim.data_history["Cube / Position 6"]

        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Z Position",
            test_name="test_spring",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Spring simulates correctly on a free body.",
        )
