import os

import numpy as np
import trimesh

from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag
import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
from pynamics.boatsim import Sim
from pynamics.visualizer import Visualizer

if __name__ == "__main__":

    body = topo.Body(
        mass_properties_model=trimesh.load(
                file_obj=os.path.join("models", "common", "cylinder.obj"), 
                file_type="obj", 
                force="mesh"),
        density=0.8*997
    )

    water_world = topo.Topology()
    water_world.add_connection("World", "Identity", body.copy(), "Buoy", joint.FreeJoint())
    water_world.joints["Buoy"].set_configuration(np.matrix([np.pi/4,0,0 ,1, 1, -4]).T)
    # water_world.joints["Buoy"].set_configuration(np.matrix([0, 0, 2]).T)
    water_world_vis = Visualizer(
        topology=water_world,
        visualization_models={
            (f"Buoy", "Identity"): trimesh.load(
                file_obj=os.path.join("models", "common", "cylinder.obj"), 
                file_type="obj", 
                force="mesh"),
        }
    )

    water_world_sim = Sim(
        topology=water_world,
        body_dynamics={
            "gravity": Gravity(-9.81, 2),
            "buoyancy": Buoyancy(
                buoyancy_models={
                    "Buoy": trimesh.load(
                        file_obj=os.path.join("models", "common", "cylinder.obj"), 
                        file_type="obj", 
                        force="mesh" 
                    )
                },
            ),
            "drag": QuadraticDrag(
                drag_models={
                    "Buoy": trimesh.load(
                        file_obj=os.path.join("models", "common", "cylinder.obj"), 
                        file_type="obj", 
                        force="mesh" 
                    )
                },
            )
        },
    )

    water_world_sim.simulate(delta_t=10, dt=0.01, verbose=True)
    water_world_sim.save_data("Buoyancy_Test.csv")
    water_world_vis.add_sim_data(water_world_sim)
    water_world_vis.animate(
        save_path="Buoyancy_Test.mp4", 
        verbose=True)

