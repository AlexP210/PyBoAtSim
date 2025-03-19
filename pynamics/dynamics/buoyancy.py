import typing

import numpy as np
import trimesh

from pynamics.dynamics import BodyDynamicsParent
from pynamics.kinematics.topology import Topology


class Buoyancy(BodyDynamicsParent):
    def __init__(
        self,
        buoyancy_models: typing.Dict[str, trimesh.Trimesh],
        direction: np.matrix = np.matrix([0, 0, 1]).T,
        surface_point: np.matrix = np.matrix([0, 0, 0]).T,
        fluid_density: float = 1000,
        g: float = -9.81,
    ):
        """Initialize buoyancy on a set of bodies.

        Args:
            buoyancy_models (typing.Dict[str, trimesh.Trimesh]): Dictionary \
            mapping body name to a `trimesh.Trimesh` object representing the volume \
            to use on buoyancy calculations for that body.
            direction (np.matrix, optional): Unit vector representing direction \
            in which bodies "float". Defaults to [0,0,1].
            surface_point (np.matrix, optional): Point on the surface of the fluid \
            Defaults to [0,0,0].
            fluid_density (float, optional): Density of the fluid. Defaults to 1000.
            g (float, optional): Acceleration due to gravity. Defaults to -9.81.

        Raises:
            ValueError: If the provided `buoyancy_models` are not "watertight".
        """
        super().__init__(body_names=buoyancy_models.keys())
        self.direction = direction / np.linalg.norm(direction)
        self.surface_point = surface_point
        self.fluid_density = fluid_density
        self.buoyancy_models = buoyancy_models
        self.g = g
        for name, model in self.buoyancy_models.items():
            if not model.is_watertight:
                raise ValueError(f"Buoyant volume mesh for {name} is not watertight.")

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:

        # Copy of the buoyancy model to move around
        buoyancy_model_temp: trimesh.Trimesh = self.buoyancy_models[body_name].copy()
        buoyancy_model_temp.apply_transform(
            matrix=np.array(
                topology.get_transform("World", "Identity", body_name, "Identity")
            )
        )

        plane_origin = self.surface_point.T.tolist()[0]
        plane_normal = (-self.direction.T).tolist()[0]
        submerged: trimesh.Trimesh = buoyancy_model_temp.slice_plane(
            plane_origin=plane_origin, plane_normal=plane_normal, cap=True
        )
        force = np.matrix([0, 0, 0]).T
        point_of_application = np.matrix([0, 0, 0]).T
        if not submerged.is_empty:
            water_volume = submerged.volume
            water_mass = water_volume * self.fluid_density
            force = -water_mass * self.g * self.direction
            point_of_application = np.matrix(submerged.center_mass).T
        return [
            (force, point_of_application),
        ]
