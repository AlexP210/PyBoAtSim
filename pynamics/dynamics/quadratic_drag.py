import typing

import numpy as np
import trimesh

from pynamics.dynamics import BodyDynamicsParent
from pynamics import math
from pynamics.constants import EPSILON
from pynamics.kinematics.topology import Topology


class QuadraticDrag(BodyDynamicsParent):
    def __init__(
        self,
        drag_models: typing.Dict[str, trimesh.Trimesh],
        direction: np.matrix = np.matrix([0, 0, 1]).T,
        surface_point: np.matrix = np.matrix([0, 0, 0]).T,
        fluid_density: float = 1000,
        water_velocity: np.matrix = np.matrix([0, 0, 0]).T,
        drag_coefficient: float = 1.28,
    ):
        """Initialize quadratic drag on a set of bodies.

        Args:
            drag_models (typing.Dict[str, trimesh.Trimesh]): Dictionary mapping \
            body name to a `trimesh.Trimesh` object representing the volume to \
            use for drag calculation.
            direction (np.matrix, optional): Unit vector representing direction \
            in which bodies "float". Defaults to [0,0,1].
            surface_point (np.matrix, optional): Point on the surface of the fluid \
            Defaults to [0,0,0].
            fluid_density (float, optional): Density of the fluid. Defaults to 1000.
            water_velocity (np.matrix, optional): Velocity of the water current \
            in the (World, Identity) frame. Defaults to [0,0,0].
            drag_coefficient (float, optional): Drag coefficient for each \
            surface. Defaults to 1.28.

        Raises:
            ValueError: If the provided `drag_models` are not watertight.
        """
        super().__init__(body_names=drag_models.keys())
        self.direction = direction
        self.surface_point = surface_point
        self.fluid_density = fluid_density
        self.drag_models = drag_models
        self.drag_coefficient = drag_coefficient
        self.water_linear_velocity = water_velocity
        for name, model in self.drag_models.items():
            if not model.is_watertight:
                raise ValueError(f"Drag volume mesh for {name} is not watertight.")

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:
        # Copy of the buoyancy model to move around
        drag_model_temp: trimesh.Trimesh = self.drag_models[body_name].copy()
        drag_model_temp.apply_transform(
            matrix=np.array(
                topology.get_transform("World", "Identity", body_name, "Identity")
            )
        )

        plane_origin = self.surface_point.T.tolist()[0]
        plane_normal = (-self.direction.T).tolist()[0]
        # Get the submerged section
        submerged: trimesh.Trimesh = drag_model_temp.slice_plane(
            plane_origin=plane_origin, plane_normal=plane_normal, cap=True
        )

        forces_and_points_of_application = []
        if submerged.is_empty:
            return forces_and_points_of_application
        submerged.merge_vertices()
        submerged.fix_normals()
        body_velocity = (
            topology.get_X(body_name, "Identity", "World", "Identity")
            @ topology.bodies[body_name].get_velocity()
        )
        # Calculate drag
        for face, normal in zip(submerged.faces, submerged.face_normals):

            # Get some useful quantities
            normal = np.matrix(normal).T
            vertices = [submerged.vertices[face[i]] for i in range(len(face))]
            point_of_application = np.matrix((1 / 3) * sum(vertices)).T

            triangle_linear_velocity = (
                body_velocity[3:]
                + math.linalg.R3_cross_product_matrix(body_velocity[:3])
                @ point_of_application
            )

            # Get the relative velocity in the COM Frame
            v = self.water_linear_velocity - triangle_linear_velocity
            if np.linalg.norm(v) < EPSILON:
                v_hat = np.matrix([0, 0, 0]).T
            else:
                v_hat = v / np.linalg.norm(v)

            # Check the normals
            if np.linalg.norm(normal) < EPSILON:
                normal = np.matrix([0, 0, 0]).T
            else:
                normal = normal / np.linalg.norm(normal)
            # Normals are oriented outwards. So if the drag force is not
            # directed into the face, no drag contributed by this face
            if np.dot(v_hat.T, normal) >= 0:
                continue

            # Calculate the area perpendicular to the velocity
            A = 0.5 * np.linalg.norm(
                np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
            )
            A_perp = A * np.dot(-normal.T, v_hat)[0, 0]

            # Calculate the force magnitude
            v_squared = np.dot(v.T, v)[0, 0]
            force_magnitude = (
                0.5 * self.fluid_density * self.drag_coefficient * A_perp * v_squared
            )
            force = force_magnitude * v_hat
            forces_and_points_of_application.append((force, point_of_application))

        return forces_and_points_of_application
