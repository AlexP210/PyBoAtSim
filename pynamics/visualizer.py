"""
This module contains the definition of `Visualizer`, the Pynamics module
for visualizing simulations.
"""

import typing

import trimesh
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from pynamics.sim import Sim
from pynamics.kinematics.topology import Topology


class Arrow3D(FancyArrowPatch):
    """
    Class for 3D arrows in Matplotlib figures. Not yet being used.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class Visualizer:
    """
    Class for visualizing sims.
    """

    def __init__(
        self,
        topology: Topology,
        visualization_models: typing.Dict[typing.Tuple[str, str], trimesh.Trimesh],
        sim: Sim = None,
    ):
        """Initialize a Visualizer for a gien simulation.

        Args:
            topology (Topology): The `Topology` to visualize.
            visualization_models (typing.Dict[typing.Tuple[str, str], trimesh.Trimesh]): \
            A dictionary mapping (Body Name, Frame Name) tuples to `trimesh.Trimesh` objects \
            to use for visualization.
            sim (Sim, optional): The `Sim` object to initialize the Visualizer \
            with. Defaults to None.
        """
        self.topology = topology
        self.visualization_models = visualization_models
        self.visualization_artists = {}
        self.lower_bound = None
        self.upper_bound = None
        if sim is not None:
            self.add_sim_data(sim=sim)

    def add_sim_data(self, sim: Sim):
        """Add simulation data to the visualizer.

        Args:
            sim (Sim): `Sim` object whose `data_history` will be visualized.
        """
        self.sim = sim
        self.lower_bound = np.zeros(shape=(3,))
        self.upper_bound = np.zeros(shape=(3,))
        for joint_configuration in self.sim.joint_space_position_history:
            lb, ub = self._get_bounds(joint_configuration)
            for i in range(3):
                self.lower_bound[i] = min(self.lower_bound[i], lb[i])
                self.upper_bound[i] = max(self.upper_bound[i], ub[i])

    def _get_bounds(self, joint_configuration=None):
        """
        Helper method to step over the simulation and find the minimum/maximum
        bounds of the volume needed to view the simulation.
        """
        lower_bound = np.zeros(shape=(3,))
        upper_bound = np.zeros(shape=(3,))

        # Configure the topology based on the provided joint configuration
        # or if None, the default configuration the topology is already in
        if joint_configuration is not None:
            for body_name, q in joint_configuration.items():
                self.topology.joints[body_name].set_configuration(q)
        # For each body, find the minimum and maximum bounds in this time step
        for (
            body_name,
            frame_name,
        ), visualization_model in self.visualization_models.items():
            visualization_model_temp = visualization_model.copy()
            visualization_model_temp.apply_transform(
                np.array(
                    self.topology.get_transform(
                        from_body_name="World",
                        from_frame_name="Identity",
                        to_body_name=body_name,
                        to_frame_name=frame_name,
                    )
                )
            )

            vertices = visualization_model_temp.vertices
            lb = vertices.min(axis=0)
            ub = vertices.max(axis=0)
            for i in range(3):
                lower_bound[i] = min(lower_bound[i], lb[i])
                upper_bound[i] = max(upper_bound[i], ub[i])

        return lower_bound, upper_bound

    def _trim_axes(self, axes, lower_bound, upper_bound):
        center = 0.5 * (upper_bound + lower_bound)
        max_extent = 0
        for i in range(3):
            max_extent = max(max_extent, 0.5 * (upper_bound[i] - lower_bound[i]))
        max_extent *= 1.1
        axes.set_xlim(center[0] - max_extent, center[0] + max_extent)
        axes.set_ylim(center[1] - max_extent, center[1] + max_extent)
        axes.set_zlim(center[2] - max_extent, center[2] + max_extent)
        return axes

    def _plot(self, axes: plt.Axes, joint_configuration=None):
        """
        Helper method to put the
        """
        # Configure the topology based on the provided joint configuration
        if joint_configuration is not None:
            for body_name, q in joint_configuration.items():
                self.topology.joints[body_name].set_configuration(q)

        # Go through the bodies, use their transforms to place the visualization
        # models and get the vertices and triangles.
        artists = []
        for (
            body_name,
            frame_name,
        ), visualization_model in self.visualization_models.items():
            visualization_model_temp = visualization_model.copy()
            visualization_model_temp.apply_transform(
                np.array(
                    self.topology.get_transform(
                        from_body_name="World",
                        from_frame_name="Identity",
                        to_body_name=body_name,
                        to_frame_name=frame_name,
                    )
                )
            )

            vertices = visualization_model_temp.vertices
            T = visualization_model_temp.faces

            artist = axes.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=T,
                edgecolor=[[0, 0, 0]],
                linewidth=1.0,
                alpha=0.5,
                shade=True,
                color="k",
                zorder=1,
            )
            self.visualization_artists[body_name] = artist
            artists.append(artist)

        axes.set_xlabel("X (m)")
        axes.set_ylabel("Y (m)")
        axes.set_zlabel("Z (m)")

        return axes, artists

    def _update(self, step_idx: int, axes: plt.Axes):
        # Get the state for this time
        q = self.sim.joint_space_position_history[step_idx]

        for body_name, artist in self.visualization_artists.items():
            artist.remove()

        axes, artists = self._plot(axes, joint_configuration=q)

        return artists

    def view(self, save_path: str = None, figsize=(12, 10)):
        """View the current state of the `Topology`.

        Args:
            save_path (str, optional): Path at which to save the image of the \
            `Topology`. Defaults to None.
            figsize (tuple, optional): Size of the matplotlib.Figure instance. \
            Defaults to (12, 10).
        """
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection="3d")

        # Trim te axes we're plotting on so it's not too big
        axes = self._trim_axes(axes, *self._get_bounds())

        # Plot the topology in its current config on the axes.
        axes, artists = self._plot(axes)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)
            plt.clf()
            plt.cla()
            plt.close()

    def animate(self, save_path: str = None, verbose: bool = False, figsize=(12, 10)):
        """Animate the `data_history` of the attached `Sim` object.

        Args:
            save_path (str, optional): Path at which to save the sim animation. \
            Defaults to None.
            verbose (bool, optional): Whether to display a progress bar. \
            Defaults to False.
            figsize (tuple, optional): Size of the matplotlib.Figure instance. \
            Defaults to (12, 10).
        """
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection="3d")
        # Trim te axes we're plotting on so it's not too big
        # add_sim_data already computed the bounds
        axes = self._trim_axes(axes, self.lower_bound, self.upper_bound)

        dt = self.sim.data_history["Time"][1] - self.sim.data_history["Time"][0]
        ani = animation.FuncAnimation(
            fig=fig,
            func=lambda i, axes: self._update(i, axes),
            frames=len(self.sim.joint_space_position_history),
            interval=dt * 1000,
            fargs=(axes,),
        )

        if save_path is None:
            plt.show()
        else:
            if verbose:
                with tqdm.tqdm(
                    total=len(self.sim.joint_space_position_history)
                ) as progress_bar:
                    ani.save(
                        save_path, progress_callback=lambda i, n: progress_bar.update()
                    )
                    plt.clf()
                    plt.cla()
                    plt.close()
            else:
                ani.save(save_path)
                plt.clf()
                plt.cla()
                plt.close()
