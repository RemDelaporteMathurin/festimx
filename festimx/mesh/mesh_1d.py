from dolfinx import fem, mesh
from mpi4py import MPI
import ufl
from festimx import Mesh

import numpy as np


class Mesh1D(Mesh):
    """
    1D Mesh
    Attributes:
        vertices (list): the mesh x-coordinates
        size (float): the size of the 1D mesh
        V (dolfinx.fem.FunctionSpace): the function space of the simulation
    """
    def __init__(self, vertices, **kwargs) -> None:
        """Inits Mesh1D

        Args:
            vertices (list): the mesh x-coordinates
        """

        self.vertices = vertices

        self.start = min(vertices)
        self.size = max(vertices)
        self.V = None

        mesh = self.generate_mesh()
        super().__init__(mesh=mesh, **kwargs)


    def generate_mesh(self):
        '''Generates a 1D mesh
        '''
        gdim, shape, degree = 1, "interval", 1
        cell = ufl.Cell(shape, geometric_dimension=gdim)
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

        x = np.array(self.vertices)
        indexes = np.arange(x.shape[0])
        cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
        return mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    def define_surface_markers(self):
        """Creates the surface markers
        Returns:
            dolfinx.MeshTags: the tags containing the surface
                markers
        """
        dofs_L = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], self.start))
        dofs_R = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], self.size))

        dofs_facets = np.array([dofs_L[0], dofs_R[0]])
        tags_facets = np.array([1, 2])

        mesh_tags_facets = mesh.meshtags(self.mesh, 0, dofs_facets, tags_facets) 

        return mesh_tags_facets

    def define_volume_markers(self):
        """Creates the volume markers
        Returns:
            dolfinx.MeshTags: the tags containing the volume
                markers
        """
        dofs_cells = np.array([])
        tags_cells = np.array([])
        for subdomain in self.subdomains.subdomains:
            borders = subdomain.borders
            dofs_subdomain = fem.locate_dofs_geometrical(
                self.V, lambda x: np.logical_and(x[0] >= borders[0], x[0] <= borders[1]))
            dofs_cells = np.concatenate((dofs_cells, dofs_subdomain))
            tags_cells = np.concatenate((tags_cells, subdomain.id*np.ones_like(dofs_subdomain)))

        # print(dofs_cells)
        # print(tags_cells)

        mesh_tags_cells = mesh.meshtags(self.mesh, 1, dofs_cells, tags_cells)

        return mesh_tags_cells

    def define_measures(self):
        """Creates the fenics.Measure objects for self.dx and self.ds
        """
        if self.subdomains.subdomains[0].borders is not None:
            self.subdomains.check_borders(self.size)
        self.define_markers()
        super().define_measures()
