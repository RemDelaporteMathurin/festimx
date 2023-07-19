from dolfinx import mesh, fem, nls, plot
from mpi4py import MPI
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from ufl import dx, grad, dot, jump, avg
import numpy as np

msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

V = fem.FunctionSpace(msh, ("DG", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: np.full(x[0].shape, 0.0))

# create mesh tags
def marker_interface(x):
    return np.isclose(x[0], 0.5)

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
facet_imap = msh.topology.index_map(tdim - 1)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
interface_facets = mesh.locate_entities_boundary(msh, tdim - 1, marker_interface)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
indices = np.arange(0, num_facets)
values = np.arange(0, num_facets, dtype=np.intc)

values[boundary_facets] = 1
values[interface_facets] = 2

mesh_tags_facets = mesh.meshtags(msh, tdim - 1, indices, values) 

ds = ufl.Measure("ds", domain=msh, subdomain_data=mesh_tags_facets)
dS = ufl.Measure("dS", domain=msh, subdomain_data=mesh_tags_facets)

u = fem.Function(V)
u_n = fem.Function(V)
v = ufl.TestFunction(V)

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

# Define parameters
alpha = 10
gamma = 10

# Simulation constants
f = fem.Constant(msh, PETSc.ScalarType(2.0))

# Define variational problem

F = 0

# diffusion
F += dot(grad(v), grad(u))*dx - dot(v*n, grad(u))*ds \
   - dot(avg(grad(v)), jump(u, n))*dS - dot(jump(v, n), avg(grad(u)))*dS \
   + gamma/avg(h)*dot(jump(v, n), jump(u, n))*dS

# source
F += -v*f*dx 

# Dirichlet BC
F += - dot(grad(v), u*n)*ds + alpha/h*v*u*ds\
   + uD*dot(grad(v), n)*ds - alpha/h*uD*v*ds


problem = fem.petsc.NonlinearProblem(F, u)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.solve(u)


xdmf_file = XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w")
xdmf_file.write_mesh(msh)
xdmf_file.write_function(u)


import pyvista
pyvista.OFF_SCREEN = True

pyvista.start_xvfb()

u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = u.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=False)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = u_plotter.screenshot("DG.png")