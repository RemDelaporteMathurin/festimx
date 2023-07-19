from dolfinx import mesh, fem, nls, plot
from mpi4py import MPI
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc
from ufl import inner, dx, grad, dot, dS, jump, avg, ds
import numpy as np

msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

V = fem.FunctionSpace(msh, ("DG", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: np.full(x[0].shape, 0.0))


tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# bc = fem.dirichletbc(uD, boundary_dofs)


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

# transient term
# delta_t = fem.Constant(msh, PETSc.ScalarType(0.1))
# F += inner((u-u_n) / delta_t, v) * dx

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


import dolfinx.plot
import pyvista
pyvista.OFF_SCREEN = True

pyvista.start_xvfb()
# We create a mesh consisting of the degrees of freedom for visualization
topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(msh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

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