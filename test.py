from dolfinx import mesh, fem, nls
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

xdmf_file = XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w")
# write the mesh once
xdmf_file.write_mesh(domain)

V = fem.FunctionSpace(domain, ("CG", 1))
uD = fem.Function(V)

class CustomExpression:
    def __init__(self) -> None:
        self.t = 0
    
    def eval(self, x):
        return self.t + 1 + x[0]**2 + 2 * x[1]**2

my_expr = CustomExpression()

# THIS REPLACES MeshFunctions
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = np.flatnonzero(mesh.compute_boundary_facets(domain.topology))


boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
# bc = fem.dirichletbc(fem.Constant(domain, 0.), boundary_dofs, V)

uD.interpolate(my_expr.eval)
bc = fem.dirichletbc(uD, boundary_dofs)

u = fem.Function(V, name="function_u")
v = ufl.TestFunction(V)

f = fem.Constant(domain, -6.)

F = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
F += -f * v * ufl.dx

# iterate
for t in range(10):
    # update 
    my_expr.t = t
    uD.interpolate(my_expr.eval)

    problem = fem.petsc.NonlinearProblem(F, u, bcs=[bc])
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.solve(u)

    xdmf_file.write_function(u, t=t)
