from dolfinx import fem, nls, log
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl

from festimx import MeshXDMF

my_mesh = MeshXDMF("mesh_mb/converted/mesh_domains.xdmf", "mesh_mb/converted/mesh_boundaries.xdmf", subdomains=[])

my_mesh.define_markers()
my_mesh.define_measures()

ft = my_mesh.surface_markers
ct = my_mesh.volume_markers
# prepare output file
xdmf_file = XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w")
xdmf_file.write_mesh(my_mesh.mesh)

# define FunctionSpace and functions
V = fem.FunctionSpace(my_mesh.mesh, ("CG", 1))
u = fem.Function(V)
u_n = fem.Function(V)
v = ufl.TestFunction(V)

# Dirichlet boundary conditions
top_tag = 10
water_tag = 15
water_dofs = fem.locate_dofs_topological(V, my_mesh.mesh.topology.dim-1, ft.indices[ft.values == water_tag])
bc_water = fem.dirichletbc(fem.Constant(my_mesh.mesh, 0.), water_dofs, V)

bcs = [bc_water]

# flux BC
class FluxTop:
    def __init__(self):
        self.t = 0
    
    def eval(self, x):
        return 1 + self.t + x[0]*0
flux_top_expr = FluxTop()

flux_top = fem.Function(V)
flux_top.interpolate(flux_top_expr.eval)

# Weak formulation

dt = fem.Constant(my_mesh.mesh, 1.)
F = (u-u_n)/dt*v*my_mesh.dx
F += 1*ufl.dot(ufl.grad(u), ufl.grad(v)) * my_mesh.dx(6)
F += 10*ufl.dot(ufl.grad(u), ufl.grad(v)) * my_mesh.dx(7)
F += 10*ufl.dot(ufl.grad(u), ufl.grad(v)) * my_mesh.dx(8)

F += -flux_top * v * my_mesh.ds(top_tag)

# iterate
t = 0
log.set_log_level(log.LogLevel.INFO)
while t < 30:
    print(f"Current time: {t:.1f}")
    t += dt.value

    # update BC
    flux_top_expr.t = t
    flux_top.interpolate(flux_top_expr.eval)

    # create NonLinearProblem and solve
    problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs)
    solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.solve(u)

    # export
    xdmf_file.write_function(u, t=t)

    # update previous solution
    u_n.x.array[:] = u.x.array[:]
