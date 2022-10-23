from dolfinx.io import XDMFFile
from mpi4py import MPI
from dolfinx import fem
from festimx import MeshXDMF


my_mesh = MeshXDMF("mesh_mb/mesh_domains.xdmf", "mesh_mb/mesh_boundaries.xdmf", subdomains=[])
my_mesh.define_markers()
ft = my_mesh.surface_markers
ct = my_mesh.volume_markers

# inspect values
print("List of facet tags: {}".format(list(set(ft.values))))
print("List of cell tags: {}".format(list(set(ct.values))))

# write to XDMF  for some reason cannot be opened in paraview
# volume_file = XDMFFile(MPI.COMM_WORLD, "mesh_mb/mesh_domains_dolfinx.xdmf", "w")
# volume_file.write_mesh(mesh)
# volume_file.write_meshtags(ct)

# boundary_file = XDMFFile(MPI.COMM_WORLD, "mesh_mb/mesh_boundaries_dolfinx.xdmf", "w")
# boundary_file.write_meshtags(ft)

Q = fem.FunctionSpace(my_mesh.mesh, ("DG", 0))
mf = fem.Function(Q)
mf.x.array[ct.indices] = ct.values
volume_file = XDMFFile(MPI.COMM_WORLD, "mesh_mb/mesh_domains_dolfinx.xdmf", "w")
volume_file.write_mesh(my_mesh.mesh)
volume_file.write_function(mf)

