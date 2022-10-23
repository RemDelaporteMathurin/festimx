from festimx import convert_med_to_xdmf

convert_med_to_xdmf("mesh_mb/mesh_MB_ITER.med", cell_file="mesh_mb/converted/mesh_domains.xdmf", facet_file="mesh_mb/converted/mesh_boundaries.xdmf")
