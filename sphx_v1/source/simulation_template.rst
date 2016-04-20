
.. code:: python

    from dolfin import *
    
    import numpy as np
    
    import sys
    sys.path.append('../')
    
    import cell_geom as geom
    import cell_material as mat
    import cell_computation as comp
    
    ## Linear Backend
    
    parameters['linear_algebra_backend'] = 'Eigen'
    
    ## Define Geometry
    
    mesh = Mesh(r'../m_fine.xml')
    
    cell = geom.UnitCell(mesh)
    
    # Add inclusion
    inc = geom.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)
    
    ## Define Material
    
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = mat.st_venant_kirchhoff(E_m, nu_m)
    mat_i = mat.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]
    
    ## Define Computation
    
    VFS = VectorFunctionSpace(cell.mesh, "CG", 1, 
                              constrained_domain=geom.PeriodicBoundary_no_corner(2))
    
    def deform_grad_with_macro(F_bar, w_component):
        return F_bar + grad(w_component)
    
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    compute = comp.MicroComputation(cell, mat_li, 
                                    [deform_grad_with_macro],
                                    [strain_space])
    
    F_bar = [0.9, 0., 0., 1.]
    
    compute.input([F_bar], [w])
    
    # comp.set_solver_parameters('non_lin_newton', lin_method='direct',
    #                       linear_solver='cholesky')
    
    compute.comp_fluctuation(print_progress=True, print_solver_info=False)
    
    compute.view_fluctuation()
    
    delta = 0.01
    
    for i in range(10):
        F_bar[0] -= delta
        print F_bar
        compute.input([F_bar], [w])
        compute.comp_fluctuation(print_progress=True, print_solver_info=False)

