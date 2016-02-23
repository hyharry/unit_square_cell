# coding = utf-8
# Copyright (C) Yi Hu
# python 2.7, FEniCS 1.6.0
"""
Test for cell_computation, MicroComputation
"""

import sys
sys.path.insert(0, '../')

import unittest

from dolfin import *
import numpy as np
from ..
from cell_material import Material
import cell_geom as geom

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# parameters['linear_algebra_backend'] = 'PETSc'
parameters.update({'linear_algebra_backend': 'Eigen'})
# Solver parameters for the fluctuation solving stage
solver_parameters = {}
# Solver parameters for post processing
post_solver_parameters = {}


def test_uni_field():
    """
    Test for Uni Field Problems
    """
    print 'St-Venant Kirchhoff Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = Mesh(r"m.xml")
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    inc = ce.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  2))

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    F_bar = [1., 0.8, 0., 1.]
    # F_bar = [1., 0.5, 0., 1.]
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
                            [strain_space])

    comp.input([F_bar], [w])
    comp.comp_fluctuation()
    # comp.view_fluctuation()
    # comp.view_displacement()
    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    # comp.effective_moduli_2()


def test_multi_field():
    """
    Test for Multi Field Problem
    """
    print 'Neo-Hookean EAP Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = Mesh(r"m.xml")
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    inc = ce.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  2))
    FS = FunctionSpace(cell.mesh, "CG", 1,
                       constrained_domain=ce.PeriodicBoundary_no_corner(2))

    # Set materials
    E_m, nu_m, Kappa_m = 2e5, 0.4, 7.
    # n = 1000
    n = 10  # 13.Jan
    E_i, nu_i, Kappa_i = 1000 * E_m, 0.3, n * Kappa_m

    mat_m = ma.neo_hook_eap(E_m, nu_m, Kappa_m)
    mat_i = ma.neo_hook_eap(E_i, nu_i, Kappa_i)
    mat_li = [mat_m, mat_i]

    # Macro Field Boundary
    F_bar = [1., 0.,
             0., 1.]
    E_bar = [0., -0.2]

    # Solution Field
    w = Function(VFS)
    el_pot_phi = Function(FS)
    strain_space_w = TensorFunctionSpace(mesh, 'DG', 0)
    strain_space_E = VectorFunctionSpace(mesh, 'DG', 0)

    def deform_grad_with_macro(F_bar, w_component):
        return F_bar + grad(w_component)

    def e_field_with_macro(E_bar, phi):
        return E_bar - grad(phi)

    # Computation Initialization
    comp = MicroComputation(cell, mat_li,
                            [deform_grad_with_macro, e_field_with_macro],
                            [strain_space_w, strain_space_E])

    comp.input([F_bar, E_bar], [w, el_pot_phi])
    comp.comp_fluctuation()
    # comp.view_displacement()
    # comp.view_fluctuation(1)
    # comp.view_post_processing('stress', 5)
    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    # comp.effective_moduli_2()


def test_uni_field_3d():
    """
    Test for Uni Field 3d Problem
    """
    print 'St-Venant Kirchhoff Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = UnitCubeMesh(16, 16, 16)
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    # inc = ce.InclusionRectangle(3, .25, .75, .25, .75, .25, .75)
    inc = ce.InclusionRectangle(3, 0., 1., .25, .75, .25, .75)
    inc_di = {'box': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  3))

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    # if multi field bc should match
    F_bar = [.9, 0., 0.,
             0., 1., 0.,
             0., 0., 1.]
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
                            [strain_space])

    comp.input([F_bar], [w])
    comp.comp_fluctuation()
    comp.view_fluctuation()

    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    # comp.effective_moduli_2()


def test_solver():
    """
    Test for Different Solvers
    """
    print 'Solver Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = Mesh(r"m.xml")
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    inc = ce.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  2))

    # global parameters

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    F_bar = [1., 0.8, 0., 1.]
    # F_bar = [1., 0.5, 0., 1.]
    # parameters['linear_algebra_backend'] = 'Eigen'
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
                            [strain_space])

    comp.input([F_bar], [w])

    # Test PETSc backend
    # set_solver_parameters('snes', 'iterative', 'minres')
    # set_solver_parameters('non_lin_newton', 'iterative', 'cg')
    # set_solver_parameters('non_lin_newton', 'direct', 'default')
    # set_solver_parameters('non_lin_newton')
    # set_solver_parameters('snes')
    # set_solver_parameters('non_lin_newton')

    # Test Eigen backend, backend definition is to be defined before all the
    # initialization (dolfin Function) and computation
    set_solver_parameters('non_lin_newton', lin_method='direct',
                          linear_solver='sparselu')
    # set_solver_parameters('non_lin_newton')

    # info(NonlinearVariationalSolver.default_parameters(), True)

    comp.comp_fluctuation(print_progress=True, print_solver_info=False)


if __name__ == '__main__':
    # test_uni_field()
    # test_multi_field()
    # test_uni_field_3d()
    test_solver()