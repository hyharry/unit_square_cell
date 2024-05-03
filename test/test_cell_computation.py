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
import cell_material as mat
import cell_geom as geom
import cell_computation as com

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

parameters['linear_algebra_backend'] = 'PETSc'
# parameters.update({'linear_algebra_backend': 'Eigen'})
# Solver parameters for the fluctuation solving stage
solver_parameters = {}
# Solver parameters for post processing
post_solver_parameters = {}


class TwoDimUniTestCase(unittest.TestCase):
    """
    Test for Uni Field Problems
    """
    def setUp(self):
        mesh = geom.Mesh(r"../m.xml")
        # mesh = geom.Mesh(r"../m_fine.xml")
        cell = geom.UnitCell(mesh)
        inc = geom.InclusionCircle(2, (0.5, 0.5), 0.25)
        inc_di = {'circle_inc': inc}
        cell.set_append_inclusion(inc_di)
        VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                                  constrained_domain=geom.PeriodicBoundary_no_corner(2))

        # Set materials
        E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
        mat_m = mat.st_venant_kirchhoff(E_m, nu_m)
        mat_i = mat.st_venant_kirchhoff(E_i, nu_i)
        mat_li = [mat_m, mat_i]

        # Initialize MicroComputation
        F_bar = [1., 0.8, 0., 1.]
        # F_bar = [1., 0.5, 0., 1.]
        self.w = Function(VFS)
        strain_space = TensorFunctionSpace(cell.mesh, 'DG', 0)
        self.comp = com.MicroComputation(cell, mat_li,
                                         [com.deform_grad_with_macro],
                                         [strain_space])

        self.comp.input([F_bar], [self.w])
        self.comp.comp_fluctuation()

    def test_comp_fluct(self):
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_comp_strain(self):
        self.comp.comp_strain()
        val_set = set(self.comp.F[0].vector().array())
        self.assertNotEqual(len(val_set), 1)

    def test_comp_stress(self):
        self.comp.comp_stress()
        val_set = set(self.comp.P_merge.vector().array())
        self.assertNotEqual(len(val_set), 1)

    def test_avg_merge_strain(self):
        avg_m_strain = self.comp.avg_merge_strain()
        self.assertTrue(avg_m_strain.all())

    def test_avg_merge_stress(self):
        self.assertTrue(self.comp.avg_merge_stress().all())

    def test_avg_merge_moduli(self):
        self.assertTrue(self.comp.avg_merge_moduli().all())

    def test_effective_modu_2(self):
        self.assertTrue(self.comp.effective_moduli_2().all())

    @unittest.skip('visualization is skipped')
    def test_view_fluctuation(self):
        self.comp.view_fluctuation()
        self.comp.view_displacement()


class TwoDimMultiTestCase(unittest.TestCase):
    """
    Test for Multi Field Problem
    """
    def setUp(self):
        # Set geometry
        mesh = geom.Mesh(r"../m.xml")
        # mesh = geom.Mesh(r"../m_fine.xml")
        cell = geom.UnitCell(mesh)
        inc = geom.InclusionCircle(2, (0.5, 0.5), 0.25)
        inc_di = {'circle_inc': inc}
        cell.set_append_inclusion(inc_di)

        VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                                  constrained_domain=geom.PeriodicBoundary_no_corner(
                                      2))
        FS = FunctionSpace(cell.mesh, "CG", 1,
                           constrained_domain=geom.PeriodicBoundary_no_corner(2))

        # Set materials
        E_m, nu_m, Kappa_m = 2e5, 0.4, 7.
        # n = 1000
        n = 10  # 13.Jan
        E_i, nu_i, Kappa_i = 1000 * E_m, 0.3, n * Kappa_m

        mat_m = mat.neo_hook_eap(E_m, nu_m, Kappa_m)
        mat_i = mat.neo_hook_eap(E_i, nu_i, Kappa_i)
        mat_li = [mat_m, mat_i]

        # Macro Field Boundary
        F_bar = [1., 0.,
                 0., 1.]
        E_bar = [0., -0.2]

        # Solution Field
        self.w = Function(VFS)
        self.el_pot_phi = Function(FS)
        strain_space_w = TensorFunctionSpace(cell.mesh, 'DG', 0)
        strain_space_E = VectorFunctionSpace(cell.mesh, 'DG', 0)

        def deform_grad_with_macro(F_bar, w_component):
            return F_bar + grad(w_component)

        def e_field_with_macro(E_bar, phi):
            return E_bar - grad(phi)

        # Computation Initialization
        self.comp = com.MicroComputation(cell, mat_li,
                                         [deform_grad_with_macro, e_field_with_macro],
                                         [strain_space_w, strain_space_E])

        self.comp.input([F_bar, E_bar], [self.w, self.el_pot_phi])
        self.comp.comp_fluctuation()

    def test_comp_fluct(self):
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(val_set, 0.)

    def test_comp_strain(self):
        self.comp.comp_strain()
        val_set = set(self.comp.F[0].vector().array())
        self.assertNotEqual(len(val_set), 1)

    def test_comp_stress(self):
        self.comp.comp_stress()
        val_set = set(self.comp.P_merge.vector().array())
        self.assertNotEqual(len(val_set), 1)

    def test_avg_merge_strain(self):
        avg_m_strain = self.comp.avg_merge_strain()
        self.assertTrue(avg_m_strain.all())

    def test_avg_merge_stress(self):
        self.assertTrue(self.comp.avg_merge_stress().all())

    def test_avg_merge_moduli(self):
        self.assertTrue(self.comp.avg_merge_moduli().all())

    def test_effective_modu_2(self):
        self.assertTrue(self.comp.effective_moduli_2().all())

    @unittest.skip('visualization is skipped')
    def test_view_fluctuation(self):
        self.comp.view_fluctuation(1)
        self.comp.view_displacement()
        self.comp.view_post_processing('stress', 5)


class ThreeDimUniTestCase(unittest.TestCase):
    """
    Test for Uni Field 3d Problem
    """
    def setUp(self):
        # Set geometry
        mesh = geom.UnitCubeMesh(16, 16, 16)
        cell = geom.UnitCell(mesh)
        # inc = geom.InclusionRectangle(3, .25, .75, .25, .75, .25, .75)
        inc = geom.InclusionRectangle(3, 0., 1., .25, .75, .25, .75)
        inc_di = {'box': inc}
        cell.set_append_inclusion(inc_di)
        # cell.view_domain()

        VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                                  constrained_domain=geom.PeriodicBoundary_no_corner(3))

        # Set materials
        E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
        mat_m = mat.st_venant_kirchhoff(E_m, nu_m)
        mat_i = mat.st_venant_kirchhoff(E_i, nu_i)
        mat_li = [mat_m, mat_i]

        # Initialize MicroComputation
        # if multi field bc should match
        F_bar = [.9, 0., 0.,
                 0., 1., 0.,
                 0., 0., 1.]
        self.w = Function(VFS)
        strain_space = TensorFunctionSpace(cell.mesh, 'DG', 0)
        self.comp = com.MicroComputation(cell, mat_li,
                                      [com.deform_grad_with_macro],
                                    [strain_space])

        self.comp.input([F_bar], [self.w])
        com.set_solver_parameters('snes', 'iterative', 'minres')
        self.comp.comp_fluctuation()

    def test_comp_fluct(self):
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_comp_strain(self):
        self.comp.comp_strain()
        val_set = set(self.comp.F[0].vector().array())
        self.assertNotEqual(len(val_set), 1)

    def test_comp_stress(self):
        self.comp.comp_stress()
        val_set = set(self.comp.P_merge.vector().array())
        self.assertNotEqual(len(val_set), 1)

    def test_avg_merge_strain(self):
        avg_m_strain = self.comp.avg_merge_strain()
        self.assertTrue(avg_m_strain.all())

    def test_avg_merge_stress(self):
        self.assertTrue(self.comp.avg_merge_stress().all())

    def test_avg_merge_moduli(self):
        self.assertTrue(self.comp.avg_merge_moduli().all())

    def test_effective_modu_2(self):
        com.set_post_solver_parameters(lin_method='iterative',)
        self.assertTrue(self.comp.effective_moduli_2().all())

    @unittest.skip('visualization is skipped')
    def test_view_fluctuation(self):
        self.comp.view_fluctuation()
        self.comp.view_displacement()


@unittest.skipIf(parameters['linear_algebra_backend'] is 'PETSc',
                 'linear algebra backend not PETSc')
class PETScSolverParaTestCase(unittest.TestCase):
    """
    Test for Different Solvers
    """
    def setUp(self):
        # global parameters
        # parameters['linear_algebra_backend'] = 'Eigen'

        # Set geometry
        mesh = geom.Mesh(r"../m.xml")
        # mesh = geom.Mesh(r"m_fine.xml")
        cell = geom.UnitCell(mesh)
        inc = geom.InclusionCircle(2, (0.5, 0.5), 0.25)
        inc_di = {'circle_inc': inc}
        cell.set_append_inclusion(inc_di)
        # cell.view_domain()

        VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                                  constrained_domain=geom.PeriodicBoundary_no_corner(2))

        # global parameters

        # Set materials
        E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
        mat_m = mat.st_venant_kirchhoff(E_m, nu_m)
        mat_i = mat.st_venant_kirchhoff(E_i, nu_i)
        mat_li = [mat_m, mat_i]

        # Initialize MicroComputation
        F_bar = [1., 0.8, 0., 1.]
        # F_bar = [1., 0.5, 0., 1.]
        # parameters['linear_algebra_backend'] = 'Eigen'
        self.w = Function(VFS)
        strain_space = TensorFunctionSpace(cell.mesh, 'DG', 0)
        self.comp = com.MicroComputation(cell, mat_li,
                                         [com.deform_grad_with_macro],
                                         [strain_space])

        self.comp.input([F_bar], [self.w])

    def test_snes_default(self):
        com.set_solver_parameters('snes')
        self.comp.comp_fluctuation(print_progress=True)
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_snes_setting(self):
        com.set_solver_parameters('snes', 'iterative', 'minres')
        self.comp.comp_fluctuation(print_progress=True)
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_nonlin_newton_default(self):
        com.set_solver_parameters('non_lin_newton')
        self.comp.comp_fluctuation(print_progress=True)
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_nonlin_newton_setting_1(self):
        com.set_solver_parameters('non_lin_newton', 'iterative', 'cg')
        self.comp.comp_fluctuation(print_progress=True)
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_nonlin_newton_setting_2(self):
        com.set_solver_parameters('non_lin_newton', 'direct')
        self.comp.comp_fluctuation(print_progress=True)
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_post_solver_para_dir_default(self):
        com.set_solver_parameters('non_lin_newton', 'iterative', 'cg')
        self.comp.comp_fluctuation(print_progress=True)
        com.set_post_solver_parameters(lin_method='direct')
        self.comp.effective_moduli_2()
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_post_solver_para_iter_default(self):
        com.set_solver_parameters('non_lin_newton', 'iterative', 'cg')
        self.comp.comp_fluctuation(print_progress=True)
        com.set_post_solver_parameters(lin_method='iterative')
        self.comp.effective_moduli_2()
        # val_set = set(self.comp.w_merge.vector().array()) # Yi 2024
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)


@unittest.skipIf(parameters['linear_algebra_backend'] is not 'Eigen',
                 'linear algebra backend not Eigen')
class EigenSolverParaTestCase(unittest.TestCase):
    def setUp(self):
        # global parameters
        # parameters['linear_algebra_backend'] = 'Eigen'

        # Set geometry
        mesh = geom.Mesh(r"../m.xml")
        # mesh = geom.Mesh(r"m_fine.xml")
        cell = geom.UnitCell(mesh)
        inc = geom.InclusionCircle(2, (0.5, 0.5), 0.25)
        inc_di = {'circle_inc': inc}
        cell.set_append_inclusion(inc_di)
        # cell.view_domain()

        VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                                  constrained_domain=geom.PeriodicBoundary_no_corner(2))

        # global parameters

        # Set materials
        E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
        mat_m = mat.st_venant_kirchhoff(E_m, nu_m)
        mat_i = mat.st_venant_kirchhoff(E_i, nu_i)
        mat_li = [mat_m, mat_i]

        # Initialize MicroComputation
        F_bar = [1., 0.8, 0., 1.]
        # F_bar = [1., 0.5, 0., 1.]
        # parameters['linear_algebra_backend'] = 'Eigen'
        self.w = Function(VFS)
        strain_space = TensorFunctionSpace(cell.mesh, 'DG', 0)
        self.comp = com.MicroComputation(cell, mat_li,
                                         [com.deform_grad_with_macro],
                                         [strain_space])

        self.comp.input([F_bar], [self.w])

    def test_nonlin_newton_default(self):
        com.set_solver_parameters('non_lin_newton')
        # val_set = set(self.comp.w_merge.vector().array())
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    def test_nonlin_newton_dir_sparselu(self):
        com.set_solver_parameters('non_lin_newton', lin_method='direct',
                                  linear_solver='sparselu')
        self.comp.comp_fluctuation(print_progress=True)
        # val_set = set(self.comp.w_merge.vector().array())
        val_set = set(np.array(self.comp.w_merge.vector()))
        self.assertNotEqual(len(val_set), 1)

    # info(NonlinearVariationalSolver.default_parameters(), True)

if __name__ == '__main__':
    # test_uni_field()
    # test_multi_field()
    # test_uni_field_3d()
    # test_solver()
    # unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TwoDimMultiTestCase)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TwoDimUniTestCase)
    # suite = unittest.TestLoader().loadTestsFromTestCase(ThreeDimUniTestCase)
    # parameters.update({'linear_algebra_backend': 'Eigen'})
    suite = unittest.TestLoader().loadTestsFromTestCase(PETScSolverParaTestCase)
    # suite = unittest.TestLoader().loadTestsFromTestCase(EigenSolverParaTestCase)
    unittest.TextTestRunner().run(suite)
