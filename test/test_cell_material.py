# coding = utf-8
# Copyright (C) Yi Hu
"""
Unit test for cell_material
"""
import unittest
import sys
from dolfin import *

sys.path.append('../')
import cell_material as mat


class MaterialTestCase(unittest.TestCase):
    """
    Test Various Materials
    """
    def setUp(self):
        mesh = UnitSquareMesh(2, 2)

        FS = FunctionSpace(mesh, 'CG', 1)
        TFS = TensorFunctionSpace(mesh, 'CG', 1)
        VFS = VectorFunctionSpace(mesh, 'CG', 1)

        self.w = Function(VFS)
        # F = Function(TFS)
        self.F = grad(self.w)
        self.theta = Function(FS)
        self.C = Function(TFS)
        self.M = Function(VFS)
        self.E = Function(VFS)

    def test_st_venant(self):
        """
        Test Saint Venant Material
        """
        E_m, nu_m = 10.0, 0.3
        svk = mat.st_venant_kirchhoff(E_m, nu_m)
        svk([self.F])
        self.assertIsNotNone(svk.psi)

    def test_simo_pister(self):
        """
        Test Simo Pister Thermomechanic Material
        """
        mu0, m0, lmbda0, ro0, cv, theta0 = 1, 2, 3, 4, 5, 6
        sp = mat.simo_pister(mu0, m0, lmbda0, ro0, cv, theta0)
        sp([self.F.T * self.F, self.theta, ])
        self.assertIsNotNone(sp.psi)

    def test_magneto_mechano(self):
        """
        Test Magneto Mechanic Material
        """
        N = Constant((1., 1.))
        a, b, c = 1., 2., 3.
        mre = mat.magneto_mechano(N, a, b, c)
        mre([self.C, self.M, ])
        self.assertIsNotNone(mre.psi)

    def test_neo_hookean_eap(self):
        """
        Test neo hookean EAP Material
        """
        E_m, nu_m, kappa = 10.0, 0.3, 2
        nh_eap = mat.neo_hook_eap(E_m, nu_m, kappa)
        nh_eap([self.C, self.E, ])
        self.assertIsNotNone(nh_eap.psi)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(MaterialTestCase)
    # unittest.TextTestRunner().run(suite)
