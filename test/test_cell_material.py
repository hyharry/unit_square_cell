# coding = utf-8
# Copyright (C) Yi Hu


def test_st_venant():
    print 'Test Saint Venant Material'
    mesh = UnitSquareMesh(2, 2)
    FS = FunctionSpace(mesh, 'CG', 1)
    TFS = TensorFunctionSpace(mesh, 'CG', 1)

    VFS = VectorFunctionSpace(mesh, 'CG', 1)
    w = Function(VFS)
    # F = Function(TFS)
    F = grad(w)

    E_m, nu_m = 10.0, 0.3
    svk = st_venant_kirchhoff(E_m, nu_m)

    # print id(svk)
    svk([F])
    # print id(svk)


def test_simo_pister():
    print 'Test Simo Pister Thermomechanic Material'
    mesh = UnitSquareMesh(2, 2)
    FS = FunctionSpace(mesh, 'CG', 1)
    TFS = TensorFunctionSpace(mesh, 'CG', 1)

    VFS = VectorFunctionSpace(mesh, 'CG', 1)
    w = Function(VFS)
    # F = Function(TFS)
    F = grad(w)

    mu0, m0, lmbda0, ro0, cv, theta0 = 1, 2, 3, 4, 5, 6
    theta = Function(FS)
    sp = simo_pister(mu0, m0, lmbda0, ro0, cv, theta0)
    sp([F.T * F, theta, ])


def test_magneto_mechano():
    print 'Test Magneto Mechanic Material'
    mesh = UnitSquareMesh(2, 2)
    # FS = FunctionSpace(mesh, 'CG', 1)
    TFS = TensorFunctionSpace(mesh, 'CG', 1)
    VFS = VectorFunctionSpace(mesh, 'CG', 1)

    C = Function(TFS)
    M = Function(VFS)
    N = Constant((1., 1.))
    a, b, c = 1., 2., 3.
    mre = magneto_mechano(N, a, b, c)
    mre([C, M, ])


def test_neo_hookean_eap():
    print 'Test neo hookean EAP Material'
    mesh = UnitSquareMesh(2, 2)
    # FS = FunctionSpace(mesh, 'CG', 1)
    TFS = TensorFunctionSpace(mesh, 'CG', 1)
    VFS = VectorFunctionSpace(mesh, 'CG', 1)

    E_m, nu_m, kappa = 10.0, 0.3, 2
    C = Function(TFS)
    E = Function(VFS)
    nh_eap = neo_hook_eap(E_m, nu_m, kappa)
    nh_eap([C, E, ])
    assert nh_eap is not None


if __name__ == '__main__':
    print 'this is for testing'
    from dolfin import *
    # import unittest
    #
    # test_li = [test_st_venant, test_simo_pister,
    #            test_magneto_mechano, test_neo_hookean_eap]
    # test_case_li = [unittest.FunctionTestCase(test_i) for test_i in test_li]

    test_neo_hookean_eap()

    # print nh_eap.psi
    # print mre.psi
    # print sp.psi
    # print svk.psi
