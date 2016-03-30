# coding=utf-8
# Copyright (C) 2016 Yi Hu
# python 2.7, FEniCS 1.6.0
"""
Generic Material setting and Material Library

Class: Material
Function: green_lagrange, tensor_invar, compose
Material Library: st_venant_kirchhoff, simo_pister, magneto_mechano, neo_hook_mre

"""

from dolfin import *


class Material(object):
    """
    Material Class for 2D and 3D problem
    """

    def __init__(self, energy_func, para_list, field_label_tuple_list=None,
                 invariant_generator_list_list=None):
        """
        Constructor for Material

        :param energy_func: (function) energy function depending on invariants
        :param para_list: （list) list of parameters for energy function
        :param field_label_tuple_list: (list of tuples) invariant generator
                                        dependency, (0,) <=> (F,)
        :param invariant_generator_list_list: (list of lists) function list
                                            to obtain invariants, e.g.
                                            [[I1,I2], [I1], ...]

        :return: updated Material instance

        """
        self.energy_func = energy_func
        self.para_list = para_list

        self.F = None
        self.field_dim = []
        self.num_field = 0
        self.prob_dim = 0

        if not field_label_tuple_list:
            self.invar_gen_li = {}
        else:
            self.invar_gen_li = dict(zip(field_label_tuple_list,
                                         invariant_generator_list_list))

        self.invar = []
        self.psi = None

    def __call__(self, func_list):
        """
        Assemble material energy (psi) by calling instance

        :param func_list: (list of dolfin Functions)
                            e.g. [F, M, E, T, ...] each is Function

        :return: complete instance
        """
        if not isinstance(func_list, list):
            raise Exception('please input a list')
        self.F = func_list
        # print len(self.F)
        self._material_energy()

    def invariant_generator_append(self, field_label_tuple,
                                   invariant_generator_list):
        """
        Initialize method for invariants generation for material energy

        Please be careful! When the keys are the same, it will override the
        initial one!

        :param field_label_tuple: (tuple) field dependency, e.g. (0,), (0,1)
        :param invariant_generator_list: (list of functions) function list on
                                        the field, e.g. tr

        :return: renewed or updated invar_gen_li

        """
        self.invar_gen_li[field_label_tuple] = invariant_generator_list

    def _invariant(self):
        """
        Generate invariants using invar_generator to update self.invar

        :return: updated self.invar (list)
        """
        if not self.invar_gen_li:
            raise Exception('Please initialize the invariant generator list '
                            'first')
        invariant_generator = self.invar_gen_li
        # Clear self.invar before generation
        self.invar = []
        for fields_label in invariant_generator:
            for generator in invariant_generator[fields_label]:
                fields = [self.F[i] for i in fields_label]
                self.invar.append(generator(*fields))

    def _material_energy(self):
        """
        Subroutine that builds material energy (psi)

        :return: self.psi
        """
        self._invariant()
        self.psi = self.energy_func(self.invar, *self.para_list)

    def direction(self):
        """
        For heterogeneous material
        """
        pass

    def __str__(self):
        if not self.num_field:
            self._problem_setting()
        return '%dD Material depends on %d fields' % (self.prob_dim,
                                                      self.num_field)

    def _problem_setting(self):
        """
        Obtain general problem information
        """
        func_list = self.F
        self.num_field = len(func_list)
        self.prob_dim = func_list[0].geometric_dimension()

    def get_field_dim(self):
        """
        Obtain field information
        """
        for i, field in enumerate(self.F):
            field_shape = field.shape()
            self.field_dim[i] = field_shape
            print 'field %i', i,
            if not field_shape:
                print 'scalar field'
            elif len(field_shape) > 1:
                print 'tensor field'
            else:
                print 'vector field: %d' % field_shape[0]


# Tensor measure
def green_lagrange(F):
    """
    Generate green_lagrange tensor

    :param F: (dolfin function) right cauchy green

    :return: (dolfin function) green lagrange tensor

    """
    dim = F.geometric_dimension()
    I = Identity(dim)
    C = F.T*F
    E = 0.5*(C - I)
    return E


# Assistant method to make invariants
def tensor_invar(label_tuple):
    """
    Tensor invariants generator

    :param label_tuple: (tuple) label of invariants, I1, I2, I3

    :return: (list of functions) list of functions

    """
    generator_list = []
    if 1 in label_tuple:
        I1 = tr
        generator_list.append(I1)
    if 2 in label_tuple:
        def I2(tensor_function):
            return 0.5 * (tr(tensor_function)**2 -
                          tr(tensor_function.T*tensor_function))

        generator_list.append(I2)
    if 3 in label_tuple:
        I3 = det
        generator_list.append(I3)
    return generator_list


def compose(f_list, g):
    """
    Compose every fi and g

    :param f_list: (list of functions) [f1, f2, ...]
    :param g: (function)

    :return: composed function list
    :rtype: list

    """
    return [lambda x: fi(g(x)) for fi in f_list]


def st_venant_kirchhoff(E_m, nu_m):
    """
    St Venant Kirchhoff Material

    :param E_m: (float) Young's modulus
    :param nu_m: (float) Poisson ratio

    :return: Material svk

    """
    mu = E_m / (2 * (1 + nu_m))
    lmbda = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))

    # mu = Constant(E_m / (2 * (1 + nu_m)))
    # lmbda = Constant(E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m)))

    def psi(inv, lmbda, mu):
        return 0.5 * lmbda * (inv[0]) ** 2 + mu * inv[1]

    svk = Material(psi, [lmbda, mu])

    def invariant1(F):
        dim = F.geometric_dimension()
        I = Identity(dim)
        C = F.T * F
        E = 0.5 * (C - I)
        return tr(E)

    def invariant2(F):
        dim = F.geometric_dimension()
        I = Identity(dim)
        C = F.T * F
        E = 0.5 * (C - I)
        return tr(E.T * E)

    # Method 1: Use compose to generate invar_func
    # gen_li = compose()(tensor_invar((1,2)), green_lagrange)
    # svk.invariant_generator_append((0,), gen_li)

    # Method 2: Direct generate using functions
    svk.invariant_generator_append((0,), [invariant1, invariant2])

    return svk


def simo_pister(mu0, m0, lmbda0, ro0, cv, theta0):
    def psi(inva, mu0, m0, lmbda0, ro0, cv, theta0):
        """
        simo pister model (thermo-elasticity) 'from Hoere Mech 3'

        :param inva: inva[0]: I_c=tr(C), inva[1]: theta, inva[2]: det(C)
        :param mu0: elasticty coeff
        :param m0: mass
        :param lmbda0: elasticity coeef
        :param ro0: density
        :param cv: thermal capacity
        :param theta0: initial temperature
        """
        return 0.5 * mu0 * (inva[0] - 3) + (m0 * (inva[2] - theta0) - mu0) * ln(
                inva[1]) + \
               0.5 * lmbda0 * ln(inva[1]) ** 2 - \
               ro0 * cv * (inva[2] * ln(inva[2] / theta0) - (inva[2] - theta0))

    sp = Material(psi, [mu0, m0, lmbda0, ro0, cv, theta0])

    C_invar_gen = compose(tensor_invar((1,)), green_lagrange)

    def invariant3(C):
        return det(C) ** 0.5

    C_invar_gen.append(invariant3)

    sp.invariant_generator_append((0,), C_invar_gen)
    sp.invariant_generator_append((1,), [lambda x: x, ])

    return sp


def magneto_mechano(N, a, b, c):
    """
    Artificial material model, experiment mixed invariants
    invariants are partly referred from 'K.Danas, 2012, JMPS'

    :param C: Mech variable right cauchy green
    :param M: Megneto variable
    :param N: Symmetric direction
    :param a: coeff
    :param b: coeff
    :param c: coeff
    """

    def psi(inv, a, b, c):
        return a * (inv[0] + inv[1] + inv[2]) + b * (inv[3] + inv[4]) + c * inv[
            5]

    mre = Material(psi, [a, b, c])

    C_invar_gen = tensor_invar((1, 2, 3))
    C_invar_gen.append(lambda x: inner(N, x * N))
    mre.invariant_generator_append((0,), C_invar_gen)
    mre.invariant_generator_append((1,), [lambda x: inner(x, x), ])
    mre.invariant_generator_append((0, 1), [lambda x, y: inner(y, x * y), ])

    return mre


def neo_hook_eap(E_m, nu_m, kappa, epsi0=8.85e-12):
    """
    Neo-Hookean-type EAP from 'Keip, Steinmann, Schroeder, 2014, CMAME'

    :param E_m: Young's Modulus
    :param nu_m: Poisson ratio
    :param epsi0: Vacuum Permittivity
    :param kappa: Electric Susceptivity

    :return: Matrial nh_eap
    """
    miu = E_m / (2 * (1 + nu_m))
    lmbda = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))

    def psi(inva, miu, lmbda, kappa, epsi0):
        mech_term = 0.5 * miu * (inva[0] - 3) + lmbda / 4 * (inva[1] ** 2 - 1) - \
                    (lmbda / 2 + miu) * ln(inva[1])
        couple_term = -1 / 2 * epsi0 * (1 + kappa / inva[1]) * inva[1] * inva[2]
        return mech_term + couple_term

    nh_eap = Material(psi, [miu, lmbda, kappa, epsi0])

    def sqr_tr(F):
        return tr(F.T * F)

    nh_eap.invariant_generator_append((0,), [sqr_tr, det])
    couple_invar_gen = lambda F, E: inner(inv(F.T * F), outer(E, E))
    nh_eap.invariant_generator_append((0, 1), [couple_invar_gen])

    return nh_eap


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
    # print nh_eap.psi
    assert nh_eap.psi is not None


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
