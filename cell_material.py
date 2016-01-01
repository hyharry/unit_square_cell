# coding=utf-8


class Material(object):
    """
    Material Class for 2D problem
    possible to extend it into 3D
    """
    def __init__(self, energy_func, para_list):
        self.energy_func = energy_func
        self.para_li = para_list

        self.F = None
        self.num_field = 0
        self.field_dim = []
        self.prob_dim = 0

        self.invar_gen_li = {}
        self.invar = []
        self.psi = None

    def __call__(self, func_list):
        """
        Assemble material energy (psi) by calling instance

        :param func_list: e.g. [F, M, E, T, ...] each is Function

        :return: complete instance
        """
        self.func_init(func_list)
        self.material_energy()

    # Initialize instance separately, Modularize material assemble.
    def func_init(self, func_list):
        if not isinstance(func_list, list):
            raise Exception('please input a list')
        self.F = func_list
        self.num_field = len(func_list)
        self.prob_dim = func_list[0].geometric_dimension()

    def invariant_generator_init(self, field_label_tuple,
                                 invariant_generator_list):
        self.invar_gen_li[field_label_tuple] = invariant_generator_list

    def _invariant(self):
        invariant_generator = self.invar_gen_li
        for fields_label in invariant_generator:
            for generator in invariant_generator[fields_label]:
                fields = [self.F[i] for i in fields_label]
                self.invar.append(generator(*fields))

    def material_energy(self):
        """
        Subroutine that builds material energy (psi)

        :return: self.psi
        """
        self._invariant()
        self.psi = self.energy_func(self.invar, *self.para_li)

    def get_field_dim(self):
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

    def comp_general_strain(self):
        pass

    def comp_general_stress(self):
        pass

    def mech_energy(self):
        pass

    def material_update(self):
        pass

    def direction(self):
        pass

    def green_lagrange(self, F):
        I = Identity(self.prob_dim)
        C = F.T * F
        E = 0.5 * (C - I)
        return E

    # Assistant method to make invariants
    @staticmethod
    def tensor_invar(label_tuple):
        generator_list = []
        if 1 in label_tuple:
            I1 = tr
            generator_list.append(I1)
        if 2 in label_tuple:
            def I2(tensor_function):
                return 0.5 * (tr(tensor_function) ** 2 -
                              tr(tensor_function.T * tensor_function))

            generator_list.append(I2)
        if 3 in label_tuple:
            I3 = det
            generator_list.append(I3)
        return generator_list

    @staticmethod
    def compose(f, g):
        return [lambda x: fi(g(x)) for fi in f]


def st_venant_kirchhoff(F, E_m, nu_m):
    mu = Constant(E_m / (2 * (1 + nu_m)))
    lmbda = Constant(E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m)))

    def psi(inv, lmbda, mu):
        return 0.5*lmbda*(inv[0])**2 + mu*inv[1]

    svk = Material(psi, [lmbda, mu])

    def invariant1(F):
        I = Identity(2)
        C = F.T * F
        E = 0.5 * (C - I)
        return tr(E)

    def invariant2(F):
        I = Identity(2)
        C = F.T * F
        E = 0.5 * (C - I)
        return tr(E.T * E)

    # Method 1: Use compose to generate invar_func
    # gen_li = svk.compose(svk.tensor_invar((1,2)),svk.green_lagrange)
    # svk.invariant_generator_init((0,), gen_li)

    # Method 2: Direct generate using functions
    svk.invariant_generator_init((0,), [invariant1, invariant2])

    svk([F])
    return svk


def simo_pister(C, theta, mu0, m0, lmbda0, ro0, cv, theta0):
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
        return 0.5*mu0*(inva[0]-3) + (m0*(inva[2]-theta0) - mu0) * ln(inva[1]) +\
               0.5*lmbda0*ln(inva[1])**2 -\
               ro0*cv*(inva[2]*ln(inva[2]/theta0) - (inva[2]-theta0))

    sp = Material(psi, [mu0, m0, lmbda0, ro0, cv, theta0])

    C_invar_gen = sp.compose(sp.tensor_invar((1,)), sp.green_lagrange)
    def invariant3(C):
        return det(C) ** 0.5
    C_invar_gen.append(invariant3)

    sp.invariant_generator_init((0,), C_invar_gen)
    sp.invariant_generator_init((1,), [lambda x: x, ])

    sp([C, theta])
    return sp


def magneto_mechano(C, M, N, a, b, c):
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
    mre = Material(psi,[a,b,c])

    C_invar_gen = mre.tensor_invar((1, 2, 3))
    C_invar_gen.append(lambda x: inner(N, x * N))
    mre.invariant_generator_init((0,), C_invar_gen)
    mre.invariant_generator_init((1,), [lambda x: inner(x, x), ])
    mre.invariant_generator_init((0, 1), [lambda x, y: inner(y, x * y), ])

    mre([C, M])
    return mre


if __name__ == '__main__':
    print 'this is for testing'
    from dolfin import *

    mesh = UnitSquareMesh(2, 2)
    FS = FunctionSpace(mesh, 'CG', 1)
    TFS = TensorFunctionSpace(mesh, 'CG', 1)
    # F = Function(TFS)

    VFS = VectorFunctionSpace(mesh, 'CG', 1)
    w = Function(VFS)
    F = grad(w)

    FS = FunctionSpace(mesh, 'CG', 1)

    E_m, nu_m = 10.0, 0.3
    svk = st_venant_kirchhoff(F, E_m, nu_m)

    mu0, m0, lmbda0, ro0, cv, theta0 = 1, 2, 3, 4, 5, 6
    theta = Function(FS)
    sp = simo_pister(F.T*F, theta, mu0, m0, lmbda0, ro0, cv, theta0)

    C = Function(TFS)
    M = Function(VFS)
    N = Constant((1., 1.))
    a, b, c = 1., 2., 3.
    mre = magneto_mechano(C, M, N, a, b, c)

    print mre.psi