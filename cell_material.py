# coding=utf-8

from dolfin import *
from dolfin.cpp.mesh import UnitSquareMesh


class Material(object):
    def __init__(self, func_list, energy_func, para_list):
        if not isinstance(func_list, list):
            raise Exception('please input a list')
        self.num_field = len(func_list)
        self.field_dim = []
        self.F = func_list
        self.invar = []
        self.psi = None
        self.energy_func = energy_func
        self.para_li = para_list
        self.prob_dim = func_list[0].geometric_dimension()
        self.invar_gen_li = {}

    def invariant(self):
        invariant_generator = self.invar_gen_li
        for fields_label in invariant_generator:
            for generator in invariant_generator[fields_label]:
                fields = [self.F[i] for i in fields_label]
                self.invar.append(generator(*fields))

    def invariant_generator_init(self, field_label_tuple, invariant_generator_list):
        self.invar_gen_li[field_label_tuple] = invariant_generator_list

    def total_energy(self):
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
        C = F.T * F  # Right Cauchy-Green tensor
        E = 0.5 * (C - I)  # Green Lagrange Tensor
        return E

    def tensor_invar(self, label_tuple):
        generator_list = []
        if 1 in label_tuple:
            I1 = tr
            generator_list.append(I1)
        if 2 in label_tuple:
            def I2(tensor_function):
                return 0.5*(tr(tensor_function)**2 -
                            tr(tensor_function.T*tensor_function))
            generator_list.append(I2)
        if 3 in label_tuple:
            I3 = det
            generator_list.append(I3)
        return generator_list

    def compose(self,f,g):
        return [lambda x: fi(g(x)) for fi in f]

def st_venant_kirchhoff(F, E_m, nu_m):
    svk = Material([F])
    def invariant1(F):
        I = Identity(2)
        C = F.T * F  # Right Cauchy-Green tensor
        E = 0.5 * (C - I)  # Green Lagrange Tensor
        return tr(E)


    def invariant2(F):
        I = Identity(2)
        C = F.T * F  # Right Cauchy-Green tensor
        E = 0.5 * (C - I)  # Green Lagrange Tensor
        return tr(E.T * E)

    # gen_li = svk.compose(svk.tensor_invar((1,2)),svk.green_lagrange)

    svk.invariant_generator_init((0,),[invariant1,invariant2])
    # svk.invariant_generator_init((0,), gen_li)

    svk.invariant()

    mu = Constant(E_m / (2 * (1 + nu_m)))
    lmbda = Constant(E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m)))

    def psi(inv, lmbda, mu):
        return (0.5 * lmbda) * (inv[0]) ** 2 + mu * inv[1]

    svk.total_energy(psi, lmbda, mu)

    return svk

def simo_pister(C,theta,mu0, m0, lmbda0, ro0, cv, theta0):
    sp = Material([C,theta])
    C_invar_gen = sp.compose(sp.tensor_invar((1,)),sp.green_lagrange)
    def invariant3(C):
        return det(C)**0.5
    C_invar_gen.append(invariant3)
    sp.invariant_generator_init((0,),C_invar_gen)
    sp.invariant_generator_init((1,),[lambda x:x,])
    print sp.invar_gen_li
    sp.invariant()

    def psi(inv,mu0, m0, lmbda0, ro0, cv, theta0):
        return 0.5*mu0*(inv[0]-3) + (m0*(inv[2]-theta0)-mu0)*ln(inv[1])\
               + 0.5*lmbda0*ln(inv[1])**2 -ro0*cv*(inv[2]*ln(inv[2]/theta0)-(inv[2]-theta0))

    sp.total_energy(psi,mu0, m0, lmbda0, ro0, cv, theta0)

    return sp

def magneto_mechano(C,M,N,a,b,c):
    mre = Material([C,M])
    C_invar_gen = mre.tensor_invar((1,2,3))
    C_invar_gen.append(lambda x: inner(N,x*N))
    mre.invariant_generator_init((0,), C_invar_gen)
    mre.invariant_generator_init((1,), [lambda x: inner(x,x),])
    mre.invariant_generator_init((0,1), [lambda x,y: inner(y, x*y),])
    mre.invariant()

    def psi(inv,a,b,c):
        return a*(inv[0]+inv[1]+inv[2]) + b*(inv[3]+inv[4]) + c*inv[5]

    mre.total_energy(psi,a,b,c)

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
    sp = simo_pister(F.T*F, theta,mu0, m0, lmbda0, ro0, cv, theta0)

    C = Function(TFS)
    M = Function(VFS)
    N = Constant((1.,1.))
    a, b, c = 1., 2., 3.
    mre = magneto_mechano(C,M,N,a,b,c)

    print mre.psi

    # todo how to deep copy a user object?

    print 'copy testing'

    from copy import deepcopy

    svk2 = deepcopy(svk)

    print id(svk), id(svk2)
    F = Function(TFS)

    svk2.F = F

    print id(svk), id(svk2)
