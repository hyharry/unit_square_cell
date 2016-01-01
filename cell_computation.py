# coding=utf-8

from dolfin import *
import numpy as np

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}


class MicroComputation(object):
    """
    unit cell computation for multi field porblem
    """
    def __init__(self, cell, bc, function_space):
        self.domain_mesh = cell.mesh
        self.subdom_data = cell.domain
        self.bc = bc

        self.function_space = function_space
        self.dw = TrialFunction(function_space)  # Incremental displacement
        self.v = TestFunction(function_space)  # Test function
        self.w = Function(function_space)  # Fluctuation

        self.F_w = None
        self.J = None

        self.F_bar = None
        self.FF = None

    def compute(self, *component_psi):
        self.fem_formulation_composite(*component_psi)

        solve(self.F_w == 0, self.w, self.bc, J=self.J,
              form_compiler_parameters=ffc_options)

        # plot(self.w, mode='displacement', interactive=True)

        print 'fluctuation computation finished'

    def compute_strain(self):
        F_space = TensorFunctionSpace(self.domain_mesh, 'DG', 0)
        self.FF = project(self.F_bar + grad(self.w), F_space)

        print 'strain computation finished'

    def comp_stress(self, *psi_material):
        # todo energy-like computation
        TFS = TensorFunctionSpace(self.domain_mesh, 'DG', 0)
        FF = self.FF
        Pi = self.total_energy(*psi_material)
        PP = derivative(Pi, FF)
        PP_assem = assemble(PP)
        print PP_assem

    def avg_stress(self, *psi_material):
        dx = Measure('dx', domain=self.domain_mesh,
                     subdomain_data=self.subdom_data)
        FF = self.FF
        d = self.w.geometric_dimension()

        mat_num = len(psi_material)
        PP_i = range(mat_num)
        PP_avg = np.zeros((d, d))
        for i in range(mat_num):
            PP_i[i] = diff(psi_material[i], FF)
        for i in range(d):
            for j in range(d):
                li = [PP_i[k][i, j] * dx(k) for k in range(mat_num)]
                PP_avg[i, j] = assemble(sum(li))

        return PP_avg

    def fem_formulation_composite(self, *material_list):
        Pi = self.total_energy(*material_list)

        # Compute first variation of Pi (directional derivative
        # about u in the direction of v)
        F_w = derivative(Pi, self.w, self.v)

        # Compute Jacobian of F
        J = derivative(F_w, self.w, self.dw)

        self.F_w = F_w
        self.J = J

    def material_energy(self, Em, nu):
        # todo make a class of material and define all the options for material
        F_bar = self.F_bar
        d = self.w.geometric_dimension()
        I = Identity(d)  # Identity tensor
        # Elasticity parameters
        mu = Constant(Em / (2 * (1 + nu)))
        lmbda = Constant(Em * nu / ((1 + nu) * (1 - 2 * nu)))

        if all(x == 0 for x in self.w.vector().array()):
            # if w is not obtained, the energy is expressed with grad(w)+F_bar
            F = F_bar + grad(self.w)  # Deformation gradient
        else:
            # if w is already calculated, the energy is expressed with F
            F = self.FF

        C = F.T * F  # Right Cauchy-Green tensor
        E = 0.5 * (C - I)  # Green Lagrange Tensor
        # free energy
        psi = (0.5 * lmbda) * (tr(E)) ** 2 + mu * tr(E.T * E)

        return psi

    def total_energy(self, *material):
        # please be careful about the ordering dx(0) -> matrix
        dx = Measure('dx', domain=self.domain_mesh,
                     subdomain_data=self.subdom_data)

        int_i_l = [mat * dx(i) for i, mat in enumerate(material)]
        Pi = sum(int_i_l)

        return Pi

    def tangent_moduli(self, material_energy):
        # todo calculate the tangent moduli from energy
        pass

    def effective_moduli_2(self, E_m, nu_m, E_i, nu_i):
        FF = self.FF
        psi_m_ = self.material_energy(E_m, nu_m)
        psi_i_ = self.material_energy(E_i, nu_i)

        # calculate the local moduli
        P_i = diff(psi_i_, FF)
        P_m = diff(psi_m_, FF)

        C_i_ = diff(P_i, FF)
        C_m_ = diff(P_m, FF)

        # trial and test function for the L matrix
        TFS_R = TensorFunctionSpace(self.domain_mesh, 'R', 0)
        FF_bar_trial = TrialFunction(TFS_R)
        FF_bar_test = TestFunction(TFS_R)

        # generate the constant form of C and assemble C
        dx = Measure('dx', domain=self.domain_mesh,
                     subdomain_data=self.subdom_data)
        i, j, k, l = indices(4)
        c = FF_bar_test[i, j] * C_i_[i, j, k, l] * FF_bar_trial[k, l] * dx(1)\
            + \
            FF_bar_test[i, j] * C_m_[i, j, k, l] * FF_bar_trial[k, l] * dx(0)
        cc = assemble(c)
        CC = cc.array()

        TFS_R = TensorFunctionSpace(self.domain_mesh, 'R', 0)
        F_bar_2_trial = TrialFunction(TFS_R)
        L2_der = derivative(self.F_w, self.F_bar, F_bar_2_trial)
        B2 = assemble(L2_der)

        LTKL2 = self.sensitivity(B2)

        # print CC-LTKL2

        return CC - LTKL2

    def sensitivity(self, B):
        w_test = self.v
        J = self.J
        bc = self.bc
        f = Constant((0.0, 0.0))
        b = inner(w_test, f) * dx
        K_a, L1_a = assemble_system(J, b, bc)

        rows = []
        for bc_i in bc:
            rows.extend(bc_i.get_boundary_values().keys())

        cols = [0, 1, 2, 3]
        vals = [0, 0, 0, 0]

        rs = np.array(rows, dtype=np.uintp)
        cs = np.array(cols, dtype=np.uintp)
        vs = np.array(vals, dtype=np.float_)
        for i in rs:
            B.setrow(i, cs, vs)
        B.setrow(1, cs, vs)
        # print B.array()
        B.apply('insert')
        L = B.array()
        # use the matrix from 1st method
        LTKL = np.zeros((4, 4))

        L_assign = Function(self.function_space)
        x = Vector()

        # compute four columns
        for i in range(4):
            L_assign.vector().set_local(L[:,
                                        i])  # set the coefficient of Function as a column of L matrix
            solve(K_a, x,
                  L_assign.vector())  # solve the system and store it in x (all arguments should be dolfin.Vector or Matrix)
            LTKL[:, i] = L.T.dot(
                x.array())  # set the answer to the column of LTKL3
        return LTKL

    def F_bar_init(self, F_bar):
        F_11 = Constant(F_bar[0])
        F_12 = Constant(F_bar[1])
        F_21 = Constant(F_bar[2])
        F_22 = Constant(F_bar[3])

        F_bar_ex = Expression((("F_11", "F_12"), ("F_21", "F_22")),
                              F_11=F_11,
                              F_12=F_12,
                              F_21=F_21,
                              F_22=F_22)
        TFS_R = TensorFunctionSpace(self.domain_mesh, "R", 0)
        F_bar_f = Function(TFS_R)
        F_bar_f.interpolate(F_bar_ex)
        self.F_bar = F_bar_f


if __name__ == '__main__':
    print 'this is for testing'
    import cell_geom as ce
    import cell_material as ma

    mesh = Mesh(r"m.xml")
    cell = ce.unit_cell(mesh)
    inc = ce.Inclusion_Circle()
    inc = [inc]
    cell.inclusion(inc)

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner())

    corners = cell.mark_corner_bc()
    fixed_corner = Constant((0.0, 0.0))
    bc = []
    for c in corners:
        bc.append(DirichletBC(VFS, fixed_corner, c, method='pointwise'))

    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    w = Function(VFS)
    # material_m = ma.st_venant_kirchhoff(grad(w),E_m,nu_m)
    # material_i = ma.st_venant_kirchhoff(grad(w),E_i,nu_i)
    # material_list = [material_m,material_i]

    # todo consider how to initialize materials which have func_list same as in MicroComp
    # todo every computation should remember the previous fluctuation
    # todo material function_list should be updated after comp_fluctuation

    F_bar = [0.9, 0., 0., 1.]
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)

    comp = computation(cell, bc, VFS)

    comp.F_bar_init(F_bar)

    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    psi_m = comp.material_energy(E_m, nu_m)
    psi_i = comp.material_energy(E_i, nu_i)

    comp.compute(psi_m, psi_i)

    comp.compute_strain()
    psi_m = comp.material_energy(E_m, nu_m)
    psi_i = comp.material_energy(E_i, nu_i)
    print comp.avg_stress(psi_m, psi_i)

    comp.effective_moduli_2(E_m, nu_m, E_i, nu_i)

    # print comp.w.vector().array()
