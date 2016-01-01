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
    unit cell computation for 2D multi field problem
    !! 3D should be extended !!
    """

    def __init__(self, cell, bc, material_li,
                 strain_gen_li, strain_FS_li,
                 multi_field_label):
        """
        Initialize cell properties such as geometry, material, boundary,
        post processing space, and switch of multi field

        :param cell: geometry
        :param bc: boundary conditions
        :param material_li: [mat1, mat2, ...]
        :param strain_FS_li: post processing spaces
        :param multi_field_label: 1: multi, 2: uni
        """
        self.cell = cell
        self.bc = bc
        self.w = None
        self.material = material_li
        self.strain_gen = strain_gen_li
        self.strain_FS = strain_FS_li
        self.F_bar = None

        self.F = None

        self.F_w = None
        self.J = None

        self.FF = None

        # __Helping class member__
        # Geometery Dimension 2D
        self.geom_dim = 2
        self.w_composed = None
        # Test function
        self.v_composed = None
        # Trial function
        self.dw_composed = None
        self.w_composed_tuple = None
        self.multi_field_label = multi_field_label
        self.material_num = len(material_li)

    def input(self, F_bar_li, w_li):
        """
        Internal vars F_bar_li and w_li should be updated for each cell and
        each time step

        :param F_bar_li: macro F, written as list
        :param w_li: field list
        """
        self._F_bar_init(F_bar_li)
        self.w = w_li
        self._field_compose()
        self._strain_init()

    def _F_bar_init(self, F_bar_li):
        """
        Macro field input, initialize F_bar
        Mind that if multiple field, multiple field input will require

        :param F_bar_li: !! only constant macro field input are support !!
        :return self.F_bar: [Function for F, Function for M, Function for T,...]
        """
        if self.multi_field_label:
            self.F_bar = [self._li_to_func(F_bar_field_i)
                          for F_bar_field_i in F_bar_li]
        else:
            self.F_bar = self._li_to_func(F_bar_li)

    def _li_to_func(self, F_li):
        dim = self.geom_dim
        F_dim = len(F_li)
        if F_dim == 1:
            FS = FunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression('F', F=F_li[0])
            return project(F_ex, FS)
        elif F_dim == dim:
            VFS = VectorFunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression(('F1','F2'), F1=F_li[0], F2=F_li[1])
            return project(F_ex, VFS)
        elif F_dim == dim**2:
            TFS = TensorFunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression((("F11", "F12"), ("F21", "F22")),
                              F11=F_li[0],
                              F12=F_li[1],
                              F21=F_li[2],
                              F22=F_li[3])
            return project(F_ex, TFS)
        else:
            raise Exception('Please Input Right Dimension')


    # ==== Pre-Processing Stage ====
    def _field_compose(self):
        if self.multi_field_label:
            FS_li = [wi.function_space() for wi in self.w]
            MFS = MixedFunctionSpace(FS_li)
            self.w_composed = Function(MFS)
            self.v_composed = TestFunctions(MFS)
            self.dw_composed = TrialFunction(MFS)
            self.w_composed_tuple = self.w_composed.split()
        else:
            FS = self.w[0].function_space()
            self.w_composed = Function(FS)
            self.v_composed = TestFunctions(FS)
            self.dw_composed = TrialFunction(FS)
            # TODO no w_composed_tuple for one field case, is it correct??

    def _strain_init(self):
        generator_li = self.strain_gen
        if self.multi_field_label:
            self.F = [gen(self.F_bar[i], self.w_composed_tuple[i])
                      for i, gen in enumerate(generator_li)]
        else:
            self.F = [generator_li[0](self.F_bar, self.w_composed)]

    def _material_assem(self):
        for i in range(self.material_num):
            self.material[i](self.F)

    def _total_energy(self):
        # Please be careful about the ordering dx(0) -> matrix
        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)

        self._material_assem()

        int_i_li = [mat.psi * dx(i)
                    for i, mat in enumerate(self.material)]
        Pi = sum(int_i_li)
        return Pi

    def _fem_formulation_composite(self):
        Pi = self._total_energy()

        F_w = derivative(Pi, self.w_composed, self.v_composed)

        # Compute Jacobian of F
        J = derivative(F_w, self.w_composed, self.dw_composed)

        self.F_w = F_w
        self.J = J

    # ==== Solution ====
    def comp_fluctuation(self):
        self._fem_formulation_composite()

        solve(self.F_w == 0, self.w_composed, self.bc, J=self.J,
              form_compiler_parameters=ffc_options)

        # plot(self.w, mode='displacement', interactive=True)

        print 'fluctuation computation finished'

    # ==== Post-Processing Stage ====
    def compute_strain(self):
        F_space = TensorFunctionSpace(self.cell.mesh, 'DG', 0)
        self.FF = project(self.F_bar + grad(self.w), F_space)

        print 'strain computation finished'

    def comp_stress(self, *psi_material):
        # todo energy-like computation
        TFS = TensorFunctionSpace(self.cell.mesh, 'DG', 0)
        FF = self.FF
        Pi = self.total_energy(*psi_material)
        PP = derivative(Pi, FF)
        PP_assem = assemble(PP)
        print PP_assem

    def avg_stress(self, *psi_material):
        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
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
        TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
        FF_bar_trial = TrialFunction(TFS_R)
        FF_bar_test = TestFunction(TFS_R)

        # generate the constant form of C and assemble C
        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        i, j, k, l = indices(4)
        c = FF_bar_test[i, j] * C_i_[i, j, k, l] * FF_bar_trial[k, l] * dx(1) \
            + \
            FF_bar_test[i, j] * C_m_[i, j, k, l] * FF_bar_trial[k, l] * dx(0)
        cc = assemble(c)
        CC = cc.array()

        TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
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


def cauchy_green_with_macro(F_bar, w_component):
    return F_bar + grad(w_component)

if __name__ == '__main__':
    print 'this is for testing'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = Mesh(r"m.xml")
    cell = ce.unit_cell(mesh)
    inc = ce.Inclusion_Circle()
    inc = [inc]
    cell.inclusion(inc)

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner())

    # Set boundary conditions
    corners = cell.mark_corner_bc()
    fixed_corner = Constant((0.0, 0.0))
    bc = []
    for c in corners:
        bc.append(DirichletBC(VFS, fixed_corner, c, method='pointwise'))

    # Set materials
    w = Function(VFS)
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    F_bar = [0.9, 0., 0., 1.]
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, bc, mat_li, [cauchy_green_with_macro],
                            strain_space, 0)

    comp.input(F_bar, [w])
    comp.comp_fluctuation()

    # comp.compute_strain()
    # psi_m = comp.material_energy(E_m, nu_m)
    # psi_i = comp.material_energy(E_i, nu_i)
    # print comp.avg_stress(psi_m, psi_i)

    # comp.effective_moduli_2(E_m, nu_m, E_i, nu_i)

    # print comp.w.vector().array()
