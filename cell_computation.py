# coding=utf-8

from dolfin import *
import numpy as np
from cell_geom import PeriodicBoundary_no_corner

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

    def __init__(self, cell, material_li,
                 strain_gen_li, strain_FS_li,
                 multi_field_label):
        """
        Initialize cell properties such as geometry, material, boundary,
        post processing space, and switch of multi field

        :param cell: geometry
        :param bc: boundary conditions
        :param material_li: [mat1, mat2, ...]
        :param strain_FS_li: post processing spaces
        :param multi_field_label: 1: multi, 0: uni

        ** Attention: multi_field_label should be set currently, when uni
        field, assign in merge method does not work **
        """
        self.cell = cell
        self.w = None
        self.material = material_li
        self.strain_gen = strain_gen_li
        self.strain_FS = strain_FS_li
        self.F_bar = None

        self.F = None

        self.Pi = None
        self.F_w = None
        self.J = None

        # __Helping class member__
        # Geometry Dimension 2D
        self.field_num = None
        self.geom_dim = 2
        self.w_merge = None
        # Test function
        self.v_merge = None
        # Trial function
        self.dw_merge = None
        self.w_split = None
        # Boundary Condition
        self.bc = None
        self.w_dim = None
        self.multi_field_label = multi_field_label
        self.material_num = len(material_li)

        # F for Post Processing
        self.F_merge = None
        self.F_merge_test = None
        self.F_merge_trial = None
        self.F_split = None

        self.P = None

    def input(self, F_bar_li, w_li):
        """
        Internal vars F_bar_li and w_li should be updated for each cell and
        each time step

        :param F_bar_li: macro F, written as list
        :param w_li: field list
        """
        assert isinstance(F_bar_li, list) and isinstance(w_li, list)
        self._F_bar_init(F_bar_li)
        self.w = w_li
        self.w_dim = [w_i.shape() for w_i in w_li]
        self.field_num = len(w_li)
        label = 'w'
        self._field_merge(label)
        self._field_split(label)
        self._strain_init(label)

    def _F_bar_init(self, F_bar_li):
        """
        Macro field input, initialize F_bar
        Mind that if multiple field, multiple field input will require

        :param F_bar_li: !! only constant macro field input are support !!
        F_bar_li should be a list of F_bar, each entry is furthermore a list
        representing input from each field
        :return self.F_bar: [Function for F, Function for M, Function for T,...]
        """
        assert isinstance(F_bar_li[0], list)
        self.F_bar = [self._li_to_func(F_bar_field_i)
                      for F_bar_field_i in F_bar_li]

    def _li_to_func(self, F_li):
        dim = self.geom_dim
        F_dim = len(F_li)
        if F_dim == 1:
            FS = FunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression('F', F=F_li[0])
            return project(F_ex, FS)
        elif F_dim == dim:
            VFS = VectorFunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression(('F1', 'F2'), F1=F_li[0], F2=F_li[1])
            return project(F_ex, VFS)
        elif F_dim == dim**2:
            TFS = TensorFunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression((("F11", "F12"),
                               ("F21", "F22")),
                              F11=F_li[0], F12=F_li[1],
                              F21=F_li[2], F22=F_li[3])
            return project(F_ex, TFS)
        else:
            raise Exception('Please Input Right Dimension')

    def _field_merge(self, label):
        if label is 'w':
            if self.multi_field_label:
                FS_li = [wi.function_space() for wi in self.w]
                MFS = MixedFunctionSpace(FS_li)
                self.w_merge = Function(MFS)
                self.v_merge = TestFunction(MFS)
                self.dw_merge = TrialFunction(MFS)
                for i, wi in enumerate(self.w):
                    assign(self.w_merge.sub(i), wi)
            else:
                FS = self.w[0].function_space()
                self.w_merge = Function(FS)
                self.v_merge = TestFunction(FS)
                self.dw_merge = TrialFunction(FS)
                assign(self.w_merge, self.w[0])
        elif label is 'F':
            if self.multi_field_label:
                FS_li = [Fi.function_space() for Fi in self.F]
                MFS = MixedFunctionSpace(FS_li)
                self.F_merge = Function(MFS)
                self.F_merge_test = TestFunction(MFS)
                self.F_merge_trial = TrialFunction(MFS)
                for i, Fi in enumerate(self.F):
                    assign(self.F_merge.sub(i), Fi)
            else:
                FS = self.F[0].function_space()
                self.F_merge = Function(FS)
                self.F_merge_test = TestFunction(FS)
                self.F_merge_trial = TrialFunction(FS)
                assign(self.F_merge, self.F[0])
        else:
            raise Exception('wrong label for merge!')

    def _field_split(self, label):
        if label is 'w':
            if self.multi_field_label:
                self.w_split = list(split(self.w_merge))
            else:
                self.w_split = [self.w_merge]
        elif label is 'F':
            if self.multi_field_label:
                self.F_split = list(split(self.F_merge))
            else:
                self.F_split = [self.F_merge]
        else:
            raise Exception('wrong label for split!')

    def _strain_init(self, label):
        generator_li = self.strain_gen
        if label is 'w':
            self.F = [gen(self.F_bar[i], self.w_split[i])
                      for i, gen in enumerate(generator_li)]
        elif label is 'F':
            self.F = [self.F_bar[i] + self.F_split[i]
                      for i in range(self.field_num)]
        else:
            raise Exception('wrong label for strain init!')

    # ==== Pre-Processing Stage ====
    def _material_assem(self, F):
        for i in range(self.material_num):
            self.material[i](F)

        # print id(self.material[0].invar[0])

    def _total_energy(self, F):
        # Please be careful about the ordering dx(0) -> matrix
        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)

        self._material_assem(F)

        int_i_li = [mat.psi*dx(i)
                    for i, mat in enumerate(self.material)]
        self.Pi = sum(int_i_li)

    def _bc_fixed_corner(self):
        bc = []
        corners = self.cell.mark_corner_bc()
        dim = self.w_merge.shape()
        if dim:
            fixed_corner = Constant((0,)*dim[0])
        else:
            fixed_corner = Constant(0.)
        for c in corners:
            bc.append(DirichletBC(self.w_merge.function_space(),
                                  fixed_corner, c, method='pointwise'))
        self.bc = bc

    def _fem_formulation_composite(self):
        self._total_energy(self.F)

        F_w = derivative(self.Pi, self.w_merge, self.v_merge)

        # Compute Jacobian of F
        J = derivative(F_w, self.w_merge, self.dw_merge)

        self.F_w = F_w
        self.J = J
        self._bc_fixed_corner()

    # ==== Solution ====
    def comp_fluctuation(self):
        self._fem_formulation_composite()

        solve(self.F_w == 0, self.w_merge, self.bc, J=self.J,
              form_compiler_parameters=ffc_options)

        # plot(self.w_merge, mode='displacement', interactive=True)

        print 'fluctuation computation finished'

    # ==== Post-Processing Stage ====
    def _energy_update(self):
        label = 'F'
        self.comp_strain()
        self._field_merge(label)
        self._field_split(label)
        self._strain_init(label)
        self._material_assem(self.F_split)
        self._total_energy(self.F_split)
        # plot(self.F_merge[0,0], interactive=True)
        # plot(self.F[0][0,0], interactive=True)

    def comp_strain(self):
        F_space = self.strain_FS
        self._strain_init('w')
        self.F = [project(self.F[i], F_space_i)
                  for i, F_space_i in enumerate(F_space)]

        # plot(self.F[0][0,0], interactive=True)

        print 'strain computation finished'

    def comp_stress(self):
        # FIXME need careful look
        if not self.F_merge:
            self._energy_update()

        self.P = Function(self.F_merge.function_space())
        L = derivative(self.Pi, self.F_merge, self.F_merge_test)
        # Pi_der = diff(self.Pi, self.F_merge)
        # self.P = project(Pi_der, self.F_merge.function_space())

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)

        # L = inner(Pi_der, self.F_merge_test)
        a = inner(self.F_merge_test, self.F_merge_trial)
        #
        # int_L_li = [L*dx(i) for i in range(self.material_num)]
        int_a_li = [a*dx(i) for i in range(self.material_num)]
        #
        # L = sum(int_L_li)
        a = sum(int_a_li)
        # self.P = project(Pi_der, self.F_merge.function_space())
        solve(a == L, self.P)
        # solve(L == 0, self.P, self.bc, J=a,
        #       form_compiler_parameters=ffc_options)

        # Pi_der = derivative(Pi, self.F[0], F_test)
        # Pi_der = diff(Pi, self.F[0])
        # self.P = project(Pi_der, F_space[0])

        print 'stress computation finished'

        plot(self.P[0,0], interactive=True)

    def avg_merge_strain(self):
        if not self.F_merge:
            self._energy_update()

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        d = self.geom_dim
        mat_num = self.material_num

        if self.multi_field_label:
            F_merge_dim = self.F_merge.shape()[0]
            F_merge_avg = np.zeros((F_merge_dim, 1))
            for i in range(F_merge_dim):
                int_li = [self.F_merge[i]*dx(k) for k in range(mat_num)]
                F_merge_avg[i, 0] = assemble(sum(int_li))
        else:
            F_merge_avg = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    int_li = [self.F_merge[i, j]*dx(k) for k in range(
                            mat_num)]
                    F_merge_avg[i, j] = assemble(sum(int_li))

        print 'average merge strain computation finished'

        print F_merge_avg
        return F_merge_avg

    def avg_merge_stress(self):
        if not self.F_merge:
            self._energy_update()

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        d = self.geom_dim
        mat_num = self.material_num

        P_i = range(mat_num)

        for i in range(mat_num):
            P_i[i] = diff(self.material[i].psi, self.F_merge)
            # P_i[i] = diff(self.material[i].psi, self.F_split[0])
            # P_i[i] = diff(self.material[i].psi, self.F_bar[0])
            # P_i[i] = derivative(self.material[i].psi, self.F_merge, self.F_merge_test)

        # int_li = [P_i[k]*dx(k) for k in range(mat_num)]
        # P_merge_avg = assemble(sum(int_li))

        if self.multi_field_label:
            F_merge_dim = self.F_merge.shape()[0]
            P_merge_avg = np.zeros((F_merge_dim, 1))
            for i in range(F_merge_dim):
                int_li = [P_i[k][i]*dx(k) for k in range(mat_num)]
                P_merge_avg[i, 0] = assemble(sum(int_li))
        else:
            P_merge_avg = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    int_li = [P_i[k][i, j]*dx(k) for k in range(mat_num)]
                    P_merge_avg[i, j] = assemble(sum(int_li))

        # Pi_der = derivative(self.Pi, self.F_split[0], FF_bar_trial)
        # P_merge_avg = assemble(Pi_der)

        print 'average merge stress computation finished'

        print P_merge_avg

        return P_merge_avg

    def tangent_moduli(self, material_energy):
        # todo calculate the tangent moduli from energy
        pass

    def effective_moduli_2(self):
        if not self.F_merge:
            self._energy_update()

        FF = self.F_merge
        psi_m_ = self.material[0].psi
        psi_i_ = self.material[1].psi

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
        c = FF_bar_test[i, j]*C_i_[i, j, k, l]*FF_bar_trial[k, l]*dx(1) + \
            FF_bar_test[i, j]*C_m_[i, j, k, l]*FF_bar_trial[k, l]*dx(0)
        cc = assemble(c)
        CC = cc.array()

        print CC

        TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
        F_bar_2_trial = TrialFunction(TFS_R)
        L2_der = derivative(self.F_w, self.F_bar, F_bar_2_trial)
        B2 = assemble(L2_der)

        LTKL2 = self.sensitivity(B2)

        print LTKL2

        print CC - LTKL2
        return CC - LTKL2

    def sensitivity(self, B):
        w_test = self.v_merge
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

        L_assign = Function(self.w_merge.function_space())
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

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    # if multi field bc should match
    F_bar = [0.9, 0., 0., 1.]
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [cauchy_green_with_macro],
                            [strain_space], 0)

    comp.input([F_bar], [w])
    comp.comp_fluctuation()

    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    comp.effective_moduli_2()

