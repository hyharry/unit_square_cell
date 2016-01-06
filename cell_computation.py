# coding=utf-8

from dolfin import *
import numpy as np
from cell_geom import PeriodicBoundary_no_corner
from cell_material import Material

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
                 multi_field_flag):
        """
        Initialize cell properties such as geometry, material, boundary,
        post processing space, and switch of multi field

        :param cell: geometry
        :param bc: boundary conditions
        :param material_li: [mat1, mat2, ...]
        :param strain_FS_li: post processing spaces
        :param multi_field_flag: 1: multi, 0: uni

        ** Attention: multi_field_flag should be set currently, when uni
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

        self.comp_state = None

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
        self.multi_field_flag = multi_field_flag
        self.material_num = len(material_li)

        # F for Post Processing
        self.material_post = [Material(mat_i.energy_func, mat_i.para_list,
                                       mat_i.invar_gen_li.keys(),
                                       mat_i.invar_gen_li.values())
                              for mat_i in material_li]
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
        self.field_num = len(w_li)

        self.comp_state = 'pre'
        self._field_merge(self.comp_state)
        self._field_split(self.comp_state)
        self._strain_init(self.comp_state)

    def _F_bar_init(self, F_bar_li):
        """
        Macro field input, initialize F_bar
        Mind that if multiple field, multiple field input will require

        :param F_bar_li: !! only constant macro field input are supported !!
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

    def _field_merge(self, state):
        if state is 'pre':
            if self.multi_field_flag:
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
        elif state is 'post':
            if self.multi_field_flag:
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
            raise Exception('wrong state for merge!')

    def _field_split(self, state):
        if state is 'pre':
            if self.multi_field_flag:
                self.w_split = list(split(self.w_merge))
            else:
                self.w_split = [self.w_merge]
        elif state is 'post':
            if self.multi_field_flag:
                self.F_split = list(split(self.F_merge))
            else:
                self.F_split = [self.F_merge]
        else:
            raise Exception('wrong state for split!')

    def _strain_init(self, state):
        generator_li = self.strain_gen
        if state is 'pre':
            self.F = [gen(self.F_bar[i], self.w_split[i])
                      for i, gen in enumerate(generator_li)]
        elif state is 'post':
            self.F = [self.F_bar[i] + self.F_split[i]
                      for i in range(self.field_num)]
        else:
            raise Exception('wrong state for strain init!')

    # ==== Pre-Processing Stage ====
    def _material_assem(self, F, material_list):
        for i in range(self.material_num):
            material_list[i](F)

        # print id(self.material[0].invar[0])

    def _total_energy(self, F, material_list):
        # Please be careful about the ordering dx(0) -> matrix
        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)

        self._material_assem(F, material_list)

        int_i_li = [mat.psi*dx(i)
                    for i, mat in enumerate(material_list)]
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
        self._total_energy(self.F, self.material)

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
        self.comp_strain()
        self._field_merge(self.comp_state)
        self._field_split(self.comp_state)
        self._strain_init(self.comp_state)
        self._total_energy(self.F_split, self.material_post)
        # plot(self.F_merge[0,0], interactive=True)
        # plot(self.F[0][0,0], interactive=True)

    def comp_strain(self):
        F_space = self.strain_FS
        self.F = [project(self.F[i], F_space_i)
                  for i, F_space_i in enumerate(F_space)]

        self.comp_state = 'post'

        # plot(self.F[0][0,0], interactive=True)

        print 'strain computation finished'

    def comp_stress(self):
        if not self.F_merge:
            self._energy_update()

        P = Function(self.F_merge.function_space())

        L = derivative(self.Pi, self.F_merge, self.F_merge_test)

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        a = inner(self.F_merge_test, self.F_merge_trial)
        int_a_li = [a*dx(i) for i in range(self.material_num)]
        a = sum(int_a_li)

        solve(a == L, P)

        print 'stress computation finished'

        plot(P[0,0], interactive=True)

    def avg_merge_strain(self):
        if not self.F_merge:
            self._energy_update()

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        d = self.geom_dim
        mat_num = self.material_num

        if self.multi_field_flag:
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

        # 1. Method: direct make derivative to global energy and use Constant
        # Function as TestFunction. Result is a Vector. No use of mat_num
        # explicitly
        TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
        F_const_test = TestFunction(TFS_R)

        L = derivative(self.Pi, self.F_merge, F_const_test)

        P_merge_avg = assemble(L).array()

        # 2. Method: make derivative to local energy and integrate then
        # assemble, mind that also use Constant Function as TestFunction.
        # Result is a Vector. derivative() seems to stack the DoF into a column!
        # dx = Measure('dx', domain=self.cell.mesh,
        #              subdomain_data=self.cell.domain)
        # mat_num = self.material_num
        #
        # P_i = range(mat_num)
        #
        # TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
        # F_const_test = TestFunction(TFS_R)
        # for i in range(mat_num):
        #     P_i[i] = derivative(self.material[i].psi, self.F_merge, F_const_test)
        #
        # int_li = [P_i[k]*dx(k) for k in range(mat_num)]
        # P_merge_avg = assemble(sum(int_li)).array()

        # 3. Method: make diff to local energy and integrate finally
        # assemble. Without using Constant TestFunction. Result maybe a
        # tensor. diff() conserve the tensor structure. The reason may due to
        # the integration
        # dx = Measure('dx', domain=self.cell.mesh,
        #              subdomain_data=self.cell.domain)
        # d = self.geom_dim
        # mat_num = self.material_num
        #
        # P_i = range(mat_num)
        # for i in range(mat_num):
        #     P_i[i] = diff(self.material[i].psi, self.F_merge)
        #
        # if self.multi_field_flag:
        #     F_merge_dim = self.F_merge.shape()[0]
        #     P_merge_avg = np.zeros((F_merge_dim, 1))
        #     for i in range(F_merge_dim):
        #         int_li = [P_i[k][i]*dx(k) for k in range(mat_num)]
        #         P_merge_avg[i, 0] = assemble(sum(int_li))
        # else:
        #     P_merge_avg = np.zeros((d, d))
        #     for i in range(d):
        #         for j in range(d):
        #             int_li = [P_i[k][i, j]*dx(k) for k in range(mat_num)]
        #             P_merge_avg[i, j] = assemble(sum(int_li))

        print 'average merge stress computation finished'

        print P_merge_avg

        return P_merge_avg

    def avg_merge_moduli(self):
        if not self.F_merge:
            self._energy_update()

        # Using derivative() direct on self.Pi is a more compact way!! and
        # they yield the same result!! : integrate then derivative
        TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
        F_const_trial = TrialFunction(TFS_R)
        F_const_test = TestFunction(TFS_R)

        dPi_dF = derivative(self.Pi, self.F_merge, F_const_test)
        ddPi_dF = derivative(dPi_dF, self.F_merge, F_const_trial)
        C_avg = assemble(ddPi_dF)

        print 'average merge moduli computation finished'

        return C_avg.array()

    def effective_moduli_2(self):
        if not self.F_merge:
            self._energy_update()

        C_avg = self.avg_merge_moduli()

        TFS_R = TensorFunctionSpace(self.cell.mesh, 'R', 0)
        F_bar_trial = TrialFunction(TFS_R)
        L2 = derivative(self.F_w, self.F_bar, F_bar_trial)
        B2 = assemble(L2)

        LTKL2 = self.sensitivity(B2)

        # print LTKL2

        print C_avg - LTKL2
        return C_avg - LTKL2

    def sensitivity(self, B):
        w_test = self.v_merge
        J = self.J
        bc = self.bc
        vec_dim = (w_test.ufl_shape[0] if w_test.ufl_shape else 1)

        # Assemble K symmetrically
        f = Constant((0.,)*vec_dim)
        b = inner(w_test, f) * dx
        K_a, L_a = assemble_system(J, b, bc)

        # Assemble L
        F_merge_len = sum(self.F_merge.ufl_shape)
        rows = []
        for bc_i in bc:
            rows.extend(bc_i.get_boundary_values().keys())
        cols = range(F_merge_len)
        vals = [0.]*F_merge_len
        rs = np.array(rows, dtype=np.uintp)
        cs = np.array(cols, dtype=np.uintp)
        vs = np.array(vals, dtype=np.float_)
        for i in rs:
            B.setrow(i, cs, vs)
        B.apply('insert')
        L = B.array()

        LTKL = np.zeros((F_merge_len, F_merge_len))
        L_assign = Function(self.w_merge.function_space())
        x = Vector()

        for i in range(F_merge_len):
            L_assign.vector().set_local(L[:, i])
            solve(K_a, x, L_assign.vector())
            LTKL[:, i] = L.T.dot(x.array())
        return LTKL


def deform_grad_with_macro(F_bar, w_component):
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
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
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

