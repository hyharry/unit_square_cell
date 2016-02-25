# coding=utf-8
# Copyright (C) 2016 Yi Hu
# python 2.7, FEniCS 1.6.0
"""
Micro scale computation for composite material under multi fields

Class: MicroComputation (pre-processing, fe formulation,
        solution, post-processing, view results, write and output results)
Function: field_merge, field_split, set_field, extend_strain,
        deformation_grad_with_macro

"""

from dolfin import *
import numpy as np
from cell_material import Material
import cell_geom as geom

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True,
               "eliminate_zeros": True,
               "precompute_basis_const": True,
               "precompute_ip_const": True}

# WARNING: Linear algebra backend setting should be done before any
# computation and initialization, specific storage method according to the
# backend will be used. If not done, error of 'down_cast' will be raised

parameters['linear_algebra_backend'] = 'PETSc'
# parameters.update({'linear_algebra_backend': 'Eigen'})

# Solver parameters for the fluctuation solving stage
solver_parameters = {}
# Solver parameters for post processing
post_solver_parameters = {}


class MicroComputation(object):
    """
    Unit Cell Computation both 2d and 3d, solve fluctuation, averaged strain
    stress, moduli, homogenized moduli, view results
    """

    def __init__(self, cell, material_li,
                 strain_gen_li, strain_FS_li):
        """
        Initialize cell properties such as geometry, material, boundary,
        post processing space, and switch of multi field

        :param cell: (UnitCell) geometry
        :param material_li: (list of Materials) [mat1, mat2, ...]
        :param strain_gen_li: (list of Functions) [deform_grad_with_macro, ...]
        :param strain_FS_li: (list of FunctionSpaces) strain post processing
        spaces

        """
        self.cell = cell
        self.material = material_li
        self.strain_gen = strain_gen_li
        self.strain_FS = strain_FS_li

        self.w = None
        self.F_bar = None
        self.F = None

        self.Pi = None
        self.F_w = None
        self.J = None

        # Geometry Dimension 2D
        self.field_num = None
        self.geom_dim = cell.dim

        self.w_merge = None
        self.w_split = None
        # Test function
        self.v_merge = None
        # Trial function
        self.dw_merge = None
        # Boundary Condition
        self.bc = None

        self.material_num = len(material_li)
        # Duplicated Material for post processing, save time and manipulation
        # for post processing
        self.material_post = [Material(mat_i.energy_func, mat_i.para_list,
                                       mat_i.invar_gen_li.keys(),
                                       mat_i.invar_gen_li.values())
                              for mat_i in material_li]

        # F for Problem Formulation
        self.F_merge = None
        self.F_merge_test = None
        self.F_merge_trial = None
        self.F_split = None

        self.F_bar_merge = None
        self.F_bar_merge_test = None
        self.F_bar_merge_trial = None
        self.F_bar_split = None

        self.strain_const_FS = None

        self.P_merge = None

    def input(self, F_bar_li, w_li):
        """
        Input FunctionSpace for F and w to complete initialization, update
        instance members

        F_bar_li and w_li should be updated for each cell and each time step
        values from the previous step is inherited

        :param F_bar_li: (list of lists) macro F, each term in list is written
                        as list, [F_bar, E_bar, T_bar, ...],
                        F_bar = [F11,F12,F21,F22]
        :param w_li: (list of dolfin Functions) field list, [w, e, ...]

        """
        assert isinstance(F_bar_li, list) and isinstance(w_li, list)
        self._F_bar_init(F_bar_li)
        self.w = w_li
        self.field_num = len(w_li)
        (self.w_merge, self.v_merge, self.dw_merge, self.w_split) \
            = set_field(w_li)
        (self.F_bar_merge, self.F_bar_merge_test, self.F_bar_merge_trial,
         self.F_bar_split) = set_field(self.F_bar)
        self.F = extend_strain(self.F_bar_split, self.w_split, self.strain_gen)
        # Clear self.F_merge for Post-Processing
        self.F_merge = None

    def _F_bar_init(self, F_bar_li):
        """
        Macro field input, initialize F_bar, called by input()

        :param F_bar_li: F_bar_li should be a list of F_bar, each entry is
                        furthermore a list representing input from each field

        :return self.F_bar: [Function for F, Function for M, Function for T,...]

        """
        assert isinstance(F_bar_li[0], list)
        self.F_bar = [self._li_to_func(F_bar_field_i)
                      for F_bar_field_i in F_bar_li]

    def _li_to_func(self, F_li):
        """
        Transform list into Constant Function, called by _F_bar_init()

        :param F_li: (list) F_li = [F11,F12,F21,F22]

        :return: (dolfin Function) constant value over mesh

        """
        dim = self.geom_dim
        F_dim = len(F_li)
        if F_dim == 1:
            FS = FunctionSpace(self.cell.mesh, 'R', 0)
            F_ex = Expression('F', F=F_li[0])
            return project(F_ex, FS)
        elif F_dim == dim:
            VFS = VectorFunctionSpace(self.cell.mesh, 'R', 0)
            if dim == 2:
                F_ex = Expression(('F1', 'F2'), F1=F_li[0], F2=F_li[1])
            else:
                F_ex = Expression(('F1', 'F2', 'F3'), F1=F_li[0], F2=F_li[1],
                                  F3=F_li[2])
            return project(F_ex, VFS)
        elif F_dim == dim ** 2:
            TFS = TensorFunctionSpace(self.cell.mesh, 'R', 0)
            if dim == 2:
                F_ex = Expression((("F11", "F12"),
                                   ("F21", "F22")),
                                  F11=F_li[0], F12=F_li[1],
                                  F21=F_li[2], F22=F_li[3])
            else:
                F_ex = Expression((("F11", "F12", "F13"),
                                   ("F21", "F22", "F23"),
                                   ("F31", "F32", "F33")),
                                  F11=F_li[0], F12=F_li[1], F13=F_li[2],
                                  F21=F_li[3], F22=F_li[4], F23=F_li[5],
                                  F31=F_li[6], F32=F_li[7], F33=F_li[8])
            return project(F_ex, TFS)
        else:
            raise Exception('Please Input Right Dimension')

    # ==== Pre-Processing Stage ====
    def _total_energy(self, F, material_list):
        """
        Energy over the whole cell, called by _fem_formulation_composite()

        integrate and sum up for composites， ordering should be considered,
        dx(0) -> composite matrix

        :param F: (list of dolfin Functions) extended strain, [F,E,...]
        :param material_list: (list of Materials) [matrix,component1,...]

        :return: self.Pi (energy) is updated

        """
        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)

        material_assem(F, material_list)

        int_i_li = [mat.psi * dx(i)
                    for i, mat in enumerate(material_list)]
        self.Pi = sum(int_i_li)

    def _bc_fixed_corner(self):
        """
        Generate dirichlet boundary condition for fe problem, all the corners
        are fixed, invoked by _fem_formulation_composite()

        :return: updated self.bc

        """
        bc = []
        corners = geom.compiled_corner_subdom(self.geom_dim)
        dim = self.w_merge.shape()
        if dim:
            fixed_corner = Constant((0,) * dim[0])
        else:
            fixed_corner = Constant(0.)
        for c in corners:
            bc.append(DirichletBC(self.w_merge.function_space(),
                                  fixed_corner, c, method='pointwise'))
        self.bc = bc

    def _fem_formulation_composite(self):
        """
        Formulate FE problem, using derivative() to energy

        :return: updated self.F_w -> linear form
                         self.J -> bilinear form
                         self.bc -> fixed corner bc

        """
        self._total_energy(self.F, self.material)

        F_w = derivative(self.Pi, self.w_merge, self.v_merge)

        # Compute Jacobian of F
        J = derivative(F_w, self.w_merge, self.dw_merge)

        self.F_w = F_w
        self.J = J
        if self.bc is None:
            self._bc_fixed_corner()

    # ==== Solution ====
    def comp_fluctuation(self, print_progress=False, print_solver_info=False):
        """
        Solve fluctuation, solver parameters are set before solving

        :param print_progress: (bool) print detailed solving progress
        :param print_solver_info: (bool) print detailed solver info

        :return: updated self.w_merge

        """
        self._fem_formulation_composite()

        # 1.Method of defining solution: direct set solver_parameters
        # solve(self.F_w == 0, self.w_merge, self.bc, J=self.J,
        #       solver_parameters=solver_parameters,
        #       form_compiler_parameters=ffc_options)

        # 2.Method of defining solution: define variational prob, and update,
        #  and solve
        problem = NonlinearVariationalProblem(self.F_w, self.w_merge, self.bc,
                                              self.J)
        solver = NonlinearVariationalSolver(problem)
        solver_setting(solver, solver_parameters,
                       print_progress=print_progress,
                       print_solver_info=print_solver_info)
        solver.solve()

        print 'fluctuation computation finished'

    # ==== Post-Processing Stage ====
    def _energy_update(self):
        """
        Post-processing stage, update self.Pi using computed strain, invoked
        for all other post processing methods

        :return: updated self.Pi

        """
        self.comp_strain()
        (self.F_merge, self.F_merge_test, self.F_merge_trial, self.F_split) = \
            set_field(self.F)
        self._total_energy(self.F_split, self.material_post)
        # plot(self.F_merge[0,0], interactive=True)
        # plot(self.F[0][0,0], interactive=True)

    def _const_strain_FS_init(self):
        """
        Generate constant strain FunctionSpace for Post-Processing

        multi field: FS.shape = (n,)
        uni field: FS.shape = (2,2)

        :return: updated self.strain_const_FS

        """
        if self.field_num > 1:
            dim = self.F_merge.shape()[0]
            FS = FunctionSpace(self.cell.mesh, 'R', 0)
            RFS = MixedFunctionSpace([FS] * dim)
        else:
            RFS = TensorFunctionSpace(self.cell.mesh, 'R', 0)

        self.strain_const_FS = RFS

    def comp_strain(self):
        """
        Compute strain using project()

        :return: updated self.F

        """
        F_space = self.strain_FS
        self.F = [project(self.F[i], F_space_i)
                  for i, F_space_i in enumerate(F_space)]

        print 'strain computation finished'

    def comp_stress(self):
        """
        Compute stress at each node.

        use derivative() w.r.t self.F_merge in self.Pi -> linear form
        integrate F_merge_test*F_merge_trial over domain -> bili form
        solve use default FEniCS linear solver

        :return: updated self.P_merge

        """
        if not self.F_merge:
            self._energy_update()

        P = Function(self.F_merge.function_space())

        L = derivative(self.Pi, self.F_merge, self.F_merge_test)

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        a = inner(self.F_merge_test, self.F_merge_trial)
        int_a_li = [a * dx(i) for i in range(self.material_num)]
        a = sum(int_a_li)

        solve(a == L, P, solver_parameters=post_solver_parameters)

        self.P_merge = P

        print 'stress computation finished'

    def avg_merge_strain(self):
        """
        Average merged strain with explicit integration

        :return: F_merge_avg
        :rtype: dolfin Function

        """
        if not self.F_merge:
            self._energy_update()

        dx = Measure('dx', domain=self.cell.mesh,
                     subdomain_data=self.cell.domain)
        d = self.geom_dim
        mat_num = self.material_num

        # Not that left and right multiply with constant functions is also
        # possible
        if self.field_num > 1:
            F_merge_dim = self.F_merge.shape()[0]
            F_merge_avg = np.zeros((F_merge_dim, 1))
            for i in range(F_merge_dim):
                int_li = [self.F_merge[i] * dx(k) for k in range(mat_num)]
                F_merge_avg[i, 0] = assemble(sum(int_li))
        else:
            F_merge_avg = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    int_li = [self.F_merge[i, j] * dx(k) for k in range(
                        mat_num)]
                    F_merge_avg[i, j] = assemble(sum(int_li))

        print 'average merge strain computation finished'

        # print F_merge_avg
        return F_merge_avg

    def avg_merge_stress(self):
        """
        Average merged stress, 3 methods.

        here the consistent one with derivative() is used

        :return: P_merge_avg
        :rtype: numpy array

        """
        if not self.F_merge:
            self._energy_update()
        if not self.strain_const_FS:
            self._const_strain_FS_init()

        # 1. Method: direct make derivative to global energy and use Constant
        # Function as TestFunction. Result is a Vector. No use of mat_num
        # explicitly
        F_const_test = TestFunction(self.strain_const_FS)

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
        # F_const_test = TestFunction(self.strain_const_FS)
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
        # if self.field_num > 1:
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

        # print P_merge_avg
        return P_merge_avg

    def avg_merge_moduli(self):
        """
        Average merged moduli, derivative() of self.Pi w.r.t. self.F_merge,
        and assemble

        :return: C_avg
        :rtype: numpy array

        """
        if not self.F_merge:
            self._energy_update()
        if not self.strain_const_FS:
            self._const_strain_FS_init()

        F_const_trial = TrialFunction(self.strain_const_FS)
        F_const_test = TestFunction(self.strain_const_FS)

        dPi_dF = derivative(self.Pi, self.F_merge, F_const_test)
        ddPi_dF = derivative(dPi_dF, self.F_merge, F_const_trial)
        C_avg = assemble(ddPi_dF)

        print 'average merge moduli computation finished'

        return C_avg.array()

    def effective_moduli_2(self):
        """
        Effective moduli calculation according to homogenization method

        :return: C_avg-LTKL2 <=> C_eff
        :rtype: numpy array

        """
        if not self.F_merge:
            self._energy_update()
        if not self.strain_const_FS:
            self._const_strain_FS_init()

        C_avg = self.avg_merge_moduli()

        F_bar_trial = TrialFunction(self.strain_const_FS)
        L2 = derivative(self.F_w, self.F_bar_merge, F_bar_trial)
        B2 = assemble(L2)

        LTKL2 = self.sensitivity(B2)

        # print LTKL2

        print C_avg - LTKL2
        return C_avg - LTKL2

    def sensitivity(self, B):
        """
        Sensitivity matrix calculation. Assemble linear and bilinear form
        symetrically. Solve use the default FEniCS linear solver

        :param B: (numpy array) linear form w.r.t fluctuation as well as
                                linear form w.r.t macro deformation

        :return: LTKL
        :rtype: numpy array

        """
        w_test = self.v_merge
        J = self.J
        bc = self.bc
        vec_dim = (w_test.ufl_shape[0] if w_test.ufl_shape else 1)

        # Assemble K symmetrically
        # !!Notice Here only PETScSolver is used!!
        K_a = PETScMatrix()
        f = Constant((0.,) * vec_dim)
        b = inner(w_test, f) * dx
        K_a, L_a = assemble_system(J, b, bc, A_tensor=K_a)

        # Assemble L
        if self.field_num == 1:
            F_merge_len = self.F_merge.ufl_shape[0]**2
        else:
            F_merge_len = sum(self.F_merge.ufl_shape)
        rows = []
        for bc_i in bc:
            rows.extend(bc_i.get_boundary_values().keys())
        cols = range(F_merge_len)
        vals = [0.] * F_merge_len
        rs = np.array(rows, dtype=np.uintp)
        cs = np.array(cols, dtype=np.uintp)
        vs = np.array(vals, dtype=np.float_)
        for i in rs:
            B.setrow(i, cs, vs)
        B.apply('insert')
        L = B.array()

        LTKL = np.zeros((F_merge_len, F_merge_len))
        L_assign = Function(self.w_merge.function_space())
        x = PETScVector()
        # x = Vector()

        if post_solver_parameters:
            lin_sol_meth = post_solver_parameters['linear_solver']
            if lin_sol_meth in lu_solver_methods().keys():
                solver = LUSolver(K_a, lin_sol_meth)
            elif lin_sol_meth in krylov_solver_methods().keys():
                if post_solver_parameters.has_key('preconditioner'):
                    iter_pre_cond = post_solver_parameters['preconditioner']
                    solver = KrylovSolver(K_a, lin_sol_meth, iter_pre_cond)
                else:
                    solver = KrylovSolver(K_a, lin_sol_meth)
        else:
            solver = LUSolver(K_a)

        for i in range(F_merge_len):
            L_assign.vector().set_local(L[:, i])
            b = as_backend_type(L_assign.vector())
            # print type(b)
            solver.solve(x, b)
            LTKL[:, i] = L.T.dot(x.array())
        return LTKL

    def view_fluctuation(self, field_label=1):
        """
        View fluctuation of different field

        :param field_label: (int) field_label 1->displacement

        :return: plot

        """
        if field_label is 1:
            plot(self.w_split[0], mode='displacement', interactive=True)
        else:
            label = field_label - 1
            w = self.w_split[label]
            if not w.shape():
                plot(w, mode='color', interactive=True)
            elif len(w.shape()) is 1:
                if w.shape()[0] != self.geom_dim:
                    print 'plot dimension does not match'
                    return
                plot(w, mode='displacement', interactive=True)
            else:
                print 'this is a tensor field'

    def view_displacement(self, field_label=1):
        """
        View displacement of different field

        :param field_label: (int) field_label 1->displacement

        :return: plot

        """
        if field_label is 1:
            F_bar = self.F_bar[0]
            if self.geom_dim == 2:
                coord = Expression(('x[0]', 'x[1]'))
            else:
                coord = Expression(('x[0]', 'x[1]', 'x[2]'))
            plot(self.w_split[0] + dot(F_bar, coord), mode='displacement',
                 interactive=True)
        else:
            label = field_label - 1
            w = self.w_split[label]
            dim = w.geometric_dimension()
            F_bar = self.F_bar[0]
            if dim == 1:
                coord = Expression('x[0]')
            elif dim == 2:
                coord = Expression(('x[0]', 'x[1]'))
            else:
                coord = Expression(('x[0]', 'x[1]', 'x[2]'))
            if not w.shape():
                plot(w + dot(F_bar, coord), mode='color', interactive=True)
            elif len(w.shape()) is 1:
                if w.shape()[0] != self.geom_dim:
                    print 'plot dimension does not match'
                    return
                plot(w + dot(F_bar, coord), mode='displacement',
                     interactive=True)
            else:
                print 'this is a tensor field'

    def view_post_processing(self, label, component):
        """
        Plot strain or stress from Post-Processing

        :param label: (int) 'stress' or 'strain'
        :param component: (int) component label should be label for the merged
        field

        :return:
        """
        # FIXME uni field post processing the index should be reconsidered
        if label is 'strain':
            if not self.F_merge:
                self._energy_update()
            plot(self.F_merge[component], mode='color', interactive=True)
        elif label is 'stress':
            if not self.P_merge:
                self.comp_stress()
            plot(self.P_merge[component], mode='color', interactive=True)
        else:
            raise Exception('invalid output name')

    def write_output(self, label):
        """
        Write output in a file for post processing of other kinds

        :param label: field label for writing output
        :return: file
        """
        # TODO output
        pass


def set_solver_parameters(non_lin_method, lin_method=None,
                          linear_solver='default', preconditioner=None,
                          para=None):
    """
    Assistance function to set global solver parameters

    Some parameters should be tuned inside this method. Please keep in mind
    that some parameters set could differ, when backend changes

    :param non_lin_method: (string) name of non linear solver
                            from ['snes', 'non_lin_newton]
    :param lin_method: (string) name of linear solver
                        from ['direct', 'iterative']
                        if (None) default is used

    :param linear_solver: (string) name of newton solver
                            if 'direct', from ['default', 'mumps', 'petsc',
                            'umfpack]
                            if 'iterative', from ['biccgstab', 'cg',
                            'default', 'gmres', 'minres', 'richardson', 'tfqmr']

    :param preconditioner: (string) from ['amg', 'default', 'hypre_amg',
                                        'hypre_euclid', 'hypre_parasails',
                                        'icc', 'ilu', 'none', 'petsc_amg', 'sor']

    :param para: (dict) parameters for the specific solver, e.g.
                krylov_solver (iterative solver):
                {"absolute_tolerance": 1E-9, "relative_tolerance": 1E-7,
                "maximum_iterations": 1000,
                "gmres": {"restart": 30},
                "preconditioner": {"ilu": {"fill_level": 0}}}

                newton_solver:
                {"absolute_tolerance": 1E-8, "relative_tolerance": 2E-7,
                "maximum_iterations": 25, "relaxation_parameter": 1.}

                snes_solver:
                {"linear_solver": "lu", "line_search": "bt",
                "maximum_iterations": 50, "report": True,
                "error_on_nonconvergence": False,}

    :return: (dict) global nested dictionary of parameters setting

    Reference: The formal list of linear solve can be viewed in the following
    website of PETSc
    `<http://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html>`_

    for Eigen3 see the following
    `<http://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html>`_

    """
    # Global solve_parameters parameters
    global solver_parameters

    print '.-------------------.'
    print '| Solver Parameters |'
    print '.-------------------.'

    # Set non lin solver and parameters
    if non_lin_method == 'snes':
        solver_parameters = {"nonlinear_solver": "snes",
                             "snes_solver": {}}
        if para:
            solver_parameters['snes_solver'].update(para)
        solver_type = 'snes_solver'
    elif non_lin_method == 'non_lin_newton':
        solver_parameters = {"nonlinear_solver": "newton",
                             "newton_solver": {}}
        if para:
            solver_parameters["newton_solver"].update(para)
        solver_type = 'newton_solver'
    else:
        raise Exception('Not Defined Nonlinear Variational Solver! Please')

    # Set linear solver
    lin_method_dict = {'direct': lu_solver_methods().keys(),
                       'iterative': krylov_solver_methods().keys()}

    if lin_method in lin_method_dict.keys():
        print lin_method + ' method is used'
    elif lin_method is None:
        print 'Default Setting is used'
    else:
        raise Exception('Linear Solver Method Not Valid!')

    if lin_method is not None:
        if linear_solver in lin_method_dict[lin_method]:
            solver_parameters[solver_type]["linear_solver"] = linear_solver
        else:
            raise Exception('Error in Newton Method Setup')

    if lin_method is 'iterative':
        # Preconditioner for iterative solver
        if preconditioner in krylov_solver_preconditioners().keys():
            solver_parameters["newton_solver"]["preconditioner"] = \
                preconditioner
        else:
            print 'a valid preconditioner should be provided'


def set_post_solver_parameters(lin_method=None, linear_solver='default',
                               preconditioner=None, para=None):
    global post_solver_parameters

    print '+----------------------------+'
    print '| Post Processing Parameters |'
    print '+----------------------------+'

    # Linear solver types
    lin_method_dict = {'direct': lu_solver_methods().keys(),
                       'iterative': krylov_solver_methods().keys()}

    if lin_method in lin_method_dict.keys():
        print lin_method + ' method is used'
    elif lin_method is None:
        print 'Default Setting is used'
    else:
        raise Exception('Linear Solver Method Not Valid!')

    if lin_method is not None:
        if linear_solver in lin_method_dict[lin_method]:
            post_solver_parameters = {"linear_solver": linear_solver}
        else:
            raise Exception('Error in Newton Method Setup')

    if lin_method is 'iterative':
        # Preconditioner for iterative solver
        if preconditioner in krylov_solver_preconditioners().keys():
            post_solver_parameters["preconditioner"] = preconditioner
        else:
            print 'a valid preconditioner should be provided'

    if para is not None:
        post_solver_parameters.update(para)


def solver_setting(solver, solver_para,
                   print_solver_info=False, print_progress=False):
    """
    Solver Setter when a solver is at hand

    :param solver: an instance of NonlinearVariationalSolver
    :param solver_para: (nested dict) initialized from set_solver_parameters
    :param print_solver_info: (bool) print all solver parameters possibilities
    :param print_progress: (bool) print solver progress

    :return: updated solver instance

    """
    if print_solver_info:
        info(solver.parameters, True)

    solver.parameters.update(solver_para)

    if print_progress:
        set_log_level(PROGRESS)


def field_merge(func_li):
    """
    Merge field for derivation

    :param func_li: (list of dolfin Functions) [Func1, Func2, ...]

    :return: func_merge, func_merge_test, func_merge_trial
    :rtype: (tuple of dolfin Functions)

    """
    # Determine Function Space
    if len(func_li) > 1:
        FS_li = [func_i.function_space() for func_i in func_li]
        MFS = MixedFunctionSpace(FS_li)
        FS = MFS
        func_merge = Function(FS)
        for i, func_i in enumerate(func_li):
            assign(func_merge.sub(i), func_i)
    else:
        FS = func_li[0].function_space()
        func_merge = func_li[0]

    # Generate Functions
    func_merge_test = TestFunction(FS)
    func_merge_trial = TrialFunction(FS)

    return func_merge, func_merge_test, func_merge_trial


def field_split(merged_func, field_num):
    """
    Split the merged function to get dependency

    :param merged_func: (dolfin Function)
    :param field_num: if uni-field no need to split

    :return: list of dolfin split Functions

    """
    if field_num > 1:
        return list(split(merged_func))
    else:
        return [merged_func]


def set_field(func_li):
    """
    One-stand Function dependency setting (merge and split in one step)

    :param func_li: (list of dolfin Functions) w or F to merge and split,
                    or other Function_list

    :return: f_merge, f_merge_test, f_merge_trial, f_split
    :type: tuple

    """
    f_merge, f_merge_test, f_merge_trial = field_merge(func_li)
    f_split = field_split(f_merge, len(func_li))
    return f_merge, f_merge_test, f_merge_trial, f_split


def extend_strain(macro_li, func_li, generator_li=None):
    """
    Generate strain and extend strain into a list

    :param macro_li: (list of dolfin Functions) list of macro fields
    :param func_li: (list of dolfin Functions) displacement or other
                    fluctuation variable to derive strain
    :param generator_li: (list of functions) functions using macro field and
                        fluctuation to generate strain

    :return: F, extended strain list
    :rtype: list of dolfin Functions

    """
    if generator_li:
        F = [gen(macro_li[i], func_li[i]) for i, gen in enumerate(generator_li)]
    else:
        F = [macro_i + func_li[i] for i, macro_i in enumerate(macro_li)]
    return F


def material_assem(F, material_list):
    """
    Assemble material energy (psi)

    :param F: (list of dolfin Functions) material energy variables
    :param material_list: (list of Materials)

    :return: updated Material with psi initialized

    """
    for mat_i in material_list:
        mat_i(F)


def deform_grad_with_macro(F_bar, w_component):
    """
    Basic strain generator

    :param F_bar: (dolfin Function) Macro Strain
    :param w_component: (dolfin Function) displacement fluctuation

    :return: complete strain
    :rtype: dolfin function

    """
    return F_bar + grad(w_component)


def test_uni_field():
    """
    Test for Uni Field Problems
    """
    print 'St-Venant Kirchhoff Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    # mesh = Mesh(r"m.xml")
    mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    inc = ce.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  2))

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    # F_bar = [.9, 0., 0., 1.]
    F_bar = [1., 0.5, 0., 1.]
    # F_bar = [1., 0.5, 0., 1.]
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
                            [strain_space])

    comp.input([F_bar], [w])
    comp.comp_fluctuation()
    # comp.view_fluctuation()
    # comp.view_displacement()
    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.view_post_processing('strain', (0,1))
    # comp.view_post_processing('stress', (0,1))
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    comp.effective_moduli_2()


def test_multi_field():
    """
    Test for Multi Field Problem
    """
    print 'Neo-Hookean EAP Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    # mesh = Mesh(r"m.xml")
    mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    inc = ce.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  2))
    FS = FunctionSpace(cell.mesh, "CG", 1,
                       constrained_domain=ce.PeriodicBoundary_no_corner(2))

    # Set materials
    E_m, nu_m, Kappa_m = 2e5, 0.4, 7.
    n = 1000
    # n = 10  # 13.Jan
    E_i, nu_i, Kappa_i = 1000 * E_m, 0.3, n * Kappa_m

    mat_m = ma.neo_hook_eap(E_m, nu_m, Kappa_m)
    mat_i = ma.neo_hook_eap(E_i, nu_i, Kappa_i)
    mat_li = [mat_m, mat_i]

    # Macro Field Boundary
    F_bar = [0.9, 0.,
             0., 1.]
    E_bar = [0., 0.2]

    # Solution Field
    w = Function(VFS)
    el_pot_phi = Function(FS)
    strain_space_w = TensorFunctionSpace(mesh, 'DG', 0)
    strain_space_E = VectorFunctionSpace(mesh, 'DG', 0)

    def deform_grad_with_macro(F_bar, w_component):
        return F_bar + grad(w_component)

    def e_field_with_macro(E_bar, phi):
        # return E_bar + grad(phi)
        return E_bar - grad(phi)

    # Computation Initialization
    comp = MicroComputation(cell, mat_li,
                            [deform_grad_with_macro, e_field_with_macro],
                            [strain_space_w, strain_space_E])

    comp.input([F_bar, E_bar], [w, el_pot_phi])
    comp.comp_fluctuation()
    # comp.view_displacement()
    # comp.view_fluctuation(1)
    # comp.view_fluctuation(2)
    # comp.view_post_processing('stress', 5)
    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    comp.effective_moduli_2()


def test_uni_field_3d():
    """
    Test for Uni Field 3d Problem
    """
    print 'St-Venant Kirchhoff Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = UnitCubeMesh(4, 4, 4)
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    # inc = ce.InclusionRectangle(3, .25, .75, .25, .75, .25, .75)
    inc = ce.InclusionRectangle(3, 0., 1., .25, .75, .25, .75)
    inc_di = {'box': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  3))

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    # if multi field bc should match
    F_bar = [.9, 0.3, 0.,
             0., 1., 0.,
             0., 0., 1.]
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
                            [strain_space])

    comp.input([F_bar], [w])

    set_solver_parameters('snes', 'iterative', 'minres')

    comp.comp_fluctuation(print_progress=False, print_solver_info=False)
    # comp.view_fluctuation()

    # Post-Processing
    # set_post_solver_parameters(lin_method='direct', linear_solver='lu')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='cg',
    #                            preconditioner='hypre_amg')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='gmres',
    #                            preconditioner='ilu')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='minres',
    #                            preconditioner='amg')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='minres')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='gmres')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='tfqmr')
    # set_post_solver_parameters(lin_method='iterative', linear_solver='cg')
    # set_post_solver_parameters(lin_method='iterative',
    #                            linear_solver='richardson')

    # comp._energy_update()
    # comp.comp_strain()
    comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    comp.effective_moduli_2()


def test_multi_field_3d():
    """
    Test for Multi Field 3d Problem
    """
    print 'St-Venant Kirchhoff Material Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = UnitCubeMesh(4, 4, 4)
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    # inc = ce.InclusionRectangle(3, .25, .75, .25, .75, .25, .75)
    inc = ce.InclusionRectangle(3, 0., 1., .25, .75, .25, .75)
    inc_di = {'box': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  3))
    FS = FunctionSpace(cell.mesh, "CG", 1,
                       constrained_domain=ce.PeriodicBoundary_no_corner(3))

    # Set materials
    E_m, nu_m, Kappa_m = 2e5, 0.4, 7.
    n = 1000
    # n = 10  # 13.Jan
    E_i, nu_i, Kappa_i = 1000 * E_m, 0.3, n * Kappa_m

    mat_m = ma.neo_hook_eap(E_m, nu_m, Kappa_m)
    mat_i = ma.neo_hook_eap(E_i, nu_i, Kappa_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    # if multi field bc should match
    F_bar = [.9, 0.3, 0.,
             0., 1., 0.,
             0., 0., 1.]
    E_bar = [0., 0., 0.]

    # Solution Field
    w = Function(VFS)
    el_pot_phi = Function(FS)
    strain_space_w = TensorFunctionSpace(mesh, 'DG', 0)
    strain_space_E = VectorFunctionSpace(mesh, 'DG', 0)

    def deform_grad_with_macro(F_bar, w_component):
        return F_bar + grad(w_component)

    def e_field_with_macro(E_bar, phi):
        # return E_bar + grad(phi)
        return E_bar - grad(phi)

    # Computation Initialization
    comp = MicroComputation(cell, mat_li,
                            [deform_grad_with_macro, e_field_with_macro],
                            [strain_space_w, strain_space_E])

    comp.input([F_bar, E_bar], [w, el_pot_phi])
    comp.comp_fluctuation()
    # comp.view_fluctuation(1)

    # Post-Processing
    # comp._energy_update()
    # comp.comp_strain()
    # comp.comp_stress()
    # comp.avg_merge_strain()
    # comp.avg_merge_stress()
    # comp.avg_merge_moduli()
    comp.effective_moduli_2()


def test_solver():
    """
    Test for Different Solvers
    """
    print 'Solver Test'
    import cell_geom as ce
    import cell_material as ma

    # Set geometry
    mesh = Mesh(r"m.xml")
    # mesh = Mesh(r"m_fine.xml")
    cell = ce.UnitCell(mesh)
    inc = ce.InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_di = {'circle_inc': inc}
    cell.set_append_inclusion(inc_di)
    # cell.view_domain()

    VFS = VectorFunctionSpace(cell.mesh, "CG", 1,
                              constrained_domain=ce.PeriodicBoundary_no_corner(
                                  2))

    # global parameters

    # Set materials
    E_m, nu_m, E_i, nu_i = 10.0, 0.3, 1000.0, 0.3
    mat_m = ma.st_venant_kirchhoff(E_m, nu_m)
    mat_i = ma.st_venant_kirchhoff(E_i, nu_i)
    mat_li = [mat_m, mat_i]

    # Initialize MicroComputation
    F_bar = [1., 0.8, 0., 1.]
    # F_bar = [1., 0.5, 0., 1.]
    # parameters['linear_algebra_backend'] = 'Eigen'
    w = Function(VFS)
    strain_space = TensorFunctionSpace(mesh, 'DG', 0)
    comp = MicroComputation(cell, mat_li, [deform_grad_with_macro],
                            [strain_space])

    comp.input([F_bar], [w])

    # Test PETSc backend
    # set_solver_parameters('snes', 'iterative', 'minres')
    # set_solver_parameters('non_lin_newton', 'iterative', 'cg')
    # set_solver_parameters('non_lin_newton', 'direct', 'default')
    # set_solver_parameters('non_lin_newton')
    # set_solver_parameters('snes')
    # set_solver_parameters('non_lin_newton')

    # Test Eigen backend, backend definition is to be defined before all the
    # initialization (dolfin Function) and computation
    set_solver_parameters('non_lin_newton', lin_method='direct',
                          linear_solver='sparselu')
    # set_solver_parameters('non_lin_newton')

    # info(NonlinearVariationalSolver.default_parameters(), True)

    comp.comp_fluctuation(print_progress=True, print_solver_info=False)


if __name__ == '__main__':
    test_uni_field()
    # test_multi_field()
    # test_uni_field_3d()
    # test_multi_field_3d()
    # test_solver()
