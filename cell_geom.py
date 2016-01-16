# coding=utf-8
"""
2D Unit Cell. 3D problem should be reformulated
"""

from dolfin import *
from math import sqrt, pi
import logging

logging.getLogger('FFC').setLevel(logging.WARNING)


class UnitCell(object):
    def __init__(self, mesh, incl_di=None):
        """
        Generate Unit Cell for Micro Computation

        :param mesh: Mesh entity from dolfin
        :param incl_di: {'inclusion_name': inclusion, ...}
        :return:
        """
        self.mesh = mesh
        self.dim = mesh.geometry().dim()
        self.domain = MeshFunction("size_t", mesh, self.dim)
        # self.boundary = {}

        self.domain.set_all(0)
        self.incl_di = {}
        if incl_di:
            self.set_append_inclusion(incl_di)

    def view_mesh(self):
        plot(self.mesh, interactive=True)

    def view_domain(self):
        plot(self.domain, interactive=True)

    def set_append_inclusion(self, incl_di):
        exist_incl_num = len(self.incl_di)
        k = exist_incl_num + 1
        for inc in incl_di.values():
            inc.mark(self.domain, k)
            k += 1
        self.incl_di.update(incl_di)

    def add_mark_boundary(self, bound_dim):
        """
        Add boundary is not really needed for the current state!!
        all kinds of boundary can be generated

        in fact it is needed when Neumann Boundary is required to impose
        :param bound_dim: dimension of boundary entity
        """
        dim = self.dim
        if bound_dim >= dim:
            raise Exception("invalid boundary dimension")

        boundary = MeshFunction("size_t", self.mesh, bound_dim)
        if bound_dim == 0:
            compiled_boundary = compiled_corner_subdom(dim)
        elif bound_dim == 1:
            compiled_boundary = compiled_line_subdom(dim)
        else:
            compiled_boundary = compiled_face_subdom(dim)

        for i, bound in enumerate(compiled_boundary):
            bound.mark(boundary, i)


def string_template(dim, with_boundary=False, coord_label=None):
    val = [0., 1.]
    str_template = dict()
    if not coord_label:
        coord_label = range(dim)
    for coord_i in coord_label:
        # str_template[coord_i] = "near(x[{coord_i}], {{val:f}}, " \
        #                         "DOLFIN_EPS) &&".format(coord_i=coord_i)
        str_template[coord_i] = "near(x[{coord_i}], {{val:f}}) &&".format(coord_i=coord_i)

    # Expand template using values
    str_template_2 = dict()
    for i in coord_label:
        str_template_2[i] = [str_template[i].format(val=val_i) for val_i in val]

    # Generate string and join
    ke = str_template_2.keys()
    if dim == 1:
        str_template_3 = [i
                          for i in str_template_2[ke[0]]]
    elif dim == 2:
        str_template_3 = [' '.join((i, j))
                          for i in str_template_2[ke[0]]
                          for j in str_template_2[ke[1]]]
    elif dim == 3:
        str_template_3 = [' '.join((i, j, k))
                          for i in str_template_2[ke[0]]
                          for j in str_template_2[ke[1]]
                          for k in str_template_2[ke[2]]]
    else:
        raise Exception("Only 1d, 2d, 3d cases are supported")

    if with_boundary:
        comp_str = [' '.join((stri, "on_boundary")) for stri in str_template_3]
    else:
        comp_str = [stri[:-3] for stri in str_template_3]
    # print comp_str
    return comp_str


def compiled_corner_subdom(dim):
    comp_str = string_template(dim)
    comp_corner_sub = [CompiledSubDomain(stri) for stri in comp_str]
    return comp_corner_sub


def compiled_line_subdom(dim):
    comp_str = []
    if dim == 2:
        comp_str.extend(string_template(1, with_boundary=True, coord_label=[0]))
        comp_str.extend(string_template(1, with_boundary=True, coord_label=[1]))
    elif dim == 3:
        comp_str.extend(string_template(2, with_boundary=True,
                                        coord_label=[0, 1]))
        comp_str.extend(string_template(2, with_boundary=True,
                                        coord_label=[1, 2]))
        comp_str.extend(string_template(2, with_boundary=True,
                                        coord_label=[2, 0]))
    else:
        raise Exception("Only 2d, 3d cases are supported")

    comp_line_sub = [CompiledSubDomain(stri) for stri in comp_str]
    return comp_line_sub


def compiled_face_subdom(dim):
    comp_str = []
    if dim == 3:
        comp_str.extend(string_template(1, with_boundary=True, coord_label=[0]))
        comp_str.extend(string_template(1, with_boundary=True, coord_label=[1]))
        comp_str.extend(string_template(1, with_boundary=True, coord_label=[2]))
    else:
        raise Exception("Only 3d cases are supported")

    comp_face_sub = [CompiledSubDomain(stri) for stri in comp_str]
    return comp_face_sub


class InclusionCircle(SubDomain):
    def __init__(self, dim, *args):
        """
        Two ways to generate Circle Inclusion

        :param args: 1. args[0] = (x_c, y_c), args[1] = r
                     2. args[0] = fraction ration, center is (0.5, 0.5)
        """
        super(InclusionCircle, self).__init__()
        self.dim = dim
        if len(args) > 1:
            if dim != len(args[0]): raise Exception(
                    "please check, dim do not match")
            self.c = args[0]
            self.r = args[1]
        else:
            self.ratio = args[0]
            self.c = (0.5,) * dim
            self.r = sqrt(self.ratio / pi)

    def inside(self, x, on_boundary):
        c = self.c
        r = self.r
        if self.dim == 2:
            d = sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)
        elif self.dim == 3:
            d = sqrt(
                    (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2 + (
                    x[2] - c[2]) ** 2)
        else:
            raise Exception("only 2d or 3d circle inclusion are supported")
        return d < r or near(d, r)


class InclusionRectangle(SubDomain):
    def __init__(self, dim, *args):
        """
        Generate Rectangle Inclusion

        :param args[0]: x_lower_bound
        :param args[1]: x_upper_bound
        :param args[2]: y_lower_bound
        :param args[3]: y_upper_bound
        :param args[4]: z_lower_bound
        :param args[5]: z_upper_bound
        """
        super(InclusionRectangle, self).__init__()
        if len(args)/2 != dim:
            raise Exception("dim does not match")
        self.dim = dim
        self.bound = args

    def inside(self, x, on_boundary):
        if self.dim == 2:
            in_or_not = (self.bound[0] <= x[0] <= self.bound[1] and
                         self.bound[2] <= x[1] <= self.bound[3])
        elif self.dim == 3:
            in_or_not = (self.bound[0] <= x[0] <= self.bound[1] and
                         self.bound[2] <= x[1] <= self.bound[3] and
                         self.bound[4] <= x[2] <= self.bound[5])
        else:
            raise Exception("only 2d or 3d circle inclusion are supported")
        return in_or_not


class PeriodicBoundary_no_corner(SubDomain):
    """
    Periodic boundary both directions (no corner)
    """
    def __init__(self, dim):
        super(PeriodicBoundary_no_corner, self).__init__()
        if dim in (2, 3):
            self.dim = dim
        else:
            raise Exception("only 2d or 3d periodic boundary")

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary
        # AND NOT on one of the two corners (0, 1) and (1, 0)
        if self.dim == 2:
            in_or_not = bool((near(x[0], 0) or near(x[1], 0)) and
                             (not ((near(x[0], 0) and near(x[1], 1)) or
                                   (near(x[0], 1) and near(x[1], 0)) or
                                   (near(x[0], 0) and near(x[1], 0)))) and
                             on_boundary)
        else:
            in_or_not = bool((near(x[0], 0) or near(x[1], 0) or
                              near(x[2], 0)) and
                             (not ((near(x[0], 0) and near(x[1], 1)) or
                                   (near(x[0], 0) and near(x[1], 0)) or
                                   (near(x[0], 0) and near(x[2], 1)) or
                                   (near(x[0], 0) and near(x[2], 0)) or
                                   (near(x[0], 1) and near(x[1], 1)) or
                                   (near(x[0], 1) and near(x[1], 0)) or
                                   (near(x[0], 1) and near(x[2], 1)) or
                                   (near(x[0], 1) and near(x[2], 0)) or
                                   (near(x[1], 0) and near(x[2], 1)) or
                                   (near(x[1], 0) and near(x[2], 0)) or
                                   (near(x[1], 1) and near(x[2], 1)) or
                                   (near(x[1], 1) and near(x[2], 0)))) and
                             on_boundary)
        return in_or_not

    def map(self, x, y):
        if self.dim == 2:
            if near(x[0], 1.):
                y[0] = x[0] - 1.
                y[1] = x[1]
            else:  # near(x[1], 1)
                y[0] = x[0]
                y[1] = x[1] - 1.
        else:
            if near(x[0], 1.):
                y[0] = x[0] - 1.
                y[1] = x[1]
                y[2] = x[2]
            elif near(x[1], 1.):
                y[0] = x[0]
                y[1] = x[1] - 1.
                y[2] = x[2]
            else:
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 1.


def gmsh_with_incl_test():
    print 'gmsh with inclusion test'
    mesh = Mesh(r"m.xml")
    mesh = Mesh(r"m_fine.xml")
    # Generate Inclusion
    inc1 = InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_group = {'circle_inc1': inc1}
    # Initiate UnitCell Instance with Inclusion
    cell = UnitCell(mesh, inc_group)
    cell.view_domain()


def init_cell_with_inclusion_and_add_test():
    print 'inclusion add test'
    mesh = UnitSquareMesh(40, 40, 'crossed')
    inc1 = InclusionCircle(2, (0.1, 0.1), 0.5)
    inc2 = InclusionCircle(2, (0.9, 0.9), 0.5)
    # inc3 = InclusionRectangle(0.2, 0.6, 0.3, 0.8)
    inc_group = {'circle_inc1': inc1}
    add_inc_group = {'circle_inc2': inc2}
    # inc_group = {'rect': inc4}
    cell = UnitCell(mesh, inc_group)
    cell.set_append_inclusion(add_inc_group)
    cell.add_mark_boundary(1)
    cell.view_domain()
    print cell.incl_di.keys()


def multiple_inclusion_test():
    print 'multiple inclusions test'
    mesh = UnitSquareMesh(40, 40, 'crossed')
    inc1 = InclusionCircle(2, (0.1, 0.1), 0.5)
    inc2 = InclusionCircle(2, (0.9, 0.9), 0.5)
    inc3 = InclusionRectangle(2, 0.1, 0.3, 0.7, 0.9)
    inc4 = InclusionRectangle(2, 0.7, 0.9, 0.1, 0.3)
    inc_group = {'circle_inc1': inc1, 'circle_inc2': inc2,
                 'rect_inc3': inc3, 'rect_inc4': inc4}
    cell = UnitCell(mesh, inc_group)
    cell.view_domain()
    print cell.incl_di.keys()


def inclusion_test_3d():
    print '3d geometry test'
    mesh = UnitCubeMesh(20, 20, 20)
    cell = UnitCell(mesh)
    inc1 = InclusionCircle(3, (0.1, 0.1, 0.1), 0.5)
    inc2 = InclusionCircle(3, (0.9, 0.9, 0.9), 0.5)
    inc3 = InclusionRectangle(3, 0.7, 1., 0., 0.3, 0.7, 1.)
    inc4 = InclusionRectangle(3, 0., 0.3, 0.7, 1., 0., 0.3)
    inc_group = {'circle_inc1': inc1, 'circle_inc2': inc2,
                 'rect_inc3': inc3, 'rect_inc4': inc4}
    cell = UnitCell(mesh, inc_group)
    cell.view_domain()

def inclusion_test_3d_2():
    print '3d geometry test'
    mesh = UnitCubeMesh(20, 20, 20)
    cell = UnitCell(mesh)
    inc = InclusionCircle(3, 0.5)
    inc1 = InclusionRectangle(3, 0., 0.3, 0., 0.3, 0., 0.3)
    inc2 = InclusionRectangle(3, 0., 0.3, 0., 0.3, 0.7, 1.)
    inc3 = InclusionRectangle(3, 0., 0.3, 0.7, 1., 0., 0.3)
    inc4 = InclusionRectangle(3, 0., 0.3, 0.7, 1., 0.7, 1.)
    inc5 = InclusionRectangle(3, 0.7, 1., 0., 0.3, 0., 0.3)
    inc6 = InclusionRectangle(3, 0.7, 1., 0., 0.3, 0.7, 1.)
    inc7 = InclusionRectangle(3, 0.7, 1., 0.7, 1., 0., 0.3)
    inc8 = InclusionRectangle(3, 0.7, 1., 0.7, 1., 0.7, 1.)
    inc_group = {'circle': inc, 'corner1': inc1, 'corner2': inc2,
                 'corner3': inc3, 'corner4': inc4, 'corner5': inc5,
                 'corner6': inc6, 'corner7': inc7, 'corner8': inc8}
    cell = UnitCell(mesh, inc_group)
    cell.view_domain()


if __name__ == "__main__":
    print 'this is for testing'

    # gmsh_with_incl_test()

    # init_cell_with_inclusion_and_add_test()

    # multiple_inclusion_test()

    # inclusion_test_3d()

    # string_template(1, True, [7])

    # compiled_face_subdom(3)

    # inclusion_test_3d()

    inclusion_test_3d_2()
