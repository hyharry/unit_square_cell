# coding=utf-8
# Copyright (C) 2016 Yi Hu
# python 2.7, FEniCS 1.6.0
"""
2D or 3D Unit Cell Geometry setting, Periodic Boundary condition, Inclusion
Library

Class: UnitCell (view_mesh, view_domain, set_append_inclusion,
                add_mark_boundary)
Function: string_template, trim_str_template, compiled_corner_subdom,
        compiled_line_subdom, InclusionCircle, InclusionRectangle，
        PeriodicBoundary_no_corner

"""

from dolfin import *
from math import sqrt, pi
import logging

logging.getLogger('FFC').setLevel(logging.WARNING)


class UnitCell(object):
    """
    2D: Unit Square, 3D: Unit Cube
    """
    def __init__(self, mesh, incl_di=None):
        """
        Generate Unit Cell for Micro Computation

        :param mesh: Mesh entity from dolfin
        :param incl_di: {'inclusion_name': inclusion, ...}

        """
        self.mesh = mesh
        self.dim = mesh.geometry().dim()
        self.domain = MeshFunction("size_t", mesh, self.dim)
        self.boundary = None

        self.domain.set_all(0)
        self.incl_di = {}
        if incl_di:
            self.set_append_inclusion(incl_di)

    def view_mesh(self):
        plot(self.mesh, interactive=True)

    def view_domain(self):
        plot(self.domain, interactive=True)

    def set_append_inclusion(self, incl_di):
        """
        Inclusion append

        :param incl_di: (dict) keys:(string), incl name, values:(
                        InclusionCircle)

        :return: updated inclu_di

        """
        exist_incl_num = len(self.incl_di)
        k = exist_incl_num + 1
        for inc in incl_di.values():
            inc.mark(self.domain, k)
            k += 1
        self.incl_di.update(incl_di)

    def add_mark_boundary(self, bound_dim):
        """
        Add index to each bondary entity.

        2D case: if boundary edges are marked, index are given to 4 edges
                    [0, 1, 2, 3]
                 if boundary points are marked, 4 corners are marked
                 while other edges are marked with 30

        3D case: similar with 2D

        Be careful that only one kind of boundary entity can be added (edges
        or points)!!

        Add boundary is not really needed for the current state!!
        all kinds of boundary can be generated

        in fact it is needed when Neumann Boundary is required to impose
        :param bound_dim: dimension of boundary entity

        """
        dim = self.dim
        if bound_dim >= dim:
            raise Exception("invalid boundary dimension")

        # FIXME: can only add one kind of boundary entity, boundary entity
        # labels are not well given
        self.boundary = MeshFunction("size_t", self.mesh, bound_dim)
        self.boundary.set_all(30)
        if bound_dim == 0:
            compiled_boundary = compiled_corner_subdom(dim)
        elif bound_dim == 1:
            compiled_boundary = compiled_line_subdom(dim)
        else:
            compiled_boundary = compiled_face_subdom(dim)

        for i, bound in enumerate(compiled_boundary):
            bound.mark(self.boundary, i)


def string_template(dim, with_boundary=False, coord_label=None,
                    joint_sym="&&", in_colon=False, dict_output=False,
                    no_origin=False):
    """
    String Expression generator

    :purpose:
    Generate a list of string for method compiled_corner_subdom or alike.
    User can specify coord_label, joint_sym, colon at both side,
    "on_boundary" at the end, or as output a dict.

    :dependency:
    contain: trim_str_template
    used in: compiled_corner_subdom, compiled_line_subdom, compiled_face_subdom
    PeriodicBoundary_no_corner

    :param no_origin: (bool) if True, return no expression at (0.,), (0.,0.),
                        it works only with dict_output=True
    :param dim: (int) in (1,2,3) - dimension of the problem
    :param with_boundary: (bool) - add "on_boundary" at the end
                            of each string
    :param coord_label: (list) [i,j,k, ..]
                        - change the coordinate label from 0,1,2 to i,j,k
                        e.g. x[i], x[j], x[k]
    :param joint_sym: (string) 'or' 'and' '&' '&&' ..
                        - substitute '&&' in 'near() && near()' as the given
    :param in_colon: 
	| (bool) - 'near()' -> '(near())'
    :param dict_output: (bool) - extract out the coord and set it as key

    :return: (list of strings), directly to compile, eval, or,
                CompiledSubDomain(), etc.

    :usage:
    | string_template(1) -> ['near(x[0], 0.)', 'near(x[0], 1.)']
    | string_template(2) -> ['near(x[0], 0.) && near(x[1], 0.)',
    |                        'near(x[0], 0.) && near(x[1], 1.)',
    |                        'near(x[0], 1.) && near(x[1], 0.)',
    |                        'near(x[0], 1.) && near(x[1], 1.)']

    string_template(2, dict_output=True, joint_sym='or') ->
    {(0.0, 1.0): 'near(x[0], 0.) or near(x[1], 1.)',
    (1.0, 0.0): 'near(x[0], 1.) or near(x[1], 0.)',
    (0.0, 0.0): 'near(x[0], 0.) or near(x[1], 0.)',
    (1.0, 1.0): 'near(x[0], 1.) or near(x[1], 1.)'}

    """
    val = [0., 1.]
    str_template = dict()
    if not coord_label:
        coord_label = range(dim)
    for coord_i in coord_label:
        # str_template[coord_i] = "near(x[{coord_i}], {{val:f}}, " \
        #                         "DOLFIN_EPS) &&".format(coord_i=coord_i)
        str_template[coord_i] = "near(x[{coord_i}], {{val:f}}) {sym}".format(
                coord_i=coord_i, sym=joint_sym)

    # Expand template using values
    str_template_2 = dict()
    for i in coord_label:
        str_template_2[i] = [str_template[i].format(val=val_i) for val_i in val]

    # Generate string, join, and trim
    ke = str_template_2.keys()
    if dim == 1:
        if dict_output:
            str_template_3 = dict(((val[lab],), i)
                                  for lab, i in
                                  enumerate(str_template_2[ke[0]]))
            comp_str_val = trim_str_template(str_template_3.values(),
                                             joint_sym, with_boundary, in_colon)
            comp_str = dict(zip(str_template_3.keys(), comp_str_val))
            if no_origin:
                comp_str.pop((0.,))
        else:
            str_template_3 = [i
                              for i in str_template_2[ke[0]]]
            comp_str = trim_str_template(str_template_3, joint_sym,
                                         with_boundary, in_colon)

    elif dim == 2:
        if dict_output:
            str_template_3 = dict(((val[lab_i], val[lab_j]), ' '.join((i, j)))
                                  for lab_i, i in
                                  enumerate(str_template_2[ke[0]])
                                  for lab_j, j in
                                  enumerate(str_template_2[ke[1]]))
            comp_str_val = trim_str_template(str_template_3.values(),
                                             joint_sym, with_boundary, in_colon)
            comp_str = dict(zip(str_template_3.keys(), comp_str_val))
            if no_origin:
                comp_str.pop((0., 0.))
        else:
            str_template_3 = [' '.join((i, j))
                              for i in str_template_2[ke[0]]
                              for j in str_template_2[ke[1]]]
            comp_str = trim_str_template(str_template_3, joint_sym,
                                         with_boundary, in_colon)

    elif dim == 3:
        if dict_output:
            str_template_3 = dict(
                    ((val[lab_i], val[lab_j], val[lab_k]), ' '.join((i, j, k)))
                    for lab_i, i in enumerate(str_template_2[ke[0]])
                    for lab_j, j in enumerate(str_template_2[ke[1]])
                    for lab_k, k in enumerate(str_template_2[ke[2]]))
            comp_str_val = trim_str_template(str_template_3.values(),
                                             joint_sym, with_boundary, in_colon)
            comp_str = dict(zip(str_template_3.keys(), comp_str_val))
            if no_origin:
                comp_str.pop((0., 0., 0.))
        else:
            str_template_3 = [' '.join((i, j, k))
                              for i in str_template_2[ke[0]]
                              for j in str_template_2[ke[1]]
                              for k in str_template_2[ke[2]]]
            comp_str = trim_str_template(str_template_3, joint_sym,
                                         with_boundary, in_colon)
    else:
        raise Exception("Only 1d, 2d, 3d cases are supported")

    # print comp_str
    return comp_str


def trim_str_template(str_temp_li, joint_sym, with_boundary=False,
                      in_colon=False):
    """
    Assistance function for string_template

    :param str_temp_li: (list of strings) - input
    :param joint_sym: (string) - trim joint_sym at the end
    :param with_boundary: (bool) - add 'on_boundary'
    :param in_colon: (bool) - put each string in a colon

    :return: trimmed String Expression List

    """
    comp_str = str_temp_li
    if with_boundary:
        comp_str = [' '.join((stri, "on_boundary"))
                    for stri in str_temp_li]
    else:
        trim_end = len(joint_sym) + 1
        comp_str = [stri[:-trim_end] for stri in str_temp_li]

    if in_colon:
        comp_str = ["(" + stri + ")" for stri in comp_str]
    # print comp_str
    return comp_str


def compiled_corner_subdom(dim):
    """
    Return compiled subdomain for corners

    :param dim: dimension
    :return: (list of SubDomains)

    """
    comp_str = string_template(dim)
    comp_corner_sub = [CompiledSubDomain(stri) for stri in comp_str]
    return comp_corner_sub


def compiled_line_subdom(dim):
    """
    Return compiled subdomain for edges

    :param dim: dimension
    :return: (list of SubDomains)

    """
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
    """
    Return compiled subdomain for faces

    :param dim: dimension
    :return: (list of SubDomains)

    """
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
    """
    Subclass of SubDomain for circle inclusion (2D and 3D)
    """
    def __init__(self, dim, *args):
        """
        Two ways to generate Circle Inclusion

        :param dim: dimension of the problem
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
            d = sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2 + (
                    x[2] - c[2]) ** 2)
        else:
            raise Exception("only 2d or 3d circle inclusion are supported")
        return d < r or near(d, r)


class InclusionRectangle(SubDomain):
    """
    Subclass of SubDomain for rectangle inclusion (2D and 3D)
    """
    def __init__(self, dim, *args):
        """
        Generate Rectangle Inclusion

        :param dim: dimension of the problem)
        :param args[0]: x_lower_bound
        :param args[1]: x_upper_bound
        :param args[2]: y_lower_bound
        :param args[3]: y_upper_bound
        :param args[4]: z_lower_bound
        :param args[5]: z_upper_bound
        """
        super(InclusionRectangle, self).__init__()
        if len(args) / 2 != dim:
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
    Periodic boundary both directions (no corner) for 2D and 3D
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
            # String Expression list for corners
            comp_corner_li = string_template(3, joint_sym='and', in_colon=True)

            # String Expression dict without origin crossing ones
            comp_edge_li_1 = string_template(2, joint_sym='and',
                                             coord_label=[0, 1],
                                             in_colon=True, dict_output=True,
                                             no_origin=True)
            comp_edge_li_2 = string_template(2, joint_sym='and',
                                             coord_label=[1, 2],
                                             in_colon=True, dict_output=True,
                                             no_origin=True)
            comp_edge_li_3 = string_template(2, joint_sym='and',
                                             coord_label=[0, 2],
                                             in_colon=True, dict_output=True,
                                             no_origin=True)

            comp_str_joint = ' or '.join(comp_corner_li +
                                         comp_edge_li_1.values() +
                                         comp_edge_li_2.values() +
                                         comp_edge_li_3.values())

            # Mark main facets and edges on the axis, filter out all corners,
            #  filter out edges not on the main axis
            # Points on 3 facets and 3 edges are like ref points,
            # other points are to be mapped into these points
            in_or_not = bool((near(x[0], 0.) or
                              near(x[1], 0.) or
                              near(x[2], 0.)) and
                             (not eval(comp_str_joint)) and on_boundary)
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
            # Periodic for edges, map 3 edges to 1 edge
            # Parallel in z direction
            if near(x[0], 1.) and near(x[1], 1.):
                y[0] = x[0] - 1.
                y[1] = x[1] - 1.
                y[2] = x[2]
            elif near(x[0], 1.) and near(x[1], 0.):
                y[0] = x[0] - 1.
                y[1] = x[1]
                y[2] = x[2]
            elif near(x[0], 0.) and near(x[1], 1.):
                y[0] = x[0]
                y[1] = x[1] - 1.
                y[2] = x[2]
            # Parallel in x direction
            elif near(x[2], 1.) and near(x[1], 1.):
                y[0] = x[0]
                y[1] = x[1] - 1.
                y[2] = x[2] - 1.
            elif near(x[2], 1.) and near(x[1], 0.):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 1.
            elif near(x[2], 0.) and near(x[1], 1.):
                y[0] = x[0]
                y[1] = x[1] - 1.
                y[2] = x[2]
            # Parallel in y direction
            elif near(x[2], 1.) and near(x[0], 1.):
                y[0] = x[0] - 1.
                y[1] = x[1]
                y[2] = x[2] - 1.
            elif near(x[2], 1.) and near(x[0], 0.):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 1.
            elif near(x[2], 0.) and near(x[0], 1.):
                y[0] = x[0] - 1.
                y[1] = x[1]
                y[2] = x[2]
            # Face Mapping
            elif near(x[2], 1.):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 1.
            elif near(x[1], 1.):
                y[0] = x[0]
                y[1] = x[1] - 1.
                y[2] = x[2]
            elif near(x[0], 1.):
                y[0] = x[0] - 1.
                y[1] = x[1]
                y[2] = x[2]
            else:
                y[0] = 100
                y[1] = 100
                y[2] = 100

        # edge_str_li_1 = string_template(2, coord_label=[0, 1],
        #                                 joint_sym='and',
        #                                 dict_output=True, no_origin=True)
        # edge_str_li_2 = string_template(2, coord_label=[1, 2],
        #                                 joint_sym='and',
        #                                 dict_output=True, no_origin=True)
        # edge_str_li_3 = string_template(2, coord_label=[0, 2],
        #                                 joint_sym='and',
        #                                 dict_output=True, no_origin=True)
        #
        # face_str_li_1 = string_template(1, coord_label=[0],
        #                                 joint_sym='and', dict_output=True)
        # face_str_li_2 = string_template(1, coord_label=[1],
        #                                 joint_sym='and', dict_output=True)
        # face_str_li_3 = string_template(1, coord_label=[2],
        #                                 joint_sym='and', dict_output=True)
        #
        # # Map edges
        # for k_coord, on_edge in edge_str_li_1.items():
        #     if eval(on_edge):
        #         y[0] = x[0] - k_coord[0]
        #         y[1] = x[1] - k_coord[1]
        #         y[2] = x[2]
        #
        # for k_coord, on_edge in edge_str_li_2.items():
        #     if eval(on_edge):
        #         y[0] = x[0]
        #         y[1] = x[1] - k_coord[0]
        #         y[2] = x[2] - k_coord[1]
        #
        # for k_coord, on_edge in edge_str_li_3.items():
        #     if eval(on_edge):
        #         y[0] = x[0] - k_coord[0]
        #         y[1] = x[1]
        #         y[2] = x[2] - k_coord[1]
        #
        # # Map faces
        # for k_coord, on_face in face_str_li_1.items():
        #     if eval(on_face):
        #         y[0] = x[0] - k_coord[0]
        #         y[1] = x[1]
        #         y[2] = x[2]
        # for k_coord, on_face in face_str_li_2.items():
        #     if eval(on_face):
        #         y[0] = x[0]
        #         y[1] = x[1] - k_coord[0]
        #         y[2] = x[2]
        # for k_coord, on_face in face_str_li_3.items():
        #     if eval(on_face):
        #         y[0] = x[0]
        #         y[1] = x[1]
        #         y[2] = x[2] - k_coord[0]


def test_gmsh_with_incl():
    print 'gmsh with inclusion test'
    mesh = Mesh(r"m.xml")
    # mesh = Mesh(r"m_fine.xml")
    # Generate Inclusion
    inc1 = InclusionCircle(2, (0.5, 0.5), 0.25)
    inc_group = {'circle_inc1': inc1}
    # Initiate UnitCell Instance with Inclusion
    cell = UnitCell(mesh, inc_group)
    cell.view_domain()


def test_init_cell_with_inclusion_and_add():
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


def test_multiple_inclusion():
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


def test_inclusion_3d():
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


def test_inclusion_3d_2():
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


def test_inclusion_3d_3():
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


def test_string_template():
    print "BASIC OPERATION"
    print string_template(1)
    print string_template(2)
    print string_template(3)

    print "TRIM TEST"
    print string_template(1, joint_sym='or', with_boundary=True,
                          coord_label=[7],
                          in_colon=True)

    print "DICT TEST"
    print string_template(1, dict_output=True, joint_sym='and')
    print string_template(3, dict_output=True)
    print string_template(2, dict_output=True, joint_sym='or')

    print "NO ORIGIN TEST"
    print string_template(1, coord_label=[2], joint_sym='and',
                          dict_output=True, no_origin=True)


def test_period_3d():
    a, b, c = 4, 4, 4
    mesh_3d = UnitCubeMesh(a, b, c)
    FS_3d = FunctionSpace(mesh_3d, 'CG', 1,
                          constrained_domain=PeriodicBoundary_no_corner(3))
    f = Function(FS_3d)

    print "dof number should be", (
        (a + 1) * (b + 1) * (c + 1) - (a - 1 + b - 1 + c - 1) * 3 -
        (a - 1) * (b - 1) - (a - 1) * (c - 1) - (b - 1) * (c - 1))
    print f.vector().size()
    # print -(a - 1) * (b - 1) - (a - 1) * (c - 1) - (b - 1) * (c - 1)
    # print (a + 1) * (b + 1) * (c + 1)


def test_period_2d():
    a, b = 6, 3
    mesh_2d = UnitSquareMesh(a, b)
    FS_2d = FunctionSpace(mesh_2d, 'CG', 1,
                          constrained_domain=PeriodicBoundary_no_corner(2))
    f = Function(FS_2d)

    print "dof number should be", (a + 1) * (b + 1) - (a - 1 + b - 1)
    print f.vector().size()


if __name__ == "__main__":
    print 'this is for testing'

    # test_gmsh_with_incl()

    # test_init_cell_with_inclusion_and_add()

    # test_multiple_inclusion()

    # print string_template(2, coord_label=[3,4], joint_sym='and',
    #                       dict_output=True)

    # compiled_face_subdom(3)

    # test_inclusion_3d()

    # test_inclusion_3d_2()

    # test_period_3d()

    # test_period_2d()

    test_period_3d()

    # test_string_template()
