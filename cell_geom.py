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
        self.domain = CellFunction("size_t", mesh)
        self.boundary = FaceFunction('size_t', mesh, mesh.topology().dim() - 1)

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
        k = exist_incl_num+1
        for inc in incl_di.values():
            inc.mark(self.domain, k)
            k += 1
        self.incl_di.update(incl_di)

    def add_boundary(self):
        # mark boundaries
        left = LeftBoundary()
        left.mark(self.boundary, 0)
        right = RightBoundary()
        right.mark(self.boundary, 1)
        bottom = BottomBoundary()
        bottom.mark(self.boundary, 2)
        top = TopBoundary()
        top.mark(self.boundary, 3)

    def mark_corner_bc(self):
        corner0 = CompiledSubDomain("near(x[0], 0.0) && near(x[1], 0.0)")
        corner1 = CompiledSubDomain("near(x[0], 1.0) && near(x[1], 0.0)")
        corner2 = CompiledSubDomain("near(x[0], 0.0) && near(x[1], 1.0)")
        corner3 = CompiledSubDomain("near(x[0], 1.0) && near(x[1], 1.0)")
        return [corner0, corner1, corner2, corner3]

    def mark_side_bc(self):
        left_side = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
        right_side = CompiledSubDomain("near(x[0], 1.0) && on_boundary")
        bottom_side = CompiledSubDomain("near(x[1], 0.0) && on_boundary")
        top_side = CompiledSubDomain("near(x[1], 1.0) && on_boundary")
        return {'l': left_side, 'r': right_side, 'b': bottom_side,
                't': top_side}


class InclusionCircle(SubDomain):
    def __init__(self, *args):
        """
        Two ways to generate Circle Inclusion

        :param args: 1. args[0] = (x_c, y_c), args[1] = r
                     2. args[0] = fraction ration, center is (0.5, 0.5)
        """
        super(InclusionCircle, self).__init__()
        if len(args) > 1:
            self.r = args[1]
            self.c = args[0]
        else:
            self.ratio = args[0]
            self.c = (0.5, 0.5)
            self.r = sqrt(self.ratio / pi)

    def inside(self, x, on_boundary):
        c = self.c
        r = self.r
        d = sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)
        return d < r or near(d, r)


class InclusionRectangle(SubDomain):
    def __init__(self, a, b, c, d):
        """
        Generate Rectangle Inclusion

        :param a: x_lower_bound
        :param b: x_upper_bound
        :param c: y_lower_bound
        :param d: y_upper_bound
        """
        super(InclusionRectangle, self).__init__()
        self.x_a = a
        self.x_b = b
        self.y_c = c
        self.y_d = d

    def inside(self, x, on_boundary):
        return self.x_a <= x[0] <= self.x_b and self.y_c <= x[1] <= self.y_d


# mark Boundary
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)


class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1)


class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)


class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1)


# periodic boudary both directions (no corner)
class PeriodicBoundary_no_corner(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary
        # AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 1)) or
                          (near(x[0], 1) and near(x[1], 0)) or
                          (near(x[0], 0) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1):
            y[0] = x[0] - 1.
            y[1] = x[1]
        else:  # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.


if __name__ == "__main__":
    print 'this is for testing'
    # mesh = Mesh(r"m.xml")
    mesh = UnitSquareMesh(10, 10, 'crossed')
    # cell = UnitCell(mesh)
    # inc1 = InclusionCircle(0.5)
    inc2 = InclusionCircle((0.1,0.1),0.5)
    inc3 = InclusionCircle((0.9,0.9),0.5)
    inc4 = InclusionRectangle(0.2, 0.6, 0.3, 0.8)
    # inc_group = {'circle_inc1': inc1}
    # inc_group = {'circle_inc1': inc2}
    # inc_group = {'circle_inc1': inc2, 'circle_inc2': inc3}
    inc_group = {'rect': inc4}
    cell = UnitCell(mesh, inc_group)
    # domains = CellFunction('size_t',mesh)
    # domains.set_all(0)
    # inc1.mark(domains,1)
    cell.set_append_inclusion(inc_group)
    cell.add_boundary()
    bc = cell.mark_corner_bc()
    bc2 = cell.mark_side_bc()
    cell.view_domain()
    print type(bc2)
