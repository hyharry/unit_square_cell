# coding=utf-8

from dolfin import *
import logging

logging.getLogger('FFC').setLevel(logging.WARNING)


class unit_cell(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.domain = CellFunction("size_t", mesh)
        self.boundary = FaceFunction('size_t', mesh, mesh.topology().dim() - 1)

    def view_mesh(self):
        plot(self.mesh, interactive=True)

    def view_domain(self):
        plot(self.domain, interactive=True)

    def inclusion(self, inclu):
        self.domain.set_all(0)
        k = 1
        for inc in inclu:
            inc.mark(self.domain, k)
            k += 1

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
        return {'l': left_side, 'r': right_side, 'b': bottom_side, 't': top_side}


# mark circle inclusion
class Inclusion_Circle(SubDomain):
    # def __init__(self, c=[0.5,0.5], r=0.25):
    #     self.__r = r
    #     self.__c = c
    # def parameter(self,r=0.25,c=[0.5,0.5]):
    #     self.r = r
    #     self.c = c
    def inside(self, x, on_boundary, c=[0.5, 0.5], r=0.25):
        d = sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)
        return d < r or near(d, r)


class Inclusion_Strip(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= 0.2 and x[0] <= 0.8)


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
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 1)) or (near(x[0], 1) and near(x[1], 0)) or
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
    mesh = Mesh(r"m.xml")
    cell = unit_cell(mesh)
    inc1 = Inclusion_Circle()
    # inc2 = Inclusion_Circle([0.1,0.1],0.05)
    # inc3 = Inclusion_Circle([0.9,0.9],0.05)
    inc_group = [inc1]
    # domains = CellFunction('size_t',mesh)
    # domains.set_all(0)
    # inc1.mark(domains,1)
    cell.inclusion(inc_group)
    cell.add_boundary()
    bc = cell.mark_corner_bc()
    bc2 = cell.mark_side_bc()
    print type(bc2)
