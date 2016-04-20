# coding=utf-8
# Copyright (C) 2016 Yi Hu
# python 2.7, FEniCS 1.6.0
"""
Unit test for cell_geom
"""
import unittest
import sys

sys.path.append('../')
import cell_geom as geom


class ImportMeshTest(unittest.TestCase):
    """
    Basic test for gmsh import
    """
    def test_gmsh_with_incl(self):
        # print 'gmsh with inclusion test'
        mesh = geom.Mesh(r"../m.xml")
        # Generate Inclusion
        inc1 = geom.InclusionCircle(2, (0.5, 0.5), 0.25)
        inc_group = {'circle_inc1': inc1}
        # Initiate UnitCell Instance with Inclusion
        cell = geom.UnitCell(mesh, inc_group)
        inc_num = len(set(cell.domain.array())) - 1
        self.assertEqual(inc_num, 1)


class TwoDimTestCase(unittest.TestCase):
    """
    Parent class for 2D geometry test only setup
    """
    def setUp(self):
        self.mesh = geom.UnitSquareMesh(40, 40, 'crossed')
        self.inc1 = geom.InclusionCircle(2, (0.1, 0.1), 0.5)
        self.inc2 = geom.InclusionCircle(2, (0.9, 0.9), 0.5)
        self.inc4 = geom.InclusionRectangle(2, 0.7, 0.9, 0.1, 0.3)
        self.inc3 = geom.InclusionRectangle(2, 0.1, 0.3, 0.7, 0.9)


class TwoDimInclGenTestCase(TwoDimTestCase):
    def test_init_cell_with_inclusion_and_add(self):
        inc_group = {'circle_inc1': self.inc1}
        add_inc_group = {'circle_inc2': self.inc2}
        cell = geom.UnitCell(self.mesh, inc_group)
        cell.set_append_inclusion(add_inc_group)
        inc_num = len(set(cell.domain.array())) - 1
        self.assertEqual(inc_num, 2)

    def test_multiple_inclusion(self):
        """
        Multiple inclusions test
        """
        inc_group = {'circle_inc1': self.inc1, 'circle_inc2': self.inc2,
                     'rect_inc3': self.inc3, 'rect_inc4': self.inc4}
        cell = geom.UnitCell(self.mesh, inc_group)
        inc_num = len(set(cell.domain.array())) - 1
        self.assertEqual(inc_num, 4)

    def test_mark_boundary(self):
        cell = geom.UnitCell(self.mesh)
        cell.add_mark_boundary(1)
        edge_num = len(set(cell.boundary.array())) - 1
        self.assertEqual(edge_num, 4)


class ThreeDimInclTestCase(unittest.TestCase):
    """
    3d geometry test
    """
    def setUp(self):
        self.mesh = geom.UnitCubeMesh(20, 20, 20)
        # cell = geom.UnitCell(mesh)

    def test_inclusion_3d(self):
        """
        3d geometry test 1
        """
        inc1 = geom.InclusionCircle(3, (0.1, 0.1, 0.1), 0.5)
        inc2 = geom.InclusionCircle(3, (0.9, 0.9, 0.9), 0.5)
        inc3 = geom.InclusionRectangle(3, 0.7, 1., 0., 0.3, 0.7, 1.)
        inc4 = geom.InclusionRectangle(3, 0., 0.3, 0.7, 1., 0., 0.3)
        inc_group = {'circle_inc1': inc1, 'circle_inc2': inc2,
                     'rect_inc3': inc3, 'rect_inc4': inc4}
        cell = geom.UnitCell(self.mesh, inc_group)
        inc_num = len(set(cell.domain.array())) - 1
        self.assertEqual(inc_num, len(inc_group))

    def test_inclusion_3d_2(self):
        """
        3d geometry test 2
        """
        inc = geom.InclusionCircle(3, 0.5)
        inc1 = geom.InclusionRectangle(3, 0., 0.3, 0., 0.3, 0., 0.3)
        inc2 = geom.InclusionRectangle(3, 0., 0.3, 0., 0.3, 0.7, 1.)
        inc3 = geom.InclusionRectangle(3, 0., 0.3, 0.7, 1., 0., 0.3)
        inc4 = geom.InclusionRectangle(3, 0., 0.3, 0.7, 1., 0.7, 1.)
        inc5 = geom.InclusionRectangle(3, 0.7, 1., 0., 0.3, 0., 0.3)
        inc6 = geom.InclusionRectangle(3, 0.7, 1., 0., 0.3, 0.7, 1.)
        inc7 = geom.InclusionRectangle(3, 0.7, 1., 0.7, 1., 0., 0.3)
        inc8 = geom.InclusionRectangle(3, 0.7, 1., 0.7, 1., 0.7, 1.)
        inc_group = {'circle': inc, 'corner1': inc1, 'corner2': inc2,
                     'corner3': inc3, 'corner4': inc4, 'corner5': inc5,
                     'corner6': inc6, 'corner7': inc7, 'corner8': inc8}
        cell = geom.UnitCell(self.mesh, inc_group)
        inc_num = len(set(cell.domain.array())) - 1
        self.assertEqual(inc_num, len(inc_group))


class StringTempTestCase(unittest.TestCase):
    @unittest.skip('string template check skipped')
    def test_string_template(self):
        print "BASIC OPERATION"
        print geom.string_template(1)
        print geom.string_template(2)
        print geom.string_template(3)

        print "TRIM TEST"
        print geom.string_template(1, joint_sym='or', with_boundary=True,
                                   coord_label=[7], in_colon=True)

        print "DICT TEST"
        print geom.string_template(1, dict_output=True, joint_sym='and')
        print geom.string_template(3, dict_output=True)
        print geom.string_template(2, dict_output=True, joint_sym='or')

        print "NO ORIGIN TEST"
        print geom.string_template(1, coord_label=[2], joint_sym='and',
                                   dict_output=True, no_origin=True)


class PeriodBoundTestCase(unittest.TestCase):
    """
    Test Case for Periodic Boundary Condition
    """
    def test_period_2d(self):
        a, b = 6, 3
        mesh_2d = geom.UnitSquareMesh(a, b)
        FS_2d = geom.FunctionSpace(mesh_2d, 'CG', 1,
                                   constrained_domain=geom.PeriodicBoundary_no_corner(2))
        f = geom.Function(FS_2d)
        self.assertEqual(f.vector().size(), (a + 1) * (b + 1) - (a - 1 + b - 1))

    def test_period_3d(self):
        a, b, c = 3, 6, 9
        mesh_3d = geom.UnitCubeMesh(a, b, c)
        FS_3d = geom.FunctionSpace(mesh_3d, 'CG', 1,
                                   constrained_domain=geom.PeriodicBoundary_no_corner(3))
        f = geom.Function(FS_3d)
        self.assertEqual(f.vector().size(),
                         (a + 1) * (b + 1) * (c + 1) - (a - 1 + b - 1 + c - 1) * 3 -
                         (a - 1) * (b - 1) - (a - 1) * (c - 1) - (b - 1) * (c - 1))


if __name__ == "__main__":
    unittest.main()
    # t = TwoDimTestCase()
    # suite = unittest.TestLoader().loadTestsFromTestCase(ImportMeshTest)
    # unittest.TextTestRunner().run(suite)
