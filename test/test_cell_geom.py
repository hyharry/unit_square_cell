# coding = utf-8
# Copyright (C) Yi Hu


def test_gmsh_with_incl():
    print 'gmsh with inclusion test'
    mesh = Mesh(r"m.xml")
    mesh = Mesh(r"m_fine.xml")
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
    a, b, c = 3, 6, 9
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

    # test_period_3d()

    # test_inclusion_3d_2()

    # test_period_2d()

    test_period_3d()

    # test_string_template()
