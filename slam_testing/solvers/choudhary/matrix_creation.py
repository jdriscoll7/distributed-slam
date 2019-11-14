def create_rotation_system(rotation_list):
    """
    Creates the linear system used to estimate the initial rotations. This is the quadratic relaxation used
    in the first stage of the centralized approach. See (6) in Choudhary paper.

    :param rotation_list: list of relative rotation measurements
    :return:              (b, A) - the matrices used in the system Ar = b to solve least squares
    """

