import logging
import numpy as np

class LogRecord(logging.LogRecord):

    def getMessage(self):
        msg = self.msg
        if self.args:
            if isinstance(self.args, dict):
                msg = msg.format(**self.args)
            else:
                msg = msg.format(*self.args)
        return msg

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([
        t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20,
        t11 - t12, t19 + t20, t1 + t2 * t24
    ],
                 axis=1)

    return np.reshape(R, (-1, 3, 3))

def trans_augment(t_augment, points, normals=None):
    """Implementation of an augmentation transform for point clouds."""

    if t_augment is None or not t_augment.get('turn_on', True):
        return points, normals

    # Initialize rotation matrix
    R = np.eye(points.shape[1])

    if points.shape[1] == 3:
        rotation_method = t_augment.get('rotation_method', None)
        if rotation_method == 'vertical':

            # Create random rotations
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif rotation_method == 'all':

            # Choose two random angles for the first vector in polar coordinates
            theta = np.random.rand() * 2 * np.pi
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)),
                                    np.reshape(alpha, (1, -1)))[0]

    R = R.astype(np.float32)

    # Choose random scales for each example
    scale_anisotropic = t_augment.get('scale_anisotropic', False)
    min_s = t_augment.get('min_s', 1.)
    max_s = t_augment.get('max_s', 1.)
    if scale_anisotropic:
        scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
    else:
        scale = np.random.rand() * (max_s - min_s) - min_s

    # Add random symmetries to the scale factor
    symmetries = t_augment.get('symmetries', False)
    symmetries = np.array(symmetries).astype(np.int32)
    symmetries = symmetries * np.random.randint(2, size=points.shape[1])
    scale = (scale * (1 - symmetries * 2)).astype(np.float32)

    noise_level = t_augment.get('noise_level', 0.001)
    noise = (np.random.randn(points.shape[0], points.shape[1]) *
             noise_level).astype(np.float32)

    augmented_points = np.sum(np.expand_dims(points, 2) * R,
                              axis=1) * scale + noise


    if normals is None:
        augmented_points = None
    else:
        # Anisotropic scale of the normals thanks to cross product formula
        normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
        augmented_normals = np.dot(normals, R) * normal_scale
        # Renormalise
        augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

    return augmented_points, augmented_normals

def construct_edges(faces):
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)
        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                
                edges_count += 1
            
    return edges
    