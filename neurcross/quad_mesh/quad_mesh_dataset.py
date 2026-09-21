import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import trimesh


class ReconDataset(data.Dataset):
    def __init__(self, file_path, n_points, n_samples=128, grid_range=1.1):

        self.file_path = file_path
        self.n_points = n_points
        self.n_samples = n_samples

        self.mesh = trimesh.load_mesh(self.file_path, process=False)
        self.grid_range = grid_range

        self.points, self.mnfld_n = self.get_face_center_points()

        self.mnfld_n, self.vector_u, self.vector_v = self.vector_u_v(self.mnfld_n)
        self.change_u_v_FLAG = 0.
        self.change_theta_hessian_term = 0.


        self.bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)]).transpose()

        # record sigma
        self.sample_gaussian_noise_around_shape()

    def get_face_center_points(self):
        # Returns points on the manifold, the points are on the face center of the mesh
        points = np.asarray(self.mesh.triangles_center, dtype=np.float32)
        normals = np.asarray(self.mesh.face_normals, dtype=np.float32)
        if normals.shape[0] == 0:
            normals = np.zeros_like(points)
        # center and scale data/point cloud
        self.cp = points.mean(axis=0)
        points = points - self.cp[None, :]
        self.scale = np.abs(points).max()
        points = points / self.scale

        return points, normals

    def vector_u_v(self, normals):
        # define the local coordinate system
        flags = np.ones_like(normals, dtype=bool)
        min_value_index = np.argmin(np.absolute(normals), axis=1)
        flags[np.arange(normals.shape[0]), min_value_index] = False
        true_ind = np.argwhere(flags)
        flag_row = true_ind[:, 0].reshape(-1, 2)
        flag_col = true_ind[:, 1].reshape(-1, 2)

        vector_u = np.zeros_like(normals, dtype=np.float32)
        vector_u[flag_row[:, 0], flag_col[:, 0]] = normals[flag_row[:, 1], flag_col[:, 1]]
        vector_u[flag_row[:, 1], flag_col[:, 1]] = -normals[flag_row[:, 0], flag_col[:, 0]]

        vector_v = np.cross(normals, vector_u)

        # normalization
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        vector_u = vector_u / (np.linalg.norm(vector_u, axis=1, keepdims=True) + 1e-12)
        vector_v = vector_v / (np.linalg.norm(vector_v, axis=1, keepdims=True) + 1e-12)

        return normals, vector_u, vector_v

    def sample_gaussian_noise_around_shape(self):
        kd_tree = spatial.KDTree(self.points)
        # query each point for sigma
        dist, _ = kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        self.sigmas = sigmas
        return

    def __getitem__(self, index):
        manifold_points = self.points  # (n_points, 3)
        manifold_normals = self.mnfld_n  # (n_points, 3)

        nonmnfld_points = np.random.uniform(-self.grid_range, self.grid_range,
                                            size=(self.n_points, 3)).astype(np.float32)  # (n_points // 2, 3)

        near_points = (manifold_points + self.sigmas * np.random.randn(manifold_points.shape[0],
                                                                       manifold_points.shape[1])).astype(
            np.float32)

        vector_u = self.vector_u
        vector_v = self.vector_v


        return {'points': manifold_points, 'mnfld_n': manifold_normals, 'nonmnfld_points': nonmnfld_points,
                'near_points': near_points, 'local_coordinates_u': vector_u, 'local_coordinates_v': vector_v}

    def __len__(self):
        return self.n_samples
