import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np


class VoxelGridNode(Node):
    def __init__(self):
        super().__init__('voxel_grid_node')

        self.voxel_size = self.declare_parameter('voxel_size', 0.3).value

        self.sub = self.create_subscription(
            PointCloud2,
            '/loxo/point_cloud',
            self.callback,
            10
        )

        # self.sub = self.create_subscription(
        #     PointCloud2,
        #     'kitti_pc',
        #     self.callback,
        #     10
        # )

        self.pub = self.create_publisher(
            PointCloud2,
            '/voxelized_cloud',
            10
        )

        self.get_logger().info(f'VoxelGridNode with intensity started. voxel_size={self.voxel_size}')

    # cloud to numpy
    def cloud_to_numpy(self, msg):
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([p[0], p[1], p[2], p[3]])
        return np.array(pts, dtype=np.float32)

    # main processing
    def callback(self, msg):

        points = self.cloud_to_numpy(msg)
        if points.shape[0] == 0:
            return

        xyz = points[:, :3]
        intensity = points[:, 3]

        # voxelization (https://stackoverflow.com/questions/5928725/hashing-2d-3d-and-nd-vectors)
        voxel_size = self.voxel_size
        idx = np.floor(xyz / voxel_size).astype(np.int32)

        keys = idx[:, 0] * 73856093 ^ \
               idx[:, 1] * 19349663 ^ \
               idx[:, 2] * 83492791

        unique_keys, inverse = np.unique(keys, return_inverse=True)

        num_voxels = len(unique_keys)

        centroids = np.zeros((num_voxels, 3), dtype=np.float32)
        counts = np.bincount(inverse)

        np.add.at(centroids, inverse, xyz)
        centroids /= counts[:, None]

        intensity_sum = np.zeros(num_voxels, dtype=np.float32)
        np.add.at(intensity_sum, inverse, intensity)

        mean_intensity = intensity_sum / counts

        voxel_points = np.hstack((centroids, mean_intensity[:, None]))

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        out_msg = pc2.create_cloud(msg.header, fields, voxel_points.tolist())

        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoxelGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()