import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
from scipy.spatial import cKDTree
import tf_transformations


class DynamicFilterNode(Node):

    def __init__(self):
        super().__init__('dynamic_filter_node')

        self.voxel_size = self.declare_parameter('voxel_size', 0.3).value
        self.frame_dist_thresh = self.declare_parameter('frame_dist_thresh', 0.35).value
        self.intensity_thresh = self.declare_parameter('intensity_thresh', 10.0).value
        self.stability_thresh = self.declare_parameter('stability_thresh', 0.75).value
        self.decay = self.declare_parameter('decay_rate', 0.9).value

        self.alpha = self.declare_parameter('hysteresis_alpha', 0.4).value
        self.beta = self.declare_parameter('hysteresis_beta', 0.1).value
        self.dynamic_threshold = self.declare_parameter('dynamic_threshold', 0.5).value

        self.prev_points = None
        self.prev_pose = None
        self.curr_pose = None

        self.voxel_map = {}
        self.dynamic_score_map = {}

        self.transform_correction = False

        self.sub_cloud = self.create_subscription(
            PointCloud2,
            '/voxelized_cloud',
            self.cloud_callback,
            10
        )

        if self.transform_correction:
            self.sub_pose = self.create_subscription(
                PoseStamped,
                '/loxo/vehicle_pose',
                self.pose_callback,
                10
            )

        self.pub_dyn = self.create_publisher(PointCloud2, '/dynamic_cloud', 10)
        self.pub_sta = self.create_publisher(PointCloud2, '/static_cloud', 10)

        self.get_logger().info("Dynamic Filter with pose alignment started")


    # pose saving
    def pose_callback(self, msg):
        self.curr_pose = msg


    # converting pc in numpy array
    def cloud_to_numpy(self, msg):
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([p[0], p[1], p[2], p[3]])
        return np.array(pts, dtype=np.float32)


    # voxel keys, primes shall be the same as in voxelizer
    def compute_voxel_keys(self, xyz):
        idx = np.floor(xyz / self.voxel_size).astype(np.int32)
        return idx[:, 0] * 73856093 ^ idx[:, 1] * 19349663 ^ idx[:, 2] * 83492791


    # voxel stability
    def update_voxel_map(self, voxel_keys):
        observed = set(voxel_keys.tolist())

        # decay
        for k in list(self.voxel_map.keys()):
            self.voxel_map[k] *= self.decay
            if self.voxel_map[k] < 0.05:
                del self.voxel_map[k]

        # update
        for k in observed:
            if k in self.voxel_map:
                self.voxel_map[k] = self.decay * self.voxel_map[k] + (1 - self.decay)
            else:
                self.voxel_map[k] = 1 - self.decay


    # from pose to transformation matrix
    def pose_to_matrix(self, pose_msg):
        p = pose_msg.pose.position
        q = pose_msg.pose.orientation

        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = p.x
        T[1, 3] = p.y
        T[2, 3] = p.z

        return T


    # pc transform
    def transform_points(self, points, T):
        xyz = points[:, :3]

        ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_h = np.hstack((xyz, ones))

        xyz_transformed = (T @ xyz_h.T).T[:, :3]

        result = points.copy()
        result[:, :3] = xyz_transformed
        return result


    # creating the pc from points
    def create_cloud_xyzi(self, header, pts):

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        return pc2.create_cloud(header, fields, pts.tolist())


    # main processing
    def cloud_callback(self, msg):

        if self.curr_pose is None and self.transform_correction:
            self.get_logger().warn("No pose received yet")
            return

        points = self.cloud_to_numpy(msg)
        if len(points) == 0:
            return

        xyz = points[:, :3]
        intensities = points[:, 3]

        voxel_keys = self.compute_voxel_keys(xyz)
        self.update_voxel_map(voxel_keys)

        dynamic_mask = np.zeros(len(points), dtype=bool)

        if self.prev_points is not None:

            if self.transform_correction and self.prev_pose is not None:
                # alignment
                T_prev = self.pose_to_matrix(self.prev_pose)
                T_curr = self.pose_to_matrix(self.curr_pose)

                T_rel = np.linalg.inv(T_curr) @ T_prev

                prev_aligned = self.transform_points(self.prev_points, T_rel)

                prev_xyz = prev_aligned[:, :3]
                prev_int = prev_aligned[:, 3]

            else:
                prev_xyz = self.prev_points[:, :3]
                prev_int = self.prev_points[:, 3]

            tree = cKDTree(prev_xyz)

            dists, idx = tree.query(xyz, k=1)

            matched_intensity = prev_int[idx]

            frame_inconsistent = dists > self.frame_dist_thresh

            intensity_diff = np.abs(intensities - matched_intensity)
            intensity_inconsistent = intensity_diff > self.intensity_thresh

            strong_dynamic = np.logical_or(frame_inconsistent, intensity_inconsistent)

            voxel_stability = np.array([
                self.voxel_map.get(k, 0.0) for k in voxel_keys
            ])

            voxel_unstable = voxel_stability < self.stability_thresh

            # raw_dynamic = np.logical_or(
            #     strong_dynamic,
            #     np.logical_and(voxel_unstable, frame_inconsistent)
            # )

            raw_dynamic = dists > self.frame_dist_thresh

            for i, k in enumerate(voxel_keys):

                score = self.dynamic_score_map.get(k, 0.0)

                if raw_dynamic[i]:
                    score += self.alpha
                else:
                    score -= self.beta

                score = np.clip(score, 0.0, 1.0)
                self.dynamic_score_map[k] = score

                dynamic_mask[i] = score > self.dynamic_threshold

            # stable override
            stable_override = voxel_stability > 0.9
            dynamic_mask[stable_override] = False

        dyn_pts = points[dynamic_mask]
        sta_pts = points[~dynamic_mask]

        header = msg.header

        self.pub_dyn.publish(self.create_cloud_xyzi(header, dyn_pts))
        self.pub_sta.publish(self.create_cloud_xyzi(header, sta_pts))

        self.prev_points = points

        if self.transform_correction:
            self.prev_pose = self.curr_pose


def main(args=None):
    rclpy.init(args=args)
    node = DynamicFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()