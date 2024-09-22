import os

import cv2
import numpy as np
from tqdm import tqdm


class VisualOdometry:
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, "calib.txt"))
        self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images = self._load_images(os.path.join(data_dir, "image_l"))

        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(
            indexParams=index_params, searchParams=search_params
        )

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, "r") as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=" ")
            # projection matrix is loaded from file
            P = np.reshape(params, (3, 4))
            # intrinsic camera matrix is extracted from P
            K = P[0:3, 0:3]

        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []

        with open(filepath, "r") as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=" ")
                """
                Each transformation matrix T represents 
                the pose of the camera in current image
                """
                T = T.reshape(3, 4)
                # Appending Homogeneous Coordinate
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)

        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [
            os.path.join(filepath, file) for file in sorted(os.listdir(filepath))
        ]

        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _make_transformation_matrix(R, t):
        """
        Makes a transformation matrix from the given 
        rotation matrix R and translation vector t
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, frame_index, matched_images_output_directory):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[frame_index - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[frame_index], None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches that do not have too high a distance
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)

        num_matches_to_draw = min(10, len(good))
        some_good = good[:num_matches_to_draw]

        from utils.plotting import draw_matches
        matched_img = draw_matches(
            self.images[frame_index - 1], kp1,
            self.images[frame_index], kp2,
            some_good,
            orientation='vertical'
        )

        # Save the matched image
        matched_img_path = os.path.join(matched_images_output_directory, f"matched_frame_{frame_index}.png")
        cv2.imwrite(matched_img_path, matched_img)

        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        # matched_img = ResizeWithAspectRatio(matched_img, width=350, inter=cv2.INTER_LANCZOS4)
        cv2.imshow("Matches", matched_img)
        cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        _, R, t, _ = cv2.recoverPose(E, q1, q2, cameraMatrix=self.K)

        # Get transformation matrix
        T = self._make_transformation_matrix(R, np.squeeze(t))

        return T


def main():
    from utils.common import clear_directory
    matched_images_output_directory = "output/matched_images"
    clear_directory(matched_images_output_directory)

    data_dir = "data/KITTI_sequence_2"
    vo = VisualOdometry(data_dir)

    # from utils.plotting import play_trip
    # play_trip(vo.images)

    gt_path = []
    estimated_path = []

    """
    current_pose = np.array([
        [r11, r12, r13, tx],
        [r21, r22, r23, ty],
        [r31, r32, r33, tz],
        [0,   0,   0,   1]
    ])
    """
    current_pose = []

    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            current_pose = gt_pose  # 4 x 4 matrix
        else:
            q1, q2 = vo.get_matches(i, matched_images_output_directory)
            T = vo.get_pose(q1, q2)

            current_pose = np.matmul(current_pose, np.linalg.inv(T))

        """
        By picking only the x and z coordinates, you're effectively creating 
        a 2D representation of the path in the x-z plane. I want to visualize 
        movement in a horizontal plane.
        
        Camera's Local Frame:

        X-axis: Points to the right of the camera.
        Y-axis: Points downward
        Z-axis: Points forward, in the direction the camera is facing.
        
        """
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((current_pose[0, 3], current_pose[2, 3]))

    cv2.destroyAllWindows()

    from utils.plotting import visualize_paths
    visualize_paths(
        gt_path,
        estimated_path,
        "Visual Odometry",
        file_out=f"output/{os.path.basename(data_dir)}.html",
    )


if __name__ == "__main__":
    main()
