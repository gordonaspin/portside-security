import numpy as np

class KalmanFilter:
    """
    Lightweight Kalman filter for tracking bounding boxes.
    State vector (8D):
        [cx, cy, area, ratio, vx, vy, va, vr]
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation model
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation noise
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """
        Create initial state from measurement [cx, cy, area, ratio].
        """
        mean = np.zeros(8)
        mean[:4] = measurement

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[2],
            1e-2,
            1e-2,
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[2],
            1e-5,
            1e-5,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Predict next state.
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[2],
            1e-2,
            1e-2,
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[2],
            1e-5,
            1e-5,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Update state with measurement.
        """
        projected_mean = self._update_mat @ mean
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T
        )

        innovation = measurement - projected_mean
        S = projected_cov
        K = covariance @ self._update_mat.T @ np.linalg.inv(S)

        new_mean = mean + K @ innovation
        new_cov = covariance - K @ S @ K.T
        return new_mean, new_cov
