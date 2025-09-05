import cv2
import numpy as np
import time

# This class implements a simple Kalman Filter for tracking 2D screen positions

class Tracker():

    def __init__(self, id: int=0, initial_position: tuple[float, float, float, float]=(0, 0, 0, 0)):
        self.id = id
        self.pos = initial_position
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurement variables
        # x, y, dx, dy
        self.q = 0.2 # increase process noise to slow prediction
        self.last_update_time = None

        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
        ], np.float32) * self.q

        # Lower measurement noise to make filter snap to measurements
        self.kf.measurementNoiseCov = np.array([
            [0.001, 0],
            [0, 0.001]
        ], np.float32)

        self.kf.statePre = np.array([[initial_position[0]], [initial_position[1]], [0], [0]], np.float32)
        self.kf.statePost = np.array([[initial_position[0]], [initial_position[1]], [0], [0]], np.float32)

    
    def update(self, measurement: tuple[float, float], timestamp: float = None) -> tuple[float, float, float, float]:
        """
        Update the Kalman Filter with a new measurement and return the predicted position.
        Optionally provide a timestamp (in seconds) for time-aware updates.
        """
        if timestamp is None:
            timestamp = time.time()
        if self.last_update_time is None:
            dt = 1.0
        else:
            dt = max(timestamp - self.last_update_time, 1e-3)  # avoid zero
        self.last_update_time = timestamp

        # Update transition matrix with dt, dampen velocity effect
        velocity_damping = 0.3  # lower = slower movement
        self.kf.transitionMatrix = np.array([
            [1, 0, velocity_damping * dt, 0],
            [0, 1, 0, velocity_damping * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        meas = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kf.correct(meas)
        pred = self.kf.predict()
        self.pos = (pred[0][0], pred[1][0], pred[2][0], pred[3][0])
        return self.pos