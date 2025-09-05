
import numpy as np
import cv2
import time
from ultralytics.utils.plotting import Annotator
from kalman_filter import Tracker


class SimpleFPS:
    def __init__(self):
        self.start_time = time.time()
        self.display_time_sec = 1  # update fps display
        self.fps = 0
        self.frame_counter = 0
        self.is_fps_updated = False

    def get_fps(self) -> tuple[float, bool]:
        elapsed = time.time() - self.start_time
        self.frame_counter += 1
        is_fps_updated = False

        if elapsed > self.display_time_sec:
            self.fps = self.frame_counter / elapsed
            self.frame_counter = 0
            self.start_time = time.time()
            is_fps_updated = True

        return self.fps, is_fps_updated


def draw_fps(img, fps) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    # putting the FPS count on the frame
    cv2.putText(img, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

def draw_annotation(img, label_names, results, tracker) -> Annotator:
    annotator = None
    # Load camera calibration results
    try:
        calib = np.load('./inference/camera_calibration.npz')
        camera_matrix = calib['camera_matrix']
        dist_coeffs = calib['dist_coeffs']
    except Exception as e:
        camera_matrix = None
        dist_coeffs = None
        print(f"Calibration file not found or invalid: {e}")

    found_box = False
    measurement = None
    timestamp = time.time()

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes

        # Always show Kalman filter prediction in pink
        pred = tracker.update(tracker.pos[:2], timestamp)
        x_pred, y_pred = int(pred[0]), int(pred[1])
        print(f"KF predicted position: ({x_pred}, {y_pred})")

        cv2.circle(img, (x_pred, y_pred), 8, (255, 0, 255), -1)  # pink
        cv2.putText(img, "KF predicted", (x_pred+10, y_pred), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, label_names[int(c)])
            # Draw a dot/circle at the center of the box
            x_center = int((b[0] + b[2]) / 2)
            y_center = int((b[1] + b[3]) / 2)
            print(f"Box center: ({x_center}, {y_center})")
            cv2.circle(img, (x_center, y_center), 6, (0, 0, 255), -1)
            # tracker = Tracker(id=0, initial_position=(x_center, y_center, 0, 0))
            found_box = True
            measurement = (x_center, y_center)
            D = 67 # true tennis ball diameter mm approx
            # Use solvePnP if calibration is available
            if camera_matrix is not None and dist_coeffs is not None:
                pos = solvepnp_from_bbox(b, img.shape, camera_matrix, dist_coeffs, D)
                if pos is not None:
                    x, y, z = pos
                    # print(f'SolvePnP ball position (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}')
                    cv2.putText(img, f"SolvePnP: x={x:.1f}, y={y:.1f}, z={z:.1f}", 
                                (x_center-140, y_center+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    print('SolvePnP failed for this box.')
            else:
                # Fallback: use pinhole model
                f = 4.74 # focal length in mm
                x, y, z = estimate_ball_position(b, img.shape, f, D)
                print(f'Estimated ball position (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}')
                cv2.putText(img, f"Estimated: x={x:.1f}, y={y:.1f}, z={z:.1f}", 
                            (x_center+20, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # After processing boxes, update Kalman filter if measurement is available
    if found_box and measurement is not None:
        tracker.update(measurement, timestamp)

    if annotator is not None:
        annotated_img = annotator.result()
    else:
        annotated_img = img.copy()

    return annotated_img

def estimate_ball_position(b, image_shape, focal_length_mm, ball_diameter_mm) -> tuple[float, float, float]:
    """
    ** highly inaccurate **
    Estimate the 3D position (x, y, z) of the ball relative to the camera.
    b: bounding box [x1, y1, x2, y2]
    image_shape: (height, width)
    focal_length_mm: focal length in mm
    ball_diameter_mm: true diameter of the ball in mm
    Returns: (x, y, z) in mm
    """
    # Center of bounding box in pixels
    x_center = (b[0] + b[2]) / 2
    y_center = (b[1] + b[3]) / 2
    w = b[2] - b[0]  # width in pixels
    h = b[3] - b[1]  # height in pixels
    img_h, img_w = image_shape[:2]

    # Depth (z) estimation using pinhole camera model
    # Z = (focal_length * true_diameter) / observed_diameter_in_pixels
    Z = (focal_length_mm * ball_diameter_mm) / h if h > 0 else 0

    # Relative x, y in mm (assuming principal point is at image center)
    # x = (x_center - cx) * Z / f, y = (y_center - cy) * Z / f
    cx = img_w / 2
    cy = img_h / 2
    X = ((x_center - cx) * Z) / focal_length_mm
    Y = ((y_center - cy) * Z) / focal_length_mm

    return X, Y, Z

def solvepnp_from_bbox(b, image_shape, camera_matrix, dist_coeffs, ball_diameter_mm) -> np.ndarray | None:
    """
    Estimate 3D position of ball center using solvePnP.
    b: [x1, y1, x2, y2] bounding box
    image_shape: (height, width)
    camera_matrix: from calibration
    dist_coeffs: from calibration
    ball_diameter_mm: true diameter
    Returns: (x, y, z) in mm (camera coordinates) or None if failed
    """
    D = ball_diameter_mm / 2
    object_points = np.array([
        [0, 0, 0],      # center
        [ D, 0, 0],     # right
        [-D, 0, 0],     # left
        [0,  D, 0],     # top
        [0, -D, 0],     # bottom
    ], dtype=np.float32)

    x1, y1, x2, y2 = b
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    image_points = np.array([
        [x_center, y_center],         # center
        [x2, y_center],               # right
        [x1, y_center],               # left
        [x_center, y1],               # top
        [x_center, y2],               # bottom
    ], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if success:
        return tvec.flatten()
    else:
        return None