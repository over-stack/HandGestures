import sys
import math
import time
import json
import yaml
import struct
from typing import Tuple, Union, List

import numpy as np
import cv2 as cv
import win32pipe, win32file, pywintypes

import mediapipe as mp
from mediapipe.tasks import python
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from mediapipe.framework.formats import landmark_pb2


with open('params.yaml') as f:
    params = yaml.safe_load(f)


DEFAULT_COORD_VALUE = params['DEFAULT_COORD_VALUE']
RELATIVE_COORDINATES = params['RELATIVE_COORDINATES']
FRONTAL_FACE_PARAM = params['FRONTAL_FACE_PARAM']
MIN_RELATIVE_FACE_SQUARE = params['MIN_RELATIVE_FACE_SQUARE']
NUM_TOP_CLOSEST_FACES = params['NUM_TOP_CLOSEST_FACES']
VISUALIZE_HAND_LANDMARKS = params['VISUALIZE_HAND_LANDMARKS']
VISUALIZE_FACE_DETECTIONS = params['VISUALIZE_FACE_DETECTIONS']
PIPE_NAME = params['PIPE_NAME']
BUFFER_SIZE = params['BUFFER_SIZE']
# FPS = params['FPS']

MAKE_SERVER = params['MAKE_SERVER']
PRINT_DATA = params['PRINT_DATA']

gesture_model_path = "palm_gesture_model.task"
face_model_path = "blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
FaceDetection = python.components.containers.Detection

VisionRunningMode = mp.tasks.vision.RunningMode


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int,
                                     keep_float: bool = False) -> Union[None, Tuple[int, int], Tuple[float, float]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    if keep_float:
        x_px = float(x_px)
        y_px = float(y_px)

    return x_px, y_px


class App:
    def __init__(self):
        self.pipe = None

        self.frame_shape = None
        self.frame_square = None

        self.gesture_recognizer_result = None
        self.face_detector_result = None

        self.num_hands = 0
        self.valid_hands = [False, False]
        self.center_mass_x = [DEFAULT_COORD_VALUE, DEFAULT_COORD_VALUE]
        self.center_mass_y = [DEFAULT_COORD_VALUE, DEFAULT_COORD_VALUE]

        self.num_faces = 0
        self.faces_pos_x = [DEFAULT_COORD_VALUE] * NUM_TOP_CLOSEST_FACES
        self.faces_pos_y = [DEFAULT_COORD_VALUE] * NUM_TOP_CLOSEST_FACES
        self.frontal_faces = [False] * NUM_TOP_CLOSEST_FACES
        self.close_faces = [False] * NUM_TOP_CLOSEST_FACES

    def main(self):
        gesture_recognizer_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_model_path),
            num_hands=2,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.gesture_recognizer_result_callback
        )

        face_detector_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.face_detector_result_callback
        )

        if MAKE_SERVER:
            self.pipe = win32pipe.CreateNamedPipe(
                r'\\.\pipe\\' + PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1, BUFFER_SIZE, BUFFER_SIZE,
                0,
                None)

            print("Waiting for client...")
            win32pipe.ConnectNamedPipe(self.pipe, None)
            print("Client connected!")

        cam = cv.VideoCapture(0)

        timestamp = 0
        '''new_frame_time = time.time()
        prev_frame_time = new_frame_time'''
        with (GestureRecognizer.create_from_options(gesture_recognizer_options) as gesture_recognizer,
              FaceDetector.create_from_options(face_detector_options) as face_detector):
            while cam.isOpened():
                success, flipped_frame_bgr = cam.read()
                if not success:
                    if PRINT_DATA:
                        print("Camera Frame is not available")
                    continue

                self.frame_shape = flipped_frame_bgr.shape
                self.frame_square = self.frame_shape[0] * self.frame_shape[1]

                timestamp += 1
                
                frame_bgr = cv.flip(flipped_frame_bgr, 1)
                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                gesture_recognizer.recognize_async(mp_frame, timestamp)
                face_detector.detect_async(mp_frame, timestamp)

                if MAKE_SERVER:
                    self.send_response()

                if VISUALIZE_HAND_LANDMARKS:
                    frame_rgb = self.draw_hand_landmarks(frame_rgb, self.gesture_recognizer_result)
                if VISUALIZE_FACE_DETECTIONS:
                    frame_rgb = self.draw_face_detections(frame_rgb, self.face_detector_result)
                frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
                cv.imshow('Show', frame_bgr)

                '''new_frame_time = time.time()
                if PRINT_DATA:
                    print(f"FPS: {1 / (new_frame_time - prev_frame_time)}")
                prev_frame_time = new_frame_time'''

                if cv.waitKey(20) & 0xff == ord('q'):
                    break

        cam.release()
        cv.destroyAllWindows()

    def draw_hand_landmarks(self, frame: np.ndarray, recognition_result: GestureRecognizerResult) -> np.ndarray:
        if recognition_result is None:
            return frame

        multi_hand_landmarks = recognition_result.hand_landmarks

        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            if recognition_result.handedness[i][0].category_name == 'Left':
                hand_idx = 1
            else:
                hand_idx = 0

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            if not self.valid_hands[hand_idx]:
                hand_landmarks_style = mp_drawing_styles.get_default_hand_landmarks_style()
                hand_connections_style = mp_drawing_styles.get_default_hand_connections_style()
            else:
                hand_landmarks_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                hand_connections_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                hand_landmarks_style,
                hand_connections_style
            )

        return frame

    def draw_face_detections(self, frame: np.ndarray, detection_result: FaceDetectorResult) -> np.ndarray:
        if detection_result is None:
            return frame

        for i, detection in enumerate(detection_result.detections):
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

            if self.close_faces[i]:
                rectangle_color = (0, 255, 0)
            else:
                rectangle_color = (255, 0, 0)
            cv.rectangle(frame, start_point, end_point, rectangle_color, 2)

            if self.frontal_faces[i]:
                keypoints_color = (0, 255, 0)
                border_line_color = (0, 255, 100)
            else:
                keypoints_color = (255, 0, 0)
                border_line_color = (255, 0, 100)

            for i, keypoint in enumerate(detection.keypoints):
                # except nose and mouth keypoints
                if i == 2 or i == 3:
                    continue
                keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                               self.frame_shape[1], self.frame_shape[0])
                cv.circle(frame, keypoint_px, 2, keypoints_color, 2)

            # center_mass line
            center_mass_x = (detection.keypoints[0].x + detection.keypoints[1].x +
                             detection.keypoints[-1].x + detection.keypoints[-2].x) / 4
            center_mass_x, _ = _normalized_to_pixel_coordinates(center_mass_x, 0,
                                                                self.frame_shape[1], self.frame_shape[0])
            cv.line(frame, (center_mass_x, start_point[1]), (center_mass_x, end_point[1]),
                    (255, 0, 255), 2)

            # left/right borders
            nose_x, _ = _normalized_to_pixel_coordinates(detection.keypoints[2].x, 0,
                                                         self.frame_shape[1], self.frame_shape[0])
            left_border_x = nose_x - int(FRONTAL_FACE_PARAM * bbox.width)
            cv.line(frame, (left_border_x, start_point[1]), (left_border_x, end_point[1]),
                    border_line_color, 2)
            right_border_x = nose_x + int(FRONTAL_FACE_PARAM * bbox.width)
            cv.line(frame, (right_border_x, start_point[1]), (right_border_x, end_point[1]),
                    border_line_color, 2)

        return frame

    def gesture_recognizer_result_callback(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.gesture_recognizer_result = result

        self.num_hands = 0
        for i in range(2):
            self.valid_hands[i] = False
            self.center_mass_x[i] = DEFAULT_COORD_VALUE
            self.center_mass_y[i] = DEFAULT_COORD_VALUE

        for i, gesture in enumerate(result.gestures):
            if result.handedness[i][0].category_name == 'Left':
                hand_idx = 1
            else:
                hand_idx = 0

            if gesture[0].category_name == "palm":
                self.num_hands += 1
                self.valid_hands[hand_idx] = True
            else:
                continue

            self.center_mass_x[hand_idx] = 0
            self.center_mass_y[hand_idx] = 0
            count = 0
            for hand_landmark in result.hand_landmarks[i]:
                self.center_mass_x[hand_idx] += hand_landmark.x
                self.center_mass_y[hand_idx] += hand_landmark.y
                count += 1
            self.center_mass_x[hand_idx] /= count
            self.center_mass_y[hand_idx] /= count

        if not RELATIVE_COORDINATES:
            for i in range(2):
                if self.center_mass_x[i] == DEFAULT_COORD_VALUE or self.center_mass_y[i] == DEFAULT_COORD_VALUE:
                    continue

                self.center_mass_x[i], self.center_mass_y[i] = _normalized_to_pixel_coordinates(
                    self.center_mass_x[i], self.center_mass_y[i], self.frame_shape[1], self.frame_shape[0],
                    keep_float=True
                )

        if PRINT_DATA:
            print(f"Num_hands: {self.num_hands}. " +
                  f"Pos_left: ({self.center_mass_y[0]}, {self.center_mass_x[0]}). " +
                  f"Pos_right: ({self.center_mass_y[1]}, {self.center_mass_x[1]}).")

    def face_detector_result_callback(self, result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        result.detections = sorted(result.detections,
                                   key=lambda d: d.bounding_box.width * d.bounding_box.height / self.frame_square,
                                   reverse=True)
        result.detections = result.detections[:min(len(result.detections), NUM_TOP_CLOSEST_FACES)]
        self.face_detector_result = result

        self.num_faces = len(result.detections)

        if PRINT_DATA:
            print(f"Num_faces: {self.num_faces}.")

        for i, detection in enumerate(result.detections):
            self.faces_pos_y[i] = float(math.floor(detection.bounding_box.origin_y + detection.bounding_box.height / 2))
            self.faces_pos_x[i] = float(math.floor(detection.bounding_box.origin_x + detection.bounding_box.width / 2))
            if RELATIVE_COORDINATES:
                self.faces_pos_y[i] /= self.frame_shape[0]
                self.faces_pos_x[i] /= self.frame_shape[1]
            self.frontal_faces[i] = self.check_face_is_frontal(detection)
            relative_face_square = detection.bounding_box.width * detection.bounding_box.height / self.frame_square
            self.close_faces[i] = relative_face_square > MIN_RELATIVE_FACE_SQUARE

            if PRINT_DATA:
                print(f"Face: {i}. Frontal: {self.frontal_faces[i]}. Close: {self.close_faces[i]}.")

        for i in range(self.num_faces, NUM_TOP_CLOSEST_FACES):
            self.faces_pos_x[i] = DEFAULT_COORD_VALUE
            self.faces_pos_y[i] = DEFAULT_COORD_VALUE
            self.frontal_faces[i] = False
            self.close_faces[i] = False

    def check_face_is_frontal(self, detection: FaceDetection) -> bool:
        face_kp = detection.keypoints
        center_mass_x = (face_kp[0].x + face_kp[1].x + face_kp[-1].x + face_kp[-2].x) / 4
        nose_x = face_kp[2].x
        relative_face_width = detection.bounding_box.width / self.frame_shape[1]
        if nose_x - relative_face_width * FRONTAL_FACE_PARAM <= center_mass_x <= nose_x + relative_face_width * FRONTAL_FACE_PARAM:
            return True
        return False

    def send_response(self):
        """
        faces_params = []
        for i in range(NUM_TOP_CLOSEST_FACES):
            faces_params.extend([self.faces_pos_y[i], self.faces_pos_x[i], self.frontal_faces[i], self.close_faces[i]])

        data = struct.pack(f"HddddH" + 'dd??' * NUM_TOP_CLOSEST_FACES, self.num_hands,
                           self.center_mass_y[0], self.center_mass_x[0], self.center_mass_y[1], self.center_mass_x[1],
                           self.num_faces, *faces_params)

        win32file.WriteFile(self.pipe, data)
        """

        response_dict = {
            "Head": self.frontal_faces[0] and self.close_faces[0],
            "LeftHand": self.valid_hands[0],
            "RightHand": self.valid_hands[1],
            "LeftHandCoordX": self.center_mass_x[0],
            "LeftHandCoordY": self.center_mass_y[0],
            "RightHandCoordX": self.center_mass_x[1],
            "RightHandCoordY": self.center_mass_y[1],
        }
        response_str = json.dumps(response_dict)
        # print(response_str)
        # print(sys.getsizeof(response_str))
        win32file.WriteFile(self.pipe, response_str.encode())


if __name__ == '__main__':
    app = App()
    app.main()
