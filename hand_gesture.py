import cv2
import mediapipe as mp
import time

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

def recognize_gesture(hand):
    fingers = []

    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    for tip, joint in zip(tips, joints):
        fingers.append(hand[tip].y < hand[joint].y)

    thumb = hand[4].x < hand[3].x

    if all(fingers) and thumb:
        return "OPEN PALM"

    elif not any(fingers) and not thumb:
        return "FIST"

    elif thumb and not any(fingers):
        return "THUMBS UP"

    else:
        return "UNKNOWN"


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not found!")
    exit()

start_time = time.monotonic_ns() // 1000000
last_timestamp = -1

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = (
                time.monotonic_ns() // 1000000
                - start_time
            )

            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = detector.detect_for_video(
                image, timestamp
            )

            gesture_text = "No Hand Detected"

            if result.hand_landmarks:

                for hand in result.hand_landmarks:

                    h, w, _ = frame.shape

                    for point in hand:
                        x = int(point.x * w)
                        y = int(point.y * h)

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

                    gesture_text = recognize_gesture(hand)

            cv2.putText(
                frame,
                gesture_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Hand Gesture Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()