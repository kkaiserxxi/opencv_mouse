import cv2qqd dcvv-
import mediapipe as mp
import pyautogui
import time

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen width and height
screen_width, screen_height = pyautogui.size()

# Sensitivity factor for cursor movement
sensitivity = 1.5  # Increase this value to make the mouse more sensitive

# Variable to store the last click time
last_click_time = 0

try:
    while True:
        # Capture a frame from the video feed
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror view
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
        output = hand_detector.process(rgb_frame)  # Process the frame to detect hands
        hands = output.multi_hand_landmarks  # Extract hand landmarks

        if hands:
            for i, hand in enumerate(hands):
                # Get hand classification (left or right)
                hand_type = output.multi_handedness[i].classification[0].label

                # Draw hand landmarks and connections
                drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                landmarks = hand.landmark

                if hand_type == "Right":
                    # Right hand controls the cursor
                    for id, landmark in enumerate(landmarks):
                        if id == 8:  # Index finger tip
                            x = int(landmark.x * frame_width)
                            y = int(landmark.y * frame_height)
                            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                            index_x = screen_width / frame_width * x * sensitivity
                            index_y = screen_height / frame_height * y * sensitivity
                            pyautogui.moveTo(index_x, index_y)

                elif hand_type == "Left":
                    # Left hand performs clicking
                    middle_x, middle_y = 0, 0
                    thumb_x, thumb_y = 0, 0

                    for id, landmark in enumerate(landmarks):
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)

                        if id == 12:  # Middle finger tip
                            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                            middle_x = screen_width / frame_width * x * sensitivity
                            middle_y = screen_height / frame_height * y * sensitivity

                        if id == 4:  # Thumb tip
                            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                            thumb_x = screen_width / frame_width * x * sensitivity
                            thumb_y = screen_height / frame_height * y * sensitivity

                    # Check distance between middle finger tip and thumb tip for the left hand
                    current_time = time.time()
                    if abs(middle_x - thumb_x) < 20 and abs(middle_y - thumb_y) < 20:
                        if current_time - last_click_time > 0.5:  # 0.5 seconds debounce time
                            pyautogui.click()  # Perform a click
                            last_click_time = current_time

        # Display the frame
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
            break

finally:
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
