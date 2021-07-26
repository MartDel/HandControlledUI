import cv2, time
import numpy as np
import hand_tracking_module as htm

RESOLUTION = (1080, 1920)
CAM_ID = 1
FULLSCREEN = True

# Init video capture
capture = cv2.VideoCapture(CAM_ID)
capture.set(3, RESOLUTION[1])
capture.set(4, RESOLUTION[0])

tracker = htm.HandTracker(detection_con=0.85)

# Loop through captured images
while True:
    success, img = capture.read()
    if success == False: break
    img = cv2.flip(img, 1)
    canvas = np.zeros((RESOLUTION[0], RESOLUTION[1], 3), np.uint8)

    # Track hands
    hands = tracker.findHands(img, default_show=False, custom_show=True, drawing_image=canvas)

    # Show the current frame with the current frame rate
    # cv2.imshow("Image", img)

    if FULLSCREEN:
        cv2.namedWindow("Hands", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Hands", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Hands", canvas)

    # Check if the loop must stop
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Exit the program
capture.release()
cv2.destroyAllWindows()
