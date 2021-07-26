import time
import cv2 as cv
import mediapipe as mp

RESOLUTION = (1080, 1920)

class HandTracker():
    """ HandTracker module. """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """ Init HandTracker object. """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_con, self.track_con)
        self.mpDraw = mp.solutions.drawing_utils

        self.tips_id = [4, 8, 12, 16, 20]

    def findHands(self, img, default_show=True, custom_show=False, drawing_image=[]):
        """ Try to find hands into the given image. """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if len(drawing_image) == 0:
            drawing_image = img
        if self.results.multi_hand_landmarks and (default_show or custom_show):
            if default_show:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            elif custom_show:
                for i in range(len(self.results.multi_hand_landmarks)):
                    lms = self.findHandPosition(RESOLUTION, hand_id=i)
                    # Print custom fingers connections
                    for finger in range(5, 22, 4):
                        if finger != 21:
                            finger_lms = []
                            for j in range(finger, 4+finger):
                                finger_lms.append(j)
                            finger_lms[:0] = [0]
                        else:
                            finger_lms = []
                            for j in range(0, 5):
                                finger_lms.append(j)
                        for lm_id in range(len(finger_lms)):
                            if lm_id != len(finger_lms) - 1:
                                lm, lm_next = finger_lms[lm_id], finger_lms[lm_id+1]
                                cv.line(drawing_image, lms[lm][1:], lms[lm_next][1:], (255, 255, 255), 2)
        return self.results.multi_hand_landmarks

    def findHandPosition(self, resolution, hand_id=0):
        """ Get given hand landmarks in the given image. """
        self.landmarks = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_id]
            for id, lm in enumerate(hand.landmark):
                h, w = resolution
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks.append([id, cx, cy])
        return self.landmarks

    def fingersUp(self):
        """ Check which fingers are up. """
        self.fingers_up = []
        for finger_id in range(0, 5):
            nb_finger_lm = 2
            if finger_id == 0: nb_finger_lm = 1
            if self.landmarks[self.tips_id[finger_id]][nb_finger_lm] <= self.landmarks[self.tips_id[finger_id] - nb_finger_lm][nb_finger_lm]:
                self.fingers_up.append(finger_id)
        return self.fingers_up


def main():
    capture = cv.VideoCapture(0)
    p_time = 0
    c_time = 0
    tracker = HandTracker()

    # Loop through captured images
    while True:
        success, img = capture.read()
        if success == False: break

        # Track hands
        img = tracker.findHands(img)
        lms = tracker.findHandPosition(img)
        if len(lms) != 0:
            print(lms[0])

        # Estimate frame rate
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time

        # Show the current frame with the current frame rate
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)

        # Check if the loop must stop
        if cv.waitKey(1) & 0xFF == ord('q'): break

    # Exit the program
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
