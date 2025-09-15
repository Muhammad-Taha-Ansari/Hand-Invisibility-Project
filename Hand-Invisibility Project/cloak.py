import cv2
import numpy as np
import math
import time

def get_background(cap, num_frames=60):
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))
    if not frames:
        return None
    avg = np.mean(frames, axis=0).astype(np.uint8)
    return avg

def count_fingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)
        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c + 1e-5)) * 57

        if angle <= 90 and d > 10000:
            finger_count += 1

    return finger_count + 1

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("📷 Capturing background... stay still for 2–3 seconds.")
    background = get_background(cap, 80)
    if background is None:
        print("❌ Background capture failed")
        return
    print("✅ Background captured!")

    is_invisible = False
    last_toggle_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Hand ROI
        roi = frame[50:350, 50:350]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Skin color mask
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fingers = 0
        if contours:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(cnt)
            if area > 5000:  # filter noise
                fingers = count_fingers(cnt)
                cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)

        # Toggle invisibility with 1 finger
        if fingers == 1 and time.time() - last_toggle_time > 1.5:
            is_invisible = not is_invisible
            print("🔄 Toggled:", "Invisible" if is_invisible else "Visible")
            last_toggle_time = time.time()

        if is_invisible:
            # Background subtraction
            diff = cv2.absdiff(background, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            mask_inv = cv2.bitwise_not(mask)
            fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            bg = cv2.bitwise_and(background, background, mask=mask)
            final = cv2.add(fg, bg)   # No transparency (alpha = 0.0)
        else:
            final = frame

        # ROI box
        cv2.rectangle(final, (50, 50), (350, 350), (255, 0, 0), 2)
        cv2.putText(final, f"Mode: {'Invisible' if is_invisible else 'Visible'}",
                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("🪄 Invisibility Cloak", final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
