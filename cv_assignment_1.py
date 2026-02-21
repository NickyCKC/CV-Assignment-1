import cv2
import numpy as np

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()  # Capture each frame
        if not ret:
            break

        # Shrink the frame to 320x240, to prevent big window once concatenated
        frame = cv2.resize(frame, (320, 240))

        # Split BGR channels
        b, g, r = cv2.split(frame)

        # Convert to grayscale and HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Split HSV channels
        h, s, v = cv2.split(hsv)

        # Calculate histograms for BGR, grayscale and HSV
        b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
        g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
        r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        h_hist = cv2.calcHist([h], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([s], [0], None, [256], [0, 256])
        v_hist = cv2.calcHist([v], [0], None, [256], [0, 256])

        # Normalize histogram for plotting
        b_hist = cv2.normalize(b_hist, b_hist, 0, 400, cv2.NORM_MINMAX)
        g_hist = cv2.normalize(g_hist, g_hist, 0, 400, cv2.NORM_MINMAX)
        r_hist = cv2.normalize(r_hist, r_hist, 0, 400, cv2.NORM_MINMAX)
        gray_hist = cv2.normalize(gray_hist, gray_hist, 0, 400, cv2.NORM_MINMAX)
        h_hist = cv2.normalize(h_hist, h_hist, 0, 400, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 400, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 400, cv2.NORM_MINMAX)

        # Flatten histogram for plotting
        b_hist = b_hist.flatten()
        g_hist = g_hist.flatten()
        r_hist = r_hist.flatten()
        gray_hist = gray_hist.flatten()
        h_hist = h_hist.flatten()
        s_hist = s_hist.flatten()
        v_hist = v_hist.flatten()

        # Plot the histogram
        hist_w, hist_h = 320, 240
        bin_w = int(hist_w / 256)
        original_hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        gray_hist_img = np.zeros((hist_h, hist_w), dtype=np.uint8)
        hsv_hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        for i in range(1, 256):
            cv2.line(original_hist_img,
                    (bin_w * (i - 1), hist_h - int(b_hist[i - 1])),
                    (bin_w * i, hist_h - int(b_hist[i])),
                    (255, 0, 0), thickness=2)  # Blue for B
            cv2.line(original_hist_img,
                    (bin_w * (i - 1), hist_h - int(g_hist[i - 1])),
                    (bin_w * i, hist_h - int(g_hist[i])),
                    (0, 255, 0), thickness=2)  # Green for G
            cv2.line(original_hist_img,
                    (bin_w * (i - 1), hist_h - int(r_hist[i - 1])),
                    (bin_w * i, hist_h - int(r_hist[i])),
                    (0, 0, 255), thickness=2)  # Red for R
            cv2.line(gray_hist_img,
                    (bin_w * (i - 1), hist_h - int(gray_hist[i - 1])),
                    (bin_w * i, hist_h - int(gray_hist[i])),
                    (255,), thickness=2)
            cv2.line(hsv_hist_img,
                    (bin_w * (i - 1), hist_h - int(h_hist[i - 1])),
                    (bin_w * i, hist_h - int(h_hist[i])),
                    (255, 0, 0), thickness=2)  # H
            cv2.line(hsv_hist_img,
                    (bin_w * (i - 1), hist_h - int(s_hist[i - 1])),
                    (bin_w * i, hist_h - int(s_hist[i])),
                    (0, 255, 0), thickness=2)  # S
            cv2.line(hsv_hist_img,
                    (bin_w * (i - 1), hist_h - int(v_hist[i - 1])),
                    (bin_w * i, hist_h - int(v_hist[i])),
                    (0, 0, 255), thickness=2)  # V

        # Convert the 1-channel grayscale image and histogram to a 3-channel BGR image in order to use hconcat
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_hist_rgb = cv2.cvtColor(gray_hist_img, cv2.COLOR_GRAY2BGR)
        
        # Now concatenate them since they all have 3 channels!
        row1 = cv2.hconcat([frame, gray_rgb, hsv])
        row2 = cv2.hconcat([original_hist_img, gray_hist_rgb, hsv_hist_img])
        dashboard = cv2.vconcat([row1, row2])
        cv2.imshow("Dashboard", dashboard)
        
        # Break on 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows