import cv2
import numpy as np

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
      print("Error: Could not open camera.")
else:
        # Dummy function for the trackbar
        def nothing(x):
                pass

        # Window
        cv2.namedWindow("My Dashboard")

        # Blur Slider
        cv2.createTrackbar("Blur", "My Dashboard", 0, 10, nothing)
        # Sobel Slider
        cv2.createTrackbar("Sobel Size", "My Dashboard", 1, 3, nothing)
        # Canny Sliders (Aperture size should be odd between 3 and 7)
        cv2.createTrackbar("Canny Size", "My Dashboard", 0, 2, nothing)
        cv2.createTrackbar("Canny Lower Threshold", "My Dashboard", 50, 255, nothing)
        cv2.createTrackbar("Canny Upper Threshold", "My Dashboard", 100, 255, nothing)

        while True:
                ret, frame = cap.read()  # Capture each frame
                if not ret:
                        break

                # Shrink the frame to 320x240, to prevent big window once concatenated
                frame = cv2.resize(frame, (320, 240))

                # Get the trackbar value and make it odd
                blur_val = cv2.getTrackbarPos("Blur", "My Dashboard")
                # Ensure Odd Values
                k = (blur_val * 2) + 1 
                
                # Apply the blur
                frame = cv2.GaussianBlur(frame, (k, k), 0)

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
                hist_w, hist_h = 320, 240
                b_hist = cv2.normalize(b_hist, b_hist, 0, hist_h, cv2.NORM_MINMAX)
                g_hist = cv2.normalize(g_hist, g_hist, 0, hist_h, cv2.NORM_MINMAX)
                r_hist = cv2.normalize(r_hist, r_hist, 0, hist_h, cv2.NORM_MINMAX)
                gray_hist = cv2.normalize(gray_hist, gray_hist, 0, hist_h, cv2.NORM_MINMAX)
                h_hist = cv2.normalize(h_hist, h_hist, 0, hist_h, cv2.NORM_MINMAX)
                s_hist = cv2.normalize(s_hist, s_hist, 0, hist_h, cv2.NORM_MINMAX)
                v_hist = cv2.normalize(v_hist, v_hist, 0, hist_h, cv2.NORM_MINMAX)

                # Flatten histogram for plotting
                b_hist = b_hist.flatten()
                g_hist = g_hist.flatten()
                r_hist = r_hist.flatten()
                gray_hist = gray_hist.flatten()
                h_hist = h_hist.flatten()
                s_hist = s_hist.flatten()
                v_hist = v_hist.flatten()

                # Plot the histogram
                
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
                        
                # Sobel
                sobel_val = cv2.getTrackbarPos("Sobel Size", "My Dashboard")
                # Ensure Odd Values
                sobel_k = (sobel_val * 2) + 1
                # Sobel Edge Detection
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_k)  # Horizontal edges
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_k)  # Vertical edges
                sobel_combined = cv2.magnitude(sobel_x, sobel_y)
                sobel_combined = cv2.convertScaleAbs(sobel_combined)
                
                # Canny
                canny_val = cv2.getTrackbarPos("Canny Size", "My Dashboard")
                canny_lower = cv2.getTrackbarPos("Canny Lower Threshold", "My Dashboard")
                canny_upper = cv2.getTrackbarPos("Canny Upper Threshold", "My Dashboard")
                # Ensure Odd Values (Aperture size should be odd between 3 and 7)
                canny_k = (canny_val * 2) + 3
                canny_edges = cv2.Canny(gray, canny_lower, canny_upper, apertureSize=canny_k)

                # Laplacian of Gaussian (LoG)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian_abs = cv2.convertScaleAbs(laplacian)

                # Convert the 1-channel grayscale image and histogram to a 3-channel BGR image in order to use hconcat
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray_hist_rgb = cv2.cvtColor(gray_hist_img, cv2.COLOR_GRAY2BGR)
                sobel_bgr = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
                canny_bgr = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
                log_bgr = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)
                
                # Now concatenate them since they all have 3 channels!
                # Create a blank image to fill empty grid slots
                blank = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
                row1 = cv2.hconcat([frame, gray_rgb, hsv])
                row2 = cv2.hconcat([original_hist_img, gray_hist_rgb, hsv_hist_img])
                row3 = cv2.hconcat([sobel_bgr, canny_bgr, log_bgr])
                mosaic = cv2.vconcat([row1, row2, row3])
                cv2.imshow("Mosaic", mosaic)
                
                # Break on 'ESC' key
                if cv2.waitKey(1) & 0xFF == 27:
                        break

        cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close all windows