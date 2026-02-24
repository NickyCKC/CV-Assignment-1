import cv2
import numpy as np
import time

# Open the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
      print("Error: Could not open camera.")
else:
        # Dummy function for the trackbar
        def nothing(x):
                pass
        # Function for adding labels
        def add_label(img, text):
                if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img, (0, 0), (220, 35), (0, 0, 0), -1)
                cv2.putText(img, text, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return img      

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

        # Manual Sobel selector (0 -> 3x3, 1 -> 5x5)
        cv2.createTrackbar("Manual Sobel", "My Dashboard", 0, 1, nothing)

        # FPS timer
        prev_time = time.time()

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

                # Manual Sobel
                man_choice = cv2.getTrackbarPos("Manual Sobel", "My Dashboard")
                man_sobel_k = 3 if man_choice == 0 else 5

                if man_sobel_k == 3:
                        Kx = np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=np.float32)
                        Ky = np.array([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=np.float32)
                else:
                        Kx = np.array([[-1, -2, 0, 2, 1],
                                       [-4, -8, 0, 8, 4],
                                       [-6, -12, 0, 12, 6],
                                       [-4, -8, 0, 8, 4],
                                       [-1, -2, 0, 2, 1]], dtype=np.float32)
                        Ky = Kx.T

                man_gx = cv2.filter2D(gray, cv2.CV_64F, Kx)
                man_gy = cv2.filter2D(gray, cv2.CV_64F, Ky)
                man_out = cv2.convertScaleAbs(cv2.magnitude(man_gx, man_gy))

                # Convert the 1-channel grayscale image and histogram to a 3-channel BGR image in order to use hconcat
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray_hist_rgb = cv2.cvtColor(gray_hist_img, cv2.COLOR_GRAY2BGR)
                sobel_bgr = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)
                canny_bgr = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
                log_bgr = cv2.cvtColor(laplacian_abs, cv2.COLOR_GRAY2BGR)
                man_bgr = cv2.cvtColor(man_out, cv2.COLOR_GRAY2BGR)  
                
                # Now concatenate them since they all have 3 channels!
                # Create a blank image to fill empty grid slots
                blank = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
                #row1 = cv2.hconcat([frame, gray_rgb, hsv])
                #row2 = cv2.hconcat([original_hist_img, gray_hist_rgb, hsv_hist_img])
                #row3 = cv2.hconcat([sobel_bgr, canny_bgr, log_bgr])
                #mosaic = cv2.vconcat([row1, row2, row3])

                # Label each panel and concatenate
                row1 = cv2.hconcat([
                        add_label(frame.copy(), "Original (BGR)"),
                        add_label(gray_rgb.copy(), "Grayscale"),
                        add_label(hsv.copy(), "HSV")
                ])
                row2 = cv2.hconcat([
                        add_label(original_hist_img.copy(), "Hist BGR"),
                        add_label(gray_hist_rgb.copy(), "Hist Gray"),
                        add_label(hsv_hist_img.copy(), "Hist HSV")
                ])
                row3 = cv2.hconcat([
                        add_label(sobel_bgr.copy(), f"Sobel (k={sobel_k})"),
                        add_label(canny_bgr.copy(), f"Canny (k={canny_k})"),
                        add_label(log_bgr.copy(), "LoG")
                ])
                row4 = cv2.hconcat([
                        add_label(man_bgr.copy(), f"Manual Sobel ({man_sobel_k}x{man_sobel_k})"),
                        blank,
                        blank
                ])

                mosaic = cv2.vconcat([row1, row2, row3, row4])

                # FPS Counter
                curr_time = time.time()
                dt = curr_time - prev_time
                fps = 1 / dt if dt > 0 else 0
                prev_time = curr_time

                cv2.rectangle(mosaic, (mosaic.shape[1] - 150, 0), (mosaic.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(mosaic, f"FPS: {int(fps)}",
                            (mosaic.shape[1] - 140, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Display
                cv2.imshow("My Dashboard", mosaic)

                # Break on 'ESC' key
                if cv2.waitKey(1) & 0xFF == 27:
                        break

        cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close all windows