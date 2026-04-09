# Computer Vision: Video Processing and Edge Detection Dashboard

---

## Project Overview
This project implements a real-time video processing pipeline using Python and OpenCV. The application captures a live camera feed and displays a comprehensive visual dashboard (mosaic) that includes multiple color spaces, live histograms, and a comparison of various edge detection algorithms

## Features

### 1. Dashboard Layout
* **Unified Interface**: Uses `cv2.hconcat` and `cv2.vconcat` to compose a grid view within a single window
* **Color Space Conversions**: Displays the live feed in Original (BGR), Grayscale, and HSV formats
* **Real-time Performance**: Includes a live FPS counter to monitor the processing speed of the pipeline

### 2. Live Histograms
* **Dynamic Visuals**: Computes and displays histograms that update continuously during the stream
* **Multi-Channel Support**: For the BGR and HSV color spaces, the dashboard displays a three-channel histogram (Blue, Green, Red) in a single plot for comparative analysis

### 3. Filtering and Edge Detection
* **Gaussian Filtering**: Applies Gaussian blur to each color space image with adjustable kernel sizes
* **Edge Detectors**: Provides a side-by-side comparison of the following methods:
    * **Sobel**: Displays the gradient magnitude image
    * **Canny**: Standard edge detection with interactive thresholding
    * **Laplacian of Gaussian (LoG)**: Advanced edge map based on the filtered output
* **Manual Implementation**: Features a custom-coded Sobel operator using 3x3 kernels applied via `cv2.filter2D` to calculate gradient magnitude

## Interactive Parameter Control
The dashboard utilizes OpenCV trackbars to allow real-time tuning of the following parameters:
* **Blur Kernel Size**: Controls the intensity of the Gaussian filter (forced to odd values)
* **Sobel/Canny Kernel Size**: Adjusts the sensitivity of the edge detection operators
* **Canny Thresholds**: High and low thresholds can be adjusted to fine-tune edge sensitivity

## Requirements
* Python 3.x
* OpenCV (`opencv-python`)
* NumPy

## How to Run
1.  Connect a webcam to your computer.
2.  Install dependencies: `pip install opencv-python numpy`.
3.  Run the script: `python cv_assignment_1.py`.
4.  Use the trackbars at the top of the dashboard to adjust filters.
5.  Press **ESC** to exit the application.
