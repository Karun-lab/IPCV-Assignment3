# IPCV-Assignment3
This assignment involves video processing with the use of various computer vision techniques. The task is to process a video and apply several filters at different time intervals, including Gaussian blur, Sobel edge detection, Canny edge detection, and Fourier Transform-based filtering (low-pass, high-pass). The aim is to manipulate the video in real-time, overlay visual effects, and track objects like drones within the frames using template matching and optical flow.

Code Explanation
The code is divided into several key sections:

Loading Templates: Templates are used for tracking objects like drones within the video. The function load_template() loads these templates, and load_and_resize_template() further resizes them for the matching process. These are useful in the template-matching process later in the code.

Discrete Fourier Transform (DFT) Filtering: The function apply_dft() applies DFT to a video frame to convert it to the frequency domain. It creates masks to implement low-pass, high-pass, and band-pass filters. This is useful for removing or preserving specific frequency components in the frame.

Template Matching: The template_matching() function compares a grayscale frame to the loaded template using OpenCV's cv.matchTemplate(), detecting and highlighting matching regions with a rectangle. This is a basic method for detecting objects within video frames.

Optical Flow Tracking: optical_flow() uses the Lucas-Kanade method for tracking movement between consecutive frames. It detects key points in one frame, calculates how those points move in the next frame, and visualizes the motion using arrows.

Drone Tracking: The function drone_tracking() specifically tracks a drone's location using template matching and keeps a record of its movement trajectory, drawing lines to show the path it followed.

Video Processing Loop: The main loop reads the input video, processes each frame, and applies different filters and transformations based on the video's current time (in seconds):

0-2 seconds: No filter (original video).
2-4 seconds: Gaussian blur is applied to smooth the frame.
4-6 seconds: The image is sharpened (though sharpening is not explicitly defined).
6-8 seconds: Sobel operator is used for edge detection.
8-10.5 seconds: Canny edge detection highlights strong edges in the frame.
10-12 seconds: DFT magnitude spectrum is shown, illustrating the frequency components.
12-15 seconds: Low-pass filter is applied via DFT to remove high-frequency noise.
15-17 seconds: High-pass filter is applied via DFT to highlight sharp transitions.
Subtitle Addition: The add_subtitle() function overlays text on the video frames, indicating which processing step is being shown.

Optical Flow Initialization: Before entering the main loop, the first frame is captured and converted to grayscale, and feature points are detected to be tracked using optical flow in subsequent frames.

Video Output: The processed video is written frame by frame to a new video file (Output Video.mp4) using OpenCV’s VideoWriter.

How to Run the Code
Prerequisites:
Install OpenCV and NumPy: Make sure you have the necessary libraries installed. You can install them using:

bash
Copy code
pip install opencv-python numpy
Place Your Video and Templates:

Ensure you have a video file named Input Video.mp4 in the working directory.
Store the object templates (such as drones or logos) inside a folder named Templates in the same directory.
The script also loads two additional template images logo.png and me.png which must be in the same directory.
Running the Script:
Download & save: The Input video can be downloaded from drive link. Save it in the same folder as the code.   

Run the Script: You can run the script in the terminal or command prompt by typing:

bash
Copy code
python code.py
Output:

The script will read the video, process each frame with various effects, and write the processed output to a new video file named Output Video.mp4.
Depending on the duration of the input video, the script may take some time to process all frames.
Notes:
The provided script processes video frames sequentially and adds specific effects at different time intervals. Make sure the input video length is long enough to see the different effects applied.
If necessary, modify the video path and template paths to suit your file structure. The current setup expects specific templates and a video file in the working directory.
