import cv2 as cv
import numpy as np
import os
import time
start_time = time.time()

def load_template(template_path):
    """Loads a template image in grayscale."""
    template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Template file could not be read: {template_path}")
    return template

def load_and_resize_template(template_path, scale=0.75):
    """Loads and resizes a template image."""
    template = load_template(template_path)
    # Resize the template to the specified scale
    resized_template = cv.resize(template, (0, 0), fx=scale, fy=scale)
    return resized_template

def apply_dft(frame, filter_type=None, d0=30, d1=60):
    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply DFT and shift zero frequency components to the center
    dft = cv.dft(np.float32(blurred), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate magnitude spectrum for visualization
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    # Create mask for filtering
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)

    # Define the filter type
    if filter_type == 'low_pass':
        mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 1
    elif filter_type == 'high_pass':
        mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 0
        mask[crow - d1:crow + d1, ccol - d1:ccol + d1] = 1
    elif filter_type == 'band_pass':
        mask[crow - d1:crow + d1, ccol - d1:ccol + d1] = 1
        mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 0

    # Apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)

    return np.uint8(img_back), magnitude_spectrum


def template_matching(frame, template):
    result = cv.matchTemplate(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), template, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    w, h = template.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), 4)
    return frame

def optical_flow(prev_gray, frame_gray, prev_pts):
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# Create a mask image for drawing the optical flow arrows
    # Check if there are points to track
    if prev_pts is None or len(prev_pts) == 0:
        print("No points to track.")
        return None, prev_pts
    # Calculate optical flow
    next_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
    # Check if the flow was successfully calculated
    if next_pts is None or status is None:
        print("Optical flow calculation failed.")
        return None, prev_pts
    # Filter good points
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    frame_color = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = int(new[0]), int(new[1])  # Convert coordinates to integers
        c, d = int(old[0]), int(old[1])  # Convert coordinates to integers
        frame_color = cv.line(frame_color, (a, b), (c, d), (0, 255, 0), 3)
        frame_color = cv.circle(frame_color, (a, b), 5, (0, 0, 255), -1)
    
    return frame_color, good_new.reshape(-1, 1, 2)

trajectory_points = []
def drone_tracking(frame, template):
    global trajectory_points
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    result = cv.matchTemplate(gray_frame, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(result)
    w, h = template.shape[::-1]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2
    center = (center_x, center_y)
    trajectory_points.append(center)
    cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), 5)
    # Draw the trajectory path
    for i in range(1, len(trajectory_points)):
        cv.line(frame, trajectory_points[i - 1], trajectory_points[i], (255, 0, 0), 2)
    return frame

# Load the video
cap = cv.VideoCapture('Input Video.mp4')

# Get video properties
fps = int(cap.get(cv.CAP_PROP_FPS))
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Load templates

# Load and resize all templates from the 'Karun's Assignment' folder
template_folder = 'Templates'
resized_templates = []

for filename in os.listdir(template_folder):
    template_path = os.path.join(template_folder, filename)
    resized_template = load_and_resize_template(template_path)
    resized_templates.append(resized_template)

logo_template = load_and_resize_template('logo.png')
bar_template = load_and_resize_template('me.png')
drone_template = load_and_resize_template('drone.png')

# Optical Flow setup
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')  
out = cv.VideoWriter('Output Video.mp4', fourcc, fps, (frame_width, frame_height))

def add_subtitle(frame, text, position=(50, 50), font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    """Add subtitle text to the frame."""
    return cv.putText(frame, text, position, font, font_scale, color, thickness, lineType=cv.LINE_AA)

# Main video processing loop
print("Starting process... ")
while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        print("End of Processing. Exiting ...")
        break

    current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    current_time = current_frame / fps
    frame_resized = cv.resize(frame, (frame_width, frame_height))
    print(f"Processing frame at {current_time:.2f} seconds")

    # Add subtitles based on current processing interval
    if current_time <= 2:  # 0-2 seconds: No filter
        subtitle = "Original Video (No filter)"
        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

    elif current_time <= 4:  # 3-5 seconds: Gaussian Blur
        subtitle = "Gaussian Blur"
        blurred_frame = cv.GaussianBlur(frame_resized, (11, 11), sigmaX=10, sigmaY=10)
        blurred_frame = add_subtitle(blurred_frame, subtitle)
        out.write(blurred_frame)

    elif current_time <= 6:  
        subtitle = "Sharpening the image "
        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

    elif current_time <= 8:  # 8-10 seconds: Sobel Operator
        subtitle = "Sobel Operator (6-8 seconds)"
        gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv.magnitude(sobelx, sobely)
        sobel_combined = cv.convertScaleAbs(sobel_combined)
        sobel_combined_bgr = cv.cvtColor(sobel_combined, cv.COLOR_GRAY2BGR)
        sobel_combined_bgr = add_subtitle(sobel_combined_bgr, subtitle)
        out.write(sobel_combined_bgr)

    elif current_time <= 10.5:  # 6-8 seconds: Canny Edge Detection
        subtitle = "Canny Edge Detection (8-10 seconds)"
        edges = cv.Canny(frame_resized, 100, 200)
        edges_colored = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # Convert edges to BGR for saving
        edges_colored = add_subtitle(edges_colored, subtitle)
        out.write(edges_colored)

    elif current_time <= 12:  # 10-12 seconds: DFT Spectrum
        subtitle = "DFT Spectrum (10-12 seconds)"
        _, magnitude_spectrum = apply_dft(frame_resized, filter_type=None)
        magnitude_spectrum_colored = cv.cvtColor(magnitude_spectrum, cv.COLOR_GRAY2BGR)
        magnitude_spectrum_colored = add_subtitle(magnitude_spectrum_colored, subtitle)
        out.write(magnitude_spectrum_colored)
    
    elif current_time <= 15:  # 10-14 seconds: Low-Pass FFT Filter
        subtitle = "Low-Pass FFT Filter (10-14 seconds)"
        filtered_frame, spectrum = apply_dft(frame_resized, filter_type='low_pass', d0=20)
        filtered_frame_bgr = cv.cvtColor(filtered_frame, cv.COLOR_GRAY2BGR)
        filtered_frame_bgr = add_subtitle(filtered_frame_bgr, subtitle)
        out.write(filtered_frame_bgr)

    elif current_time <= 17:  # 14-16 seconds: High-Pass FFT Filter
        subtitle = "High-Pass FFT Filter (14-16 seconds)"
        filtered_frame, spectrum = apply_dft(frame_resized, filter_type='high_pass', d0=40)
        filtered_frame_bgr = cv.cvtColor(filtered_frame, cv.COLOR_GRAY2BGR)
        filtered_frame_bgr = add_subtitle(filtered_frame_bgr, subtitle)
        out.write(filtered_frame_bgr)

    elif current_time <= 20:  # 16-20 seconds: Band-Pass FFT Filter
        subtitle = "Band-Pass FFT Filter (16-20 seconds)"
        filtered_frame, spectrum = apply_dft(frame_resized, filter_type='band_pass', d0=30, d1=60)
        filtered_frame_bgr = cv.cvtColor(filtered_frame, cv.COLOR_GRAY2BGR)
        filtered_frame_bgr = add_subtitle(filtered_frame_bgr, subtitle)
        out.write(filtered_frame_bgr)

    elif current_time <= 25:  # 21-25 seconds: Template Matching (Multiple Templates)
        subtitle = "Template Matching (21-25 seconds)"
        for template in resized_templates:
            frame_resized = template_matching(frame_resized, template)
        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

    elif current_time <= 30:  # 26-30 seconds: Template Matching (Logo)
        subtitle = "Template Matching: University Logo (26-30 seconds)"
        frame_resized = template_matching(frame_resized, logo_template)
        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

    elif current_time <= 35:  # 31-35 seconds: Template Matching (Bar)
        subtitle = "Template Matching: Karun's Photo (31-35 seconds)"
        frame_resized = template_matching(frame_resized, bar_template)
        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

    # Optical Flow tracking
    elif current_time <= 40.1:  # 36-40 seconds: Optical Flow
        subtitle = "Optical Flow (36-40 seconds)"
        frame_gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
        # If no points to track, reinitialize them
        if prev_pts is None or len(prev_pts) == 0:
            print("No points to track, reinitializing...")
            prev_pts = cv.goodFeaturesToTrack(frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Only attempt optical flow if points are available
        if prev_pts is not None and len(prev_pts) > 0:
            frame_resized, prev_pts = optical_flow(prev_gray, frame_gray, prev_pts)
            prev_gray = frame_gray.copy()

        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

    elif current_time <= 59:  # 41-59 seconds: Drone Tracking
        subtitle = "Drone Detection and Trajectory tracking (40-60 seconds)"
        frame_resized = drone_tracking(frame_resized, drone_template)
        frame_resized = add_subtitle(frame_resized, subtitle)
        out.write(frame_resized)

print("Total time taken: {:.2f} seconds".format(time.time() - start_time))

# Release everything
cap.release()
out.release()
