import cv2
import sys
import os
import pandas as pd
import numpy as np

# if the files are all consistent with how they are recorded, I could make the crop values permanent
# everything should have the same resolution
def crop_video(video_path, output_path, start_x, start_y, crop_width, crop_height):
    input_video = cv2.VideoCapture(video_path)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
    
    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        cropped_frame = frame[start_y:start_y+crop_height, start_x:start_x+crop_width]
        mean_pixel_value = cropped_frame.mean(axis=(0, 1))
        cropped_frame -= mean_pixel_value.astype(int)
        output_video.write(cropped_frame)

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()


def subtract_similar(video_path, output_path, timestamp, duration=2.0, tolerance=0.1):
    """
    For all frames within a video find similar frames and subtract average of similar frames
    
    Args:
    - video_path (str): Path to the input video.
    - output_path (str): Path to save the clipped video.
    - timestamp (float): Timestamp at which to center the clip.
    - duration (float): Duration of the clip.
    - tolerance (float): decimal representation (i.e. 0.1 = 10%) of what is to be considered a similar frame
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the start and end frames
    start_frame = int((timestamp - (duration / 2)) * fps)
    end_frame = int((timestamp + (duration / 2)) * fps)
    
    # Store all frames to be processed in a list
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # For each frame, find similar frames and subtract average of similar frames
    processed_frames = []
    for idx, frame in enumerate(frames):
        mean_value = frame.mean()
        lower_bound = mean_value * (1 - tolerance)
        upper_bound = mean_value * (1 + tolerance)
        
        # Find frames that are within the tolerance range
        similar_frames = [f for f in frames if lower_bound <= f.mean() <= upper_bound]
        
        # Compute the average of similar frames
        avg_frame = np.mean(similar_frames, axis=0).astype(frame.dtype)
        
        # Subtract the average frame from the current frame
        processed_frame = cv2.absdiff(frame, avg_frame)
        processed_frames.append(processed_frame)

    # Write the processed frames to the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    for frame in processed_frames:
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def subtract_threshold(video_path, output_path, timestamp, duration=2.0, tolerance=0.1, threshold_value=30):
    """
    For all frames within a video find similar frames and subtract average of similar frames
    
    Args:
    - video_path (str): Path to the input video.
    - output_path (str): Path to save the clipped video.
    - timestamp (float): Timestamp at which to center the clip.
    - duration (float): Duration of the clip.
    - tolerance (float): decimal representation (i.e. 0.1 = 10%) of what is to be considered a similar frame
    - threshold_value (float): value for thresholding images; above this value will be white, below black 
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the start and end frames
    start_frame = int((timestamp - (duration / 2)) * fps)
    end_frame = int((timestamp + (duration / 2)) * fps)
    
    # Store all frames to be processed in a list
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # For each frame, find similar frames and subtract average of similar frames
    processed_frames = []
    for idx, frame in enumerate(frames):
        mean_value = frame.mean()
        lower_bound = mean_value * (1 - tolerance)
        upper_bound = mean_value * (1 + tolerance)
        
        # Find frames that are within the tolerance range
        similar_frames = [f for f in frames if lower_bound <= f.mean() <= upper_bound]
        
        # Compute the average of similar frames
        avg_frame = np.mean(similar_frames, axis=0).astype(frame.dtype)
        
        # Subtract the average frame from the current frame
        processed_frame = cv2.absdiff(frame, avg_frame)
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply the threshold
        _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR for video saving
        bgr_frame = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)
        
        processed_frames.append(bgr_frame)

    # Write the processed frames to the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    for frame in processed_frames:
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def clip_at_timestamp(video_path, output_path, timestamp, method='none', duration=2.0):
    """
    Clips a video at a specific timestamp and applies the selected processing method.
    
    Args:
    - video_path (str): Path to the input video.
    - output_path (str): Path to save the clipped video.
    - timestamp (float): Timestamp at which to center the clip.
    - duration (float): Duration of the clip.
    - method (str): Processing method ('subtract_similar', 'threshold_equalize', or 'none').
    """
    
    if method == 'subtract_similar':
        subtract_similar(video_path, output_path, timestamp, duration)
    elif method == 'subtract_threshold':
        subtract_threshold(video_path, output_path, timestamp, duration)
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the start and end frames
        start_frame = int((timestamp - (duration / 2)) * fps)
        end_frame = int((timestamp + (duration / 2)) * fps)
        # Basic clipping without additional processing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        cap.release()
        cv2.destroyAllWindows()

def process_videos_from_date_directory(date_directory, output_directory, method):
    xlsx_file = f"{date_directory} Dead Phase.xlsx"
    xlsx_path = os.path.join(date_directory, xlsx_file)
    
    df = pd.read_excel(xlsx_path, engine='openpyxl')
    output_directory = output_directory + "_" + method
    # Loop through the rows to clip videos
    for _, row in df.iterrows():
        test_number = row['Test Number']
        seconds = row['Seconds (x)']
        cell_number = row['Cell Number']
        
        video_path = os.path.join(date_directory, f"Dead Phase/Test {test_number}/video_log.avi")
        if os.path.exists(video_path):
            output_filename = f"{date_directory.replace(' ', '')}_test{test_number}_cell{cell_number}.mp4"
            output_path = os.path.join(output_directory, output_filename)
            
            # Clip the video at the specified timestamp
            clip_at_timestamp(video_path, output_path, seconds, method)
            print(f"Completed: Date: {date_directory}, Test: {test_number}, Cell: {cell_number}")


if __name__ == "__main__":
    process_videos_from_date_directory("July 13", "Saved_Clips", "none")
