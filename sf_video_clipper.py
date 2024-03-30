import click
import cv2
import os
import numpy as np
import scipy.io

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
    #cv2.destroyAllWindows()


def subtract_similar(video_path, output_path, timestamp, duration, tolerance):
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
    total_frames = end_frame-start_frame+1
    
    # Store all frames to be processed in a list
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        print(f"Reading frame {len(frames)} / {total_frames}", end='\r')

    # For each frame, find similar frames and subtract average of similar frames
    processed_frames = []
    for idx, frame in enumerate(frames):
        print(f"Processing frame {idx+1} / {total_frames}", end='\r')
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
        print(f"Writing frame {idx+1} / {total_frames}", end='\r')

    cap.release()
    out.release()
    #cv2.destroyAllWindows()

def subtract_threshold(video_path, output_path, timestamp, duration, tolerance, threshold_value):
    """
    For all frames within a video find similar frames and subtract average of similar frames.
    After that is complete threshold the video.
    
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
    total_frames = end_frame-start_frame+1
    
    # Store all frames to be processed in a list
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        print(f"Reading frame {len(frames)} / {total_frames}", end='\r')

    # For each frame, find similar frames and subtract average of similar frames
    processed_frames = []
    for idx, frame in enumerate(frames):
        print(f"Processing frame {idx+1} / {total_frames}", end='\r')
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
    # Create the output folder if it doesn't exist
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    for frame in processed_frames:
        out.write(frame)
        print(f"Writing frame {idx+1} / {total_frames}", end='\r')

    cap.release()
    out.release()
    #cv2.destroyAllWindows()

def subtract_threshold_full_video(video_path, output_path, tolerance, threshold_value):
    """
    For all frames within a video find similar frames and subtract the average of similar frames
    This function was originally created for the manual labeling of the cells, but isn't needed for normal use.
    
    Args:
    - video_path (str): Path to the input video.
    - output_path (str): Path to save the processed video.
    - tolerance (float): Decimal representation (i.e., 0.1 = 10%) of what is to be considered a similar frame.
    - threshold_value (float): Value for thresholding images; above this value will be white, below black.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Total frames to process: {total_frames}")
    
    # Store all frames to be processed in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        print(f"Reading frame {len(frames)} / {total_frames}", end='\r')

    print("\nFinished reading frames. Starting processing...")

    # For each frame, find similar frames and subtract average of similar frames
    processed_frames = []
    for idx, frame in enumerate(frames):
        print(f"Processing frame {idx+1} / {total_frames}", end='\r')
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
        

    print("\nFinished processing frames. Saving video...")

    # Write the processed frames to the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    for idx, frame in enumerate(processed_frames):
        out.write(frame)
        print(f"Writing frame {idx+1} / {total_frames}", end='\r')

    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    print("\nVideo saved successfully.")

def clip_at_timestamp(video_path, output_path, timestamp, method, duration, tolerance, threshold_value):
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
        subtract_threshold(video_path, output_path, timestamp, duration, tolerance, threshold_value)
    elif method == "subtract_threshold_full_video":
        subtract_threshold_full_video(video_path, output_path, tolerance, threshold_value)
    else:
        # This is just no editing/processing to the video at all just clipping
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the start and end frames
        start_frame = int((timestamp - (duration / 2)) * fps)
        end_frame = int((timestamp + (duration / 2)) * fps)
        total_frames = end_frame-start_frame+1
        # Basic clipping without additional processing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            print(f"Writing frame {_} / {total_frames}", end='\r')
        out.release()
        cap.release()
        #cv2.destroyAllWindows()
        
def extract_frames(video_path, output_folder):
    """
    Function for extracting all of the frames from a video.
    Another function that was potentially for the manual labeling of data but isn't all that necessary anymore.

    Args:
    - video_path (str): Path to the input video.
    - output_path (str): Base Output folder that will contain the subfolder to save the extracted frames.
    """
    video_name = str(os.path.basename(video_path))
    video_name = "/" + video_name[:-4]
    output_path = output_folder + video_name
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly, save it
        if ret:
            frame_path = os.path.join(output_path, f"frame_{frame_count:05d}.jpeg")
            cv2.imwrite(frame_path, frame)
            image = cv2.imread(frame_path)
            height, width, channels = image.shape
            """
            This can definitely be modified and experimented width depending on the consistency of the videos.
            The issue with cropping at the moment is that some videos, even in the same set, are not consistent with where the sensor is so any specific measurement at the moment doesn't work. 
            """
            fraction_of_width = .2*width
            left_bound_width = int(fraction_of_width)
            right_bound_width = int(1-fraction_of_width)
            cropped_image = image[0:height, left_bound_width:right_bound_width]
            cv2.imwrite(frame_path, cropped_image)
            frame_count += 1
        else:
            break

    cap.release()
    print(f"Extracted {frame_count} frames to {output_path}")


@click.command()
@click.option('--input_directory', '-dd', required=True, help="Input directory that contains the 'set' subfolders and videos under the format 'video_log.mp4'")
@click.option('--output_directory', '-od', required=True, help="Directory that output will be saved to under the format of '{input_directory}/{output_directory}/Saved_Clips_{method}'")
@click.option('--method', '-m', default='none', help="Method that you want for the video to be processed with")
@click.option('--duration', '-d', default=2, help="Duration of the clips for the cells that will be looked at. Example, 1 second duration means get 0.5 seconds before timestamp and 0.5 seconds after timestamp.")
@click.option('--tolerance', '-t', default=0.1, help="Tolerance 'threshold' for what is to be considered a similar frame. Example, 10% similar frame will be 0.1 and mean that you consider frames with a mean that are within +/- 10% similar to one another.")
@click.option('--threshold_value', '-tv', default=30, help="Threshold value for thresholding images; above this value will be white, below black ")
@click.option('--extract_frames_flag', '-ef', is_flag=True, default=False, help="Save all of the frames of the outputted clips into a subfolder")
def process_videos(input_directory, output_directory, method, duration, tolerance, threshold_value, extract_frames_flag):
    """
    Main function for processing videos.
    Example Command:
    python sf_video_clipper.py -dd '1.5GHz_and_10.5GHz' -od 'Output/Saved_Clips' -m 'none'
    """

    base_input_directory = input_directory
    base_output_directory = os.path.join(base_input_directory, output_directory)
    input_directory = os.path.join(os.getcwd(), base_input_directory)
    output_directory = os.path.join(os.getcwd(), base_output_directory)
    # Load the 'sf.mat' file
    # There is most likely a much cleaner way to do this, I'm just drawing a blank on how
    mat_file_found = False
    for file in os.listdir(input_directory):
        if file.endswith(".mat"):
            mat_file_path = os.path.join(input_directory, file)
            data = scipy.io.loadmat(mat_file_path)
            mat_file_found = True
            break
    if mat_file_found == False:
        print(".mat file not found")
        exit

    final_cell_data = data['final_cell_data']
    output_directory = f"{output_directory}_{method}"

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over each element in 'final_cell_data'
    for idx in range(final_cell_data.shape[1]):
        cell_data = final_cell_data[0, idx]
        # Extracting 'set' number
        set_number = cell_data['set'][0, 0]
        # Extracting 'CellID'
        cell_id = cell_data['CellID'][0, 0]
        # Extracting 'time' value from 'f1'
        time_value = cell_data['f1']['time'][0, 0][0, 0]
        print(f"Set number: {set_number}, Cell Id: {cell_id}, Timestamp: {time_value}")
        video_path = os.path.join(input_directory, f"set{set_number}/video_log.mp4")
        if os.path.exists(video_path):
            output_filename = f"set{set_number}_cell{cell_id}.mp4"
            output_path = os.path.join(output_directory, output_filename)
            if os.path.exists(output_path):
                print(f"Already Completed: Folder {base_input_directory}, Set: {set_number}, Cell: {cell_id}. Saved at {output_path}.")
            else:
                # Clip the video at the specified timestamp
                clip_at_timestamp(video_path, output_path, time_value, method, duration, tolerance, threshold_value)
                print(f"Completed: Folder: {base_input_directory}, Set: {set_number}, Cell: {cell_id}. Saved at {output_path}.")
    
    # extract_frames is set up in this way so that the videos can fully process and get saved before attempting to get their frames.
    if (extract_frames_flag == True):
        for idx in range(final_cell_data.shape[1]):
            cell_data = final_cell_data[0, idx]
            set_number = cell_data['set'][0, 0]
            cell_id = cell_data['CellID'][0, 0]
                
            video_path = os.path.join(input_directory, f"set{set_number}/video_log.mp4")
            if os.path.exists(video_path):
                output_filename = f"set{set_number}_cell{cell_id}.mp4"
                output_path = os.path.join(output_directory, output_filename)
                extract_frames(output_path, output_directory)



if __name__ == "__main__":
    process_videos()
