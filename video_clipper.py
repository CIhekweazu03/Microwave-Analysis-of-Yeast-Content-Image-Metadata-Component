import cv2
import sys
import os

#if the files are all consistent with how they are recorded, I could make the crop values permanent
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
        output_video.write(cropped_frame)

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()


def clip_at_intervals(video_path, output_dir, interval):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure to start at the passed interval s mark and end at the passed interval s mark to avoid start/end issues
    start_frame = int(fps * interval) 
    end_frame = int(total_frames - fps * interval) 
    
    # Iterate through the frames, creating a clip every interval s because code is always getting half interval  before and after
    # In hindsight, could mess around with this (cutting interval in half) to ensure that there is some overlap, but I'm not sure
    for i in range(start_frame, end_frame, int(fps * interval)):
        output_path = os.path.join(output_dir, f"clip_{i//fps:.1f}.mp4")
        clip_video_from_frame(video_path, output_path, i - int(fps * interval/2), i + int(fps * interval/2))

    cap.release()
    cv2.destroyAllWindows()


def clip_video_from_frame(video_path, output_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(int(start_frame), int(min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_videos_in_directory(directory_path, output_base_dir):
    #This will probably end up being changed to try to match the Box setup instead over time
    for filename in os.listdir(directory_path):
        if filename.endswith(".avi"):
            video_path = os.path.join(directory_path, filename)
            output_dir = os.path.join(output_base_dir, f"{filename}_clips")
            os.makedirs(output_dir, exist_ok=True)
            sample_interval = 1 #1 as in seconds
            #1 second interval seems to work best, some of the cells do have higher velocities, but everything does seem to fit in the 1 second interval
            clip_at_intervals(video_path, output_dir, sample_interval)

if __name__ == "__main__":
    process_videos_in_directory("Sample_Video_Log", "Sample_Video_Log/Output")
