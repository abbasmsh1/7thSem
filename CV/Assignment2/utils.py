import os
import model

def is_image_or_video(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    # Get the file extension from the provided path
    _, file_extension = os.path.splitext(file_path)

    # Check if the file extension matches image or video extensions
    if file_extension.lower() in image_extensions:
        return "image"
    elif file_extension.lower() in video_extensions:
        return "video"
    else:
        return "unknown"  # Neither an image nor a video

