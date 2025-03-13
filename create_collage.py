import cv2
import os
import math
from PIL import Image, ImageOps

def extract_frames_from_video(video_path, output_folder, interval=100):
    """
    Extract frames from a video file at every `interval` frames.
    
    Parameters:
      video_path (str): Path to the video file.
      output_folder (str): Folder to save the extracted frames.
      interval (int): Save every nth frame.
    
    Returns:
      List of file paths for the extracted frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            try:
                # Convert frame from BGR to RGB and ensure it's in RGB mode
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb).convert("RGB")
                filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
                img.save(filename)
                frame_paths.append(filename)
                print(f"Saved frame {frame_count} to {filename}")
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
        frame_count += 1

    cap.release()
    return frame_paths

def create_collage_from_images(image_paths, output_path, frames_per_row=5, image_size=(200, 200)):
    """
    Create a collage from a list of image file paths.
    
    Parameters:
      image_paths (list): List of image file paths.
      output_path (str): File path to save the collage image.
      frames_per_row (int): Number of frames per row in the collage.
      image_size (tuple): Size (width, height) to which each image is resized.
    """
    # Open and resize images with error handling
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = ImageOps.fit(img, image_size, Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if not images:
        print("No images were successfully processed for the collage.")
        return

    num_images = len(images)
    num_cols = frames_per_row
    num_rows = math.ceil(num_images / num_cols)

    # Create a blank white canvas for the collage
    collage_width = num_cols * image_size[0]
    collage_height = num_rows * image_size[1]
    collage = Image.new("RGB", (collage_width, collage_height), color="white")

    # Paste images into the collage
    for idx, img in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        x = col * image_size[0]
        y = row * image_size[1]
        collage.paste(img, (x, y))

    try:
        collage.save(output_path)
        print(f"Collage saved to {output_path}")
    except Exception as e:
        print(f"Error saving collage: {e}")

if __name__ == '__main__':
    # Specify your video file path; use forward slashes or a raw string for Windows paths
    video_path = r"data/recurrent-ppo-tray.mp4"  # Adjust path if needed
    # Folder to temporarily save extracted frames
    frames_folder = 'extracted_frames'
    # File path to save the final collage
    collage_path = 'collage_robot_walk.png'
    
    # Extract frames from the video (e.g., every 100 frames)
    frame_files = extract_frames_from_video(video_path, frames_folder, interval=4)
    print(f"Extracted {len(frame_files)} frames.")
    
    # Create a collage from the extracted frames, with 5 frames per row
    create_collage_from_images(frame_files, collage_path, frames_per_row=5, image_size=(200, 200))
