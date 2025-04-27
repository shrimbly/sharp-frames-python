#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import json
import shutil
import tempfile
import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Set
# Simplify imports to avoid multiprocessing issues on Windows
from multiprocessing import cpu_count
import threading
import queue
import concurrent.futures # Add concurrent.futures import

# Add tqdm for progress visualization (mandatory dependency)
from tqdm import tqdm

# Define a custom exception for image processing errors
class ImageProcessingError(Exception):
    pass

# Helper functions for interactive mode
def get_valid_file_path(prompt: str, must_exist: bool = True) -> str:
    """Get a valid file path from user input."""
    while True:
        path = input(prompt).strip()
        
        # Handle empty input
        if not path:
            print("Please enter a valid path.")
            continue
            
        # Expand user directory if present (e.g., ~/videos)
        path = os.path.expanduser(path)
        
        # Check if file exists when required
        if must_exist and not os.path.isfile(path):
            print(f"Error: File '{path}' not found. Please enter a valid file path.")
            continue
            
        return path

def get_valid_dir_path(prompt: str, create_if_missing: bool = True, check_emptiness: bool = True) -> str:
    """Get a valid directory path from user input."""
    while True:
        path = input(prompt).strip()
        
        # Handle empty input
        if not path:
            print("Please enter a valid directory path.")
            continue
            
        # Expand user directory if present (e.g., ~/output)
        path = os.path.expanduser(path)
        
        # Check if directory exists
        if os.path.exists(path):
            if not os.path.isdir(path):
                print(f"Error: '{path}' exists but is not a directory. Please enter a directory path.")
                continue
                
            # Check if directory is empty only if requested
            if check_emptiness and os.listdir(path):
                overwrite = input(f"Directory '{path}' is not empty. Files may be overwritten. Continue? (y/n): ").strip().lower()
                if overwrite not in ['y', 'yes']:
                    continue
        elif create_if_missing:
            try:
                os.makedirs(path)
                print(f"Created directory: {path}")
            except Exception as e:
                print(f"Error creating directory '{path}': {str(e)}. Please enter a valid path.")
                continue
        else:
            print(f"Error: Directory '{path}' does not exist. Please enter a valid path.")
            continue
            
        return path

def get_valid_int(prompt: str, min_value: int = None, max_value: int = None, default: int = None) -> int:
    """Get a valid integer from user input."""
    # Add default to prompt if provided
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    
    while True:
        user_input = input(prompt).strip()
        
        # Use default value if input is empty
        if user_input == "" and default is not None:
            return default
        
        # Try to convert to integer
        try:
            value = int(user_input)
        except ValueError:
            print("Please enter a valid integer.")
            continue
            
        # Validate range if specified
        if min_value is not None and value < min_value:
            print(f"Please enter a value greater than or equal to {min_value}.")
            continue
            
        if max_value is not None and value > max_value:
            print(f"Please enter a value less than or equal to {max_value}.")
            continue
            
        return value

def get_choice(prompt: str, choices: List[str], default: str = None) -> str:
    """Get a choice from a list of options."""
    # Format choices for display
    choices_display = "/".join(choices)
    
    # Add default to prompt if provided
    if default is not None and default in choices:
        prompt = f"{prompt} ({choices_display}) [{default}]: "
    else:
        prompt = f"{prompt} ({choices_display}): "
    
    while True:
        user_input = input(prompt).strip().lower()
        
        # Use default value if input is empty
        if user_input == "" and default is not None:
            return default
        
        # Check if input is a valid full choice (case-insensitive)
        for choice in choices:
            if user_input == choice.lower():
                return choice

        # Check if input is 3 letters and matches the start of a choice (case-insensitive)
        if len(user_input) == 3:
            for choice in choices:
                if choice.lower().startswith(user_input):
                    return choice # Assume first 3 letters are unique enough

        # If no match found (full or 3-letter prefix)
        print(f"Please enter one of the following (or first 3 letters): {choices_display}")

def get_yes_no(prompt: str, default: bool = None) -> bool:
    """Get a yes/no response from the user."""
    # Add default to prompt if provided
    if default is not None:
        default_str = "y" if default else "n"
        prompt = f"{prompt} (y/n) [{default_str}]: "
    else:
        prompt = f"{prompt} (y/n): "
    
    while True:
        user_input = input(prompt).strip().lower()
        
        # Use default value if input is empty
        if user_input == "" and default is not None:
            return default
        
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'.")

class SharpFrames:
    def __init__(self,
                 # --- Core Parameters ---
                 input_path: str,
                 input_type: str,
                 output_dir: str,
                 fps: int = 10,
                 output_format: str = "jpg",
                 force_overwrite: bool = False,
                 selection_method: str = "best-n",

                 # --- Parameters for 'best-n' selection ---
                 num_frames: int = 300,
                 min_buffer: int = 3,

                 # --- Parameters for 'batched' selection ---
                 batch_size: int = 5,
                 batch_buffer: int = 0,

                 # --- Parameters for 'outlier-removal' selection ---
                 outlier_window_size: int = 15,
                 outlier_sensitivity: int = 50):

        # --- Constants ---
        # Filename format for output files
        self.OUTPUT_FILENAME_FORMAT = "output_{seq:05d}_idx_{idx:05d}_score_{score:.2f}.{ext}"
        # Supported image extensions for directory input
        self.SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
        # Weights for composite score calculation in 'best-n' selection
        self.BEST_N_SHARPNESS_WEIGHT = 0.7
        self.BEST_N_DISTRIBUTION_WEIGHT = 0.3
        # Parameters for 'outlier-removal' calculation
        self.OUTLIER_MIN_NEIGHBORS = 3
        self.OUTLIER_THRESHOLD_DIVISOR = 4

        # --- Instance Variables ---
        self.input_path = input_path
        self.input_type = input_type
        self.output_dir = output_dir
        self.fps = fps
        self.num_frames = num_frames
        self.min_buffer = min_buffer
        self.output_format = output_format
        self.temp_dir = None
        self.frames = []
        self.force_overwrite = force_overwrite
        
        # New properties for selection methods
        self.selection_method = selection_method
        self.batch_size = batch_size
        self.batch_buffer = batch_buffer
        self.outlier_window_size = outlier_window_size
        self.outlier_sensitivity = outlier_sensitivity
        
    def _setup(self) -> bool:
        """Perform initial setup checks and directory creation."""
        print(f"Processing {self.input_type}: {self.input_path}")
        # Check common dependencies (OpenCV) first
        if not self._check_dependencies(check_ffmpeg=False):
            return False

        os.makedirs(self.output_dir, exist_ok=True)
        try:
            self._check_output_dir_overwrite()
        except SystemExit: # Catch SystemExit raised by _check_output_dir_overwrite
             return False # Indicate setup failure

        # Check video-specific dependencies if needed
        if self.input_type == "video":
            if not self._check_dependencies(check_ffmpeg=True):
                 return False # Check_dependencies already prints error
        return True

    def _load_input_frames(self) -> Tuple[List[str], bool]:
        """Load frame paths from video or directory. Returns (frame_paths, cleanup_temp_dir)."""
        frame_paths = []
        cleanup_temp_dir = False

        if self.input_type == "video":
            # Create temporary directory for extracted frames
            self.temp_dir = tempfile.mkdtemp()
            cleanup_temp_dir = True # Ensure cleanup only if temp dir was created
            print(f"Created temporary directory: {self.temp_dir}")

            print("Extracting video information...")
            video_info = self._get_video_info()
            duration = self._extract_duration(video_info)
            if duration:
                print(f"Video duration: {self._format_duration(duration)}")

            print(f"Extracting frames at {self.fps} fps...")
            # Extract frames relies on self.temp_dir being set
            self._extract_frames(duration)

            # Get paths from temp directory
            frame_paths = self._get_frame_paths()
            frame_count = len(frame_paths)
            print(f"Extracted {frame_count} frames")

        elif self.input_type == "directory":
            # Scan input directory for images
            frame_paths = self._get_image_paths_from_dir()
            frame_count = len(frame_paths)
            if frame_count == 0:
                 print("No images found to process.")
                 # Return empty list, indicating nothing to process further
                 return [], False

        else:
             # Should not happen if validation in main/interactive is correct
             raise ValueError(f"Invalid input_type: {self.input_type}")

        return frame_paths, cleanup_temp_dir

    def _analyze_and_select_frames(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """Calculate sharpness, select frames based on method, and return selected data."""
        print("Calculating sharpness scores...")
        frames_with_scores = self._calculate_sharpness(frame_paths)

        if not frames_with_scores:
            print("No frames/images could be scored.")
            return [] # Return empty list

        print(f"Selecting frames/images using {self.selection_method} method...")
        selected_frames_data = [] # Initialize
        if self.selection_method == "best-n":
            selected_frames_data = self._select_best_frames(frames_with_scores)
        elif self.selection_method == "batched":
            selected_frames_data = self._select_batched_frames(frames_with_scores)
        elif self.selection_method == "outlier-removal":
            # Outlier removal returns all frames with a 'selected' flag
            all_frames_data = self._select_outlier_removal_frames(frames_with_scores)
            # Filter here based on the 'selected' flag
            selected_frames_data = [frame for frame in all_frames_data if frame.get("selected", True)]
        else:
            print(f"Warning: Unknown selection method '{self.selection_method}'. Using best-n instead.")
            selected_frames_data = self._select_best_frames(frames_with_scores)

        if not selected_frames_data:
             print("No frames/images were selected based on the criteria.")
             # Return empty list

        return selected_frames_data

    def run(self):
        """Execute the full pipeline for either video or directory input."""
        cleanup_temp_dir = False # Flag to control temp dir cleanup

        try:
            # --- Setup Phase ---
            if not self._setup():
                 print("Setup failed. Exiting.")
                 return False

            # --- Load Input Frames Phase ---
            frame_paths, cleanup_temp_dir = self._load_input_frames()
            if not frame_paths and self.input_type == "directory":
                 print("No images found or loaded. Exiting gracefully.")
                 return True # Not an error, just nothing to process
            elif not frame_paths and self.input_type == "video":
                 print("No frames extracted from video. Exiting.")
                 # This might indicate an issue with extraction or an empty video
                 return False # Consider this potentially an error state

            # --- Analyze and Select Phase ---
            selected_frames_data = self._analyze_and_select_frames(frame_paths)
            if not selected_frames_data:
                 print("Frame analysis or selection yielded no results. Exiting.")
                 return True # Not necessarily an error, could be criteria didn't match

            # --- Save Phase ---
            print(f"Saving {len(selected_frames_data)} selected frames/images...")
            with tqdm(total=len(selected_frames_data), desc="Saving selected items") as progress_bar:
                self._save_frames(selected_frames_data, progress_bar)

            print(f"Successfully processed. Selected items saved to: {self.output_dir}")
            return True

        except KeyboardInterrupt:
            print("\nProcess cancelled by user. Cleaning up...")
            return False
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up temporary directory only if it was created (video input)
            if cleanup_temp_dir and self.temp_dir and os.path.exists(self.temp_dir):
                print("Cleaning up temporary directory...")
                try:
                    shutil.rmtree(self.temp_dir)
                    print(f"Cleaned up temporary directory: {self.temp_dir}")
                except Exception as e:
                    print(f"Warning: Could not clean up temporary directory: {str(e)}")

    def _check_output_dir_overwrite(self):
        """Checks output directory and handles overwrite confirmation."""
        if not os.path.isdir(self.output_dir):
            # If it doesn't exist, it will be created, no overwrite check needed
            return

        existing_files = os.listdir(self.output_dir)
        if existing_files and not self.force_overwrite:
            print(f"Warning: Output directory '{self.output_dir}' already contains {len(existing_files)} files.")
            print("This may cause existing files to be overwritten.")
            while True:
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    print("Continuing with existing output directory...")
                    break
                elif response in ['n', 'no']:
                    print("Operation cancelled. Please specify a different output directory or use --force-overwrite.")
                    raise SystemExit(1) # Use SystemExit for controlled exit
                else:
                    print("Please enter 'y' or 'n'.")
        elif existing_files and self.force_overwrite:
            print(f"Output directory '{self.output_dir}' contains {len(existing_files)} files. Overwriting without confirmation (--force-overwrite).")

    def _check_dependencies(self, check_ffmpeg: bool = True) -> bool:
        """Check if required dependencies are installed"""
        # Determine number of checks for progress bar
        num_checks = 3 if check_ffmpeg else 1

        try:
            with tqdm(total=num_checks, desc="Checking dependencies") as progress_bar:
                if check_ffmpeg:
                    # Check for FFmpeg
                    try:
                        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        progress_bar.update(1)
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print("Error: FFmpeg is not installed or not in PATH. Required for video input.")
                        return False

                    # Check for FFprobe
                    try:
                        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        progress_bar.update(1)
                    except (subprocess.SubprocessError, FileNotFoundError):
                        # This is only a warning as duration extraction is a nice-to-have
                        print("Warning: FFprobe is not installed or not in PATH. Video duration cannot be determined.")

                # Always check for OpenCV (needed for sharpness calculation)
                # A simple check if cv2 was imported successfully is enough here
                if 'cv2' not in sys.modules:
                     print("Error: OpenCV (cv2) is not installed. Please install it (e.g., pip install opencv-python).")
                     return False
                progress_bar.update(1) # Update progress for OpenCV check

        except Exception as e:
            print(f"Error checking dependencies: {str(e)}")
            return False
            
        return True
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    
    def _extract_duration(self, video_info: Dict[str, Any]) -> float:
        """Extract duration from video info"""
        try:
            if 'format' in video_info and 'duration' in video_info['format']:
                return float(video_info['format']['duration'])
            elif 'streams' in video_info:
                for stream in video_info['streams']:
                    if 'duration' in stream:
                        return float(stream['duration'])
        except (KeyError, ValueError, TypeError) as e:
            print(f"Warning: Unable to extract duration: {str(e)}")
        return None
    
    def _get_video_info(self) -> Dict[str, Any]:
        """Get video metadata using FFmpeg"""
        # Try using ffprobe for more detailed info
        probe_command = [
            "ffprobe", 
            "-v", "error",
            "-show_entries", "format=duration",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,duration",
            "-of", "json",
            self.input_path
        ]
        
        try:
            probe_result = subprocess.run(
                probe_command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=True,
                text=True
            )
            
            video_info = json.loads(probe_result.stdout)
            return video_info
        except subprocess.CalledProcessError:
            # Fallback if ffprobe fails
            return {"error": "Failed to get video info"}
    
    def _extract_frames(self, duration: float = None) -> bool:
        """Extract frames from video using FFmpeg"""
        output_pattern = os.path.join(self.temp_dir, f"frame_%05d.{self.output_format}")
        
        # Set a timeout threshold for the process in case it hangs
        process_timeout_seconds = 3600 # 1 hour timeout for FFmpeg process
        
        command = [
            "ffmpeg",
            "-i", self.input_path,
            "-vf", f"fps={self.fps}",
            "-q:v", "1",  # Highest quality
            "-threads", str(cpu_count()),
            "-hide_banner", # Hide verbose info
            "-loglevel", "warning", # Show errors and warnings
            output_pattern
        ]
        
        # Estimate total frames if duration is available
        estimated_total_frames = None
        if duration:
            estimated_total_frames = int(duration * self.fps)
            print(f"Estimated frames to extract: {estimated_total_frames}")
        else:
            # If no duration, we can't estimate total, so progress will be indeterminate
            print("Video duration not found, cannot estimate total frames.")

        # Setup progress monitoring
        progress_desc = "Extracting frames"
        # Use total=estimated_total_frames if available, else indeterminate
        progress_bar = tqdm(total=estimated_total_frames, desc=progress_desc, unit="frame")
        
        process = None
        stderr_output = ""
        try:
            # Start the FFmpeg process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE, # Pipe stdout to avoid printing to console
                stderr=subprocess.PIPE, # Capture stderr for error reporting
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            last_file_count = 0
            start_time = time.time()

            # Monitor process completion and update progress based on file count
            while process.poll() is None:
                # Check file count periodically
                try:
                    frame_files = os.listdir(self.temp_dir)
                    file_count = len(frame_files)

                    if file_count > last_file_count:
                        update_amount = file_count - last_file_count
                        progress_bar.update(update_amount)
                        last_file_count = file_count
                        # Update description if we have an estimate
                        if estimated_total_frames:
                             progress_bar.set_description(f"{progress_desc}: {file_count}/{estimated_total_frames}")
                        else:
                            progress_bar.set_description(f"{progress_desc}: {file_count} frames")


                    # Check for process timeout
                    if time.time() - start_time > process_timeout_seconds:
                        raise subprocess.TimeoutExpired(command, process_timeout_seconds)

                except FileNotFoundError:
                     # Temp dir might not exist yet briefly at the start
                     pass
                except Exception as e:
                    print(f"\nError during progress monitoring: {str(e)}")
                    # Continue monitoring the process itself

                # Small sleep to prevent high CPU usage and allow interrupts
                try:
                    time.sleep(0.5) # Check every half second
                except KeyboardInterrupt:
                    print("\nKeyboard interrupt received. Terminating FFmpeg...")
                    if process:
                        process.terminate()
                    progress_bar.close()
                    raise

            # Process finished, capture remaining stderr and check return code
            try:
                stdout_output, stderr_output = process.communicate(timeout=15) # Short timeout for final communication
            except subprocess.TimeoutExpired:
                print("\nFFmpeg timed out during final communication. Killing process.")
                process.kill()
                stdout_output, stderr_output = process.communicate() # Try one last time

            return_code = process.returncode
            
            # Update progress bar to completion or final count
            final_frame_count = len(os.listdir(self.temp_dir))
            if estimated_total_frames:
                 progress_bar.n = final_frame_count # Set final count precisely
                 progress_bar.total = final_frame_count # Adjust total if estimate was wrong
            else:
                 # If indeterminate, just update description
                 progress_bar.set_description(f"{progress_desc}: {final_frame_count} frames")

            progress_bar.close()

            print(f"Extraction complete: {final_frame_count} frames extracted")

            # Check result
            if return_code != 0:
                error_message = f"FFmpeg failed with exit code {return_code}."
                if stderr_output:
                    error_message += f"\nFFmpeg stderr:\n{stderr_output.strip()}"
                raise Exception(error_message)

            return True

        except KeyboardInterrupt:
            # Already handled termination in the loop
            print("\nCancelled by user during frame extraction. Cleaning up...")
            # Ensure process is terminated if loop was exited prematurely
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise # Re-raise KeyboardInterrupt for outer handler
        except subprocess.TimeoutExpired:
             print(f"\nFFmpeg process timed out after {process_timeout_seconds} seconds. Terminating.")
             if process and process.poll() is None:
                 process.terminate()
                 try:
                    process.wait(timeout=5)
                 except subprocess.TimeoutExpired:
                    process.kill()
             raise Exception("FFmpeg process timed out.")
        except Exception as e:
            print(f"\nError during frame extraction: {str(e)}")
            if progress_bar and not progress_bar.disable: # Check if progress bar was initialized
                progress_bar.close()
            if process and process.poll() is None:
                process.terminate()
            # Include stderr in exception if available
            if stderr_output:
                 e = Exception(f"{str(e)}\nFFmpeg stderr:\n{stderr_output.strip()}")
            raise e # Re-raise the exception
    
    def _get_image_paths_from_dir(self) -> List[str]:
        """Scan input directory, find, sort, and return image paths."""
        image_paths = []
        # Use defined constant for supported extensions
        supported_extensions_str = ', '.join(self.SUPPORTED_IMAGE_EXTENSIONS)
        print(f"Scanning directory {self.input_path} for images ({supported_extensions_str})...")

        try:
            for entry in os.scandir(self.input_path):
                if entry.is_file():
                    _, ext = os.path.splitext(entry.name)
                    if ext.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                        image_paths.append(entry.path)
        except FileNotFoundError:
            print(f"Error: Input directory not found: {self.input_path}")
            raise
        except Exception as e:
            print(f"Error scanning directory {self.input_path}: {str(e)}")
            raise

        # Sort paths alphabetically for consistent ordering
        image_paths.sort()

        if not image_paths:
            print(f"Warning: No supported image files ({supported_extensions_str}) found in {self.input_path}.")

        print(f"Found {len(image_paths)} images.")
        return image_paths
    
    def _get_frame_paths(self) -> List[str]:
        """Get list of all extracted frame paths (from temp dir)."""
        frame_files = os.listdir(self.temp_dir)
        # Sort frames by number to maintain sequence
        frame_files.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
        return [os.path.join(self.temp_dir, f) for f in frame_files]
    
    def _calculate_sharpness(self, frame_paths: List[str]) -> List[Dict[str, Any]]:
        """Calculate sharpness scores for all frames/images using parallel processing."""
        frames_data = []
        desc = "Calculating sharpness for frames" if self.input_type == "video" else "Calculating sharpness for images"

        # Use ThreadPoolExecutor for parallel processing
        # Adjust max_workers based on your system's capabilities, cpu_count() is a reasonable default
        num_workers = min(cpu_count(), len(frame_paths)) if len(frame_paths) > 0 else 1
        
        with tqdm(total=len(frame_paths), desc=desc) as progress_bar:
            futures = {}
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit tasks: store future mapped to its original index and path
                    for idx, path in enumerate(frame_paths):
                        future = executor.submit(self._process_image, path)
                        futures[future] = {"index": idx, "path": path}

                    # Process completed futures
                    for future in concurrent.futures.as_completed(futures):
                        task_info = futures[future]
                        path = task_info["path"]
                        idx = task_info["index"]
                        frame_id = os.path.basename(path)

                        try:
                            score = future.result() # Get score or raise exception if task failed
                            frame_data = {
                                "id": frame_id,
                                "path": path,
                                "index": idx,
                                "sharpnessScore": score
                            }
                            frames_data.append(frame_data)
                        except ImageProcessingError as e:
                            # Log specific image processing errors and continue
                            print(f"\nWarning: {str(e)}")
                        except Exception as e:
                            # Log unexpected errors during future processing
                            print(f"\nError retrieving result for {path}: {str(e)}")
                            # Optionally re-raise if it's critical, or just log and skip frame

                        progress_bar.update(1) # Update progress as each task finishes

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received during sharpness calculation.")
                print("Attempting to cancel pending tasks and save partial results...")
                # Executor shutdown (implicit in 'with' block) will attempt to wait,
                # but KeyboardInterrupt should expedite this.
                # Results gathered so far in frames_data will be kept.
                pass # Let finally block handle sorting
            except Exception as e:
                 # Catch broader exceptions during executor setup/management
                 print(f"\nUnexpected error during parallel sharpness calculation: {str(e)}")
                 # Depending on the error, might want to re-raise or exit
            finally:
                 # Ensure sorting happens even if interrupted or errors occurred
                 frames_data.sort(key=lambda x: x["index"])
                 
        return frames_data

    @staticmethod
    def _process_image(path: str) -> float:
        """Process a single image and return its sharpness score"""
        try:
            img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                # Raise custom exception if image reading fails
                raise ImageProcessingError(f"Failed to read image: {path}")

            height, width = img_gray.shape
            # Use INTER_AREA for downscaling - generally preferred
            img_half = cv2.resize(img_gray, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

            # Calculate Laplacian variance
            score = float(cv2.Laplacian(img_half, cv2.CV_64F).var())

            return score
        except cv2.error as e:
            # Wrap OpenCV errors
            raise ImageProcessingError(f"OpenCV error processing {path}: {str(e)}") from e
        except Exception as e:
            # Wrap other potential errors
            raise ImageProcessingError(f"Error processing {path}: {str(e)}") from e

    def _calculate_distribution_score(self, frame_index: int, total_frames: int,
                                      selected_indices: Set[int], min_gap: int) -> float:
        """Calculate a score representing how well a frame is distributed among others.

        This score aims to balance two factors:
        1. Distance Score: Encourages selecting frames far from already selected ones,
           promoting spacing based on the minimum buffer (`min_gap`). A frame exactly
           `min_gap` away from the nearest selected frame gets a score of 1, decreasing
           linearly to 0 as it gets closer.
        2. Position Score: Encourages selecting frames closer to 'ideal' positions if
           frames were perfectly evenly distributed across the timeline. A frame at an
           ideal position gets 1, decreasing to 0 at the midpoint between ideal positions.

        These are combined with weighting to favor distance slightly.
        """
        # Calculate distance from nearest selected frame (capped by min_gap)
        nearest_selected_distance = min_gap
        for selected_index in selected_indices:
            distance = abs(frame_index - selected_index)
            if distance < nearest_selected_distance:
                nearest_selected_distance = distance
        
        distance_score = min(nearest_selected_distance / min_gap, 1)
        
        segment_size = total_frames / max(len(selected_indices), 1)
        nearest_ideal_position = round(frame_index / segment_size) * segment_size
        position_score = 1 - (abs(frame_index - nearest_ideal_position) / segment_size)
        
        return (distance_score * self.BEST_N_DISTRIBUTION_WEIGHT) + (position_score * self.BEST_N_SHARPNESS_WEIGHT)

    def _is_gap_sufficient(self, frame_index: int, selected_indices: Set[int], min_gap: int) -> bool:
        """Check if a frame index maintains the minimum gap with selected indices."""
        return all(abs(frame_index - selected_index) >= min_gap for selected_index in selected_indices)

    def _select_best_frames_pass1(self, frames: List[Dict[str, Any]], n: int, min_gap: int,
                                  progress_bar: tqdm) -> Tuple[List[Dict[str, Any]], Set[int]]:
        """First pass of best-n: Select best frame from initial segments."""
        selected_frames = []
        selected_indices = set()
        segment_size = max(1, len(frames) // n)
        segments = [frames[i:i+segment_size] for i in range(0, len(frames), segment_size)]

        for segment in segments:
            if len(selected_frames) >= n:
                break

            # Find valid frames in the segment respecting the minimum gap
            valid_frames = [
                frame for frame in segment
                if self._is_gap_sufficient(frame["index"], selected_indices, min_gap)
            ]

            if valid_frames:
                best_frame = max(valid_frames, key=lambda f: f.get("sharpnessScore", 0))
                selected_frames.append(best_frame)
                selected_indices.add(best_frame["index"])
                progress_bar.update(1)

        return selected_frames, selected_indices

    def _select_best_frames_pass2(self, frames: List[Dict[str, Any]], n: int, min_gap: int,
                                  selected_frames: List[Dict[str, Any]], selected_indices: Set[int],
                                  progress_bar: tqdm):
        """Second pass of best-n: Fill remaining slots using composite score."""
        while len(selected_frames) < n:
            best_candidate = None
            best_composite_score = -1

            for frame in frames:
                frame_index = frame["index"]
                if frame_index in selected_indices:
                    continue

                # Check gap before calculating scores
                if not self._is_gap_sufficient(frame_index, selected_indices, min_gap):
                    continue

                # Calculate composite score only for valid candidates
                distribution_score = self._calculate_distribution_score(
                    frame_index, len(frames), selected_indices, min_gap
                )
                sharpness_score = frame.get("sharpnessScore", 0)
                # Use defined constants for weights
                composite_score = (
                    (sharpness_score * self.BEST_N_SHARPNESS_WEIGHT) +
                    (distribution_score * self.BEST_N_DISTRIBUTION_WEIGHT)
                )

                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_candidate = frame

            if best_candidate:
                selected_frames.append(best_candidate)
                selected_indices.add(best_candidate["index"])
                progress_bar.update(1)
            else:
                break # No more valid frames to select

    def _select_best_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the best N frames based on sharpness and distribution.

        Uses a two-pass algorithm:
        1. Pass 1: Selects the sharpest frame from distinct segments of the timeline
           to ensure initial broad coverage, respecting the minimum gap.
        2. Pass 2: Fills remaining slots by selecting frames with the best composite
           score (sharpness + distribution score), prioritizing sharpness but using
           distribution to encourage better spacing, always respecting the minimum gap.
        """
        if not frames:
            return []

        n = min(self.num_frames, len(frames))
        min_gap = self.min_buffer

        with tqdm(total=n, desc="Selecting frames") as progress_bar:
            # First pass: Select best from segments
            selected_frames, selected_indices = self._select_best_frames_pass1(
                frames, n, min_gap, progress_bar
            )

            # Second pass: Fill remaining slots if needed
            if len(selected_frames) < n:
                self._select_best_frames_pass2(
                    frames, n, min_gap, selected_frames, selected_indices, progress_bar
                )

        return sorted(selected_frames, key=lambda f: f["index"])

    def _select_batched_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select frames using batch selection method"""
        if not frames:
            return []
        
        selected_frames = []
        
        with tqdm(total=len(frames)//(self.batch_size + self.batch_buffer) + 1, 
                 desc="Selecting batches") as progress_bar:
            i = 0
            while i < len(frames):
                batch = frames[i:i + self.batch_size]
                if not batch:
                    break
                
                best_frame = max(batch, key=lambda f: f.get("sharpnessScore", 0))
                selected_frames.append(best_frame)
                
                # Skip frames for the next batch start index
                i += self.batch_size + self.batch_buffer
                progress_bar.update(1)
        
        print(f"Batch selection: Selected {len(selected_frames)} frames")
        return selected_frames

    def _is_frame_outlier(self, index: int, frames: List[Dict[str, Any]],
                          global_range: float, inverted_sensitivity: float) -> bool:
        """Determine if a frame at a given index is an outlier based on its neighbors."""
        # Use defined constant for minimum neighbors
        min_neighbors_for_comparison = self.OUTLIER_MIN_NEIGHBORS
        half_window = self.outlier_window_size // 2
        window_start = max(0, index - half_window)
        window_end = min(len(frames) - 1, index + half_window)

        # Exclude the frame itself from the window
        window_frames = frames[window_start:index] + frames[index+1:window_end+1]

        if len(window_frames) < min_neighbors_for_comparison:
            return False # Not enough neighbors to make a reliable comparison

        window_scores = [frame.get("sharpnessScore", 0) for frame in window_frames]
        window_avg = sum(window_scores) / len(window_frames)
        current_score = frames[index].get("sharpnessScore", 0)

        # Avoid division by zero if global_range is 0 (already checked in caller, but safe)
        if global_range == 0:
            return False

        absolute_diff = window_avg - current_score
        percent_of_range = (absolute_diff / global_range) * 100

        # Check for downward trend heuristic
        in_downward_trend = False
        if index > 0 and index < len(frames) - 1:
            prev_score = frames[index-1].get("sharpnessScore", 0)
            next_score = frames[index+1].get("sharpnessScore", 0)
            # Heuristic: If scores are naturally trending downwards, a frame might be significantly lower
            # than the average of its window (which includes higher preceding scores)
            # but still be part of a valid sequence. We avoid marking such frames
            # as outliers if they fit this decreasing pattern (prev > current > next).
            if prev_score > current_score > next_score:
                in_downward_trend = True

        # Use defined constant for threshold calculation
        threshold = inverted_sensitivity / self.OUTLIER_THRESHOLD_DIVISOR

        # Outlier condition: lower than average, exceeds threshold percentage, and not part of a gentle downward trend
        return (current_score < window_avg and
                percent_of_range > threshold and
                not in_downward_trend)

    def _select_outlier_removal_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select frames using outlier removal method by checking each frame."""
        if not frames:
            return []

        # Create a mutable list of frame data including the 'selected' flag
        result_frames = [frame.copy() for frame in frames]
        for frame_data in result_frames:
            frame_data["selected"] = True # Start with all frames selected

        # Pre-calculate global score range
        all_scores = [frame.get("sharpnessScore", 0) for frame in frames]
        global_min = min(all_scores)
        global_max = max(all_scores)
        global_range = global_max - global_min

        # If all scores are the same, or sensitivity is 0, return all frames
        if global_range == 0 or self.outlier_sensitivity == 0:
             if self.outlier_sensitivity == 0:
                 print("Outlier sensitivity is 0, skipping analysis.")
             else:
                 print("All frames have identical scores, skipping outlier analysis.")
             # We already prepared result_frames with selected=True
             return result_frames

        inverted_sensitivity = 100 - self.outlier_sensitivity

        with tqdm(total=len(frames), desc="Analyzing for outliers") as progress_bar:
            for i in range(len(frames)):
                # Check if the current frame is an outlier
                if self._is_frame_outlier(i, frames, global_range, inverted_sensitivity):
                    result_frames[i]["selected"] = False
                progress_bar.update(1)

        # Filter based on the final 'selected' status (can be done here or in the caller)
        # Keeping the filtering logic here for clarity of this method's purpose.
        selected_count = sum(1 for frame in result_frames if frame["selected"])
        print(f"Outlier removal: Marked {len(frames) - selected_count} outliers. Keeping {selected_count} frames.")

        # Return the list containing *all* original frames, each with its 'selected' status
        # The caller (_analyze_and_select_frames) will filter based on this status.
        return result_frames

    def _save_frames(self, selected_frames: List[Dict[str, Any]], progress_bar=None) -> None:
        """Save selected frames/images to output directory."""
        metadata_list = []
        for i, frame_data in enumerate(selected_frames):
            src_path = frame_data["path"]
            original_id = frame_data["id"]
            original_index = frame_data["index"]
            sharpness_score = frame_data["sharpnessScore"]

            # Use the defined constant format string
            filename = self.OUTPUT_FILENAME_FORMAT.format(
                seq=i+1,
                idx=original_index,
                score=sharpness_score,
                ext=self.output_format
            )
            dst_path = os.path.join(self.output_dir, filename)

            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                 print(f"\nError copying {src_path} to {dst_path}: {e}")
                 # Optionally skip this frame and continue, or re-raise
                 continue

            metadata_list.append({
                "output_filename": filename,
                "original_id": original_id, # Original filename or frame ID
                "original_index": original_index,
                "sharpness_score": sharpness_score
            })

            if progress_bar:
                progress_bar.update(1)

        # Save metadata about the selected files
        metadata_path = os.path.join(self.output_dir, "selected_metadata.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump({
                    "input_path": self.input_path,
                    "input_type": self.input_type,
                    "total_selected": len(metadata_list),
                    "selection_method": self.selection_method,
                    "selected_items": metadata_list
                }, f, indent=2)
        except Exception as e:
             print(f"\nError writing metadata file {metadata_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract, score, and select the best frames from a video or image directory.")
    parser.add_argument("input_path", nargs="?", help="Path to the input video file or image directory")
    parser.add_argument("output_dir", nargs="?", help="Directory to save selected frames")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second to extract (video input only, default: 10)")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of frames to select (default: 300)")
    parser.add_argument("--min-buffer", type=int, default=3, help="Minimum buffer between selected frames (default: 3)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Output image format (default: jpg)")
    parser.add_argument("--force-overwrite", action="store_true", help="Overwrite existing files without confirmation")
    parser.add_argument("--selection-method", choices=["best-n", "batched", "outlier-removal"],
                       default="best-n", help="Frame selection method (default: best-n)")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Number of frames in each batch for batch selection (default: 5)")
    parser.add_argument("--batch-buffer", type=int, default=0,
                       help="Number of frames to skip between batches (default: 0)")
    parser.add_argument("--outlier-window-size", type=int, default=15,
                       help="Number of neighboring frames to compare for outlier detection (default: 15)")
    parser.add_argument("--outlier-sensitivity", type=int, default=50,
                       help="Sensitivity of outlier detection, 0-100 (default: 50)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode, prompting for options")

    args = parser.parse_args()

    # If no paths provided or interactive flag, run interactive mode
    if args.interactive or (args.input_path is None and args.output_dir is None):
        return run_interactive_mode()

    # Validate input path and determine type
    if not os.path.exists(args.input_path):
        print(f"Error: Input path not found: {args.input_path}")
        return 1

    input_type = ""
    if os.path.isfile(args.input_path):
        input_type = "video"
    elif os.path.isdir(args.input_path):
        input_type = "directory"
        print("Input path is a directory. Processing images.")
        # Ensure FPS is not used inappropriately
        if args.fps != 10: # Check if user explicitly set FPS for directory
             print("Warning: --fps argument is ignored for directory input.")
        args.fps = 0 # Set fps to 0 or None to signal directory input downstream
    else:
        print(f"Error: Input path is neither a file nor a directory: {args.input_path}")
        return 1

    # Ensure output directory is specified
    if not args.output_dir:
         print("Error: Output directory must be specified.")
         parser.print_help()
         return 1

    processor = SharpFrames(
        input_path=args.input_path,
        input_type=input_type, # Pass the detected input type
        output_dir=args.output_dir,
        fps=args.fps,
        num_frames=args.num_frames,
        min_buffer=args.min_buffer,
        output_format=args.format,
        force_overwrite=args.force_overwrite,
        selection_method=args.selection_method,
        batch_size=args.batch_size,
        batch_buffer=args.batch_buffer,
        outlier_window_size=args.outlier_window_size,
        outlier_sensitivity=args.outlier_sensitivity
    )

    success = processor.run()
    return 0 if success else 1

def run_interactive_mode():
    """Run the program in interactive mode, prompting the user for input."""
    print("\033[34m" + """
                                                                                                                                                                                                        
  ______ _     _ _______ ______  ______     _______ ______  _______ _______ _______  ______ 
 / _____|_)   (_|_______|_____ \(_____ \   (_______|_____ \(_______|_______|_______)/ _____)
( (____  _______ _______ _____) )_____) )   _____   _____) )_______ _  _  _ _____  ( (____  
 \____ \|  ___  |  ___  |  __  /|  ____/   |  ___) |  __  /|  ___  | ||_|| |  ___)  \____ \ 
 _____) ) |   | | |   | | |  \ \| |        | |     | |  \ \| |   | | |   | | |_____ _____) )
(______/|_|   |_|_|   |_|_|   |_|_|        |_|     |_|   |_|_|   |_|_|   |_|_______|______/                                                                                                                                                                      
""" + "\033[0m")

    print("\n=== Sharp Frames by Reflct.app - Interactive Mode ===")
    print("Please answer the following questions to configure the processing.\n")

    # Determine input type
    input_type_choice = get_choice(
        "Process a video file or a directory of images? (or first 3 letters)",
        ["video", "directory"],
        default="video"
    )

    input_path = ""
    if input_type_choice == "video":
        input_path = get_valid_file_path("Enter the path to the input video file: ", must_exist=True)
        input_type = "video"
        # Get frames per second only for video
        fps = get_valid_int("Enter frames per second to extract", min_value=1, max_value=60, default=10)
    else:
        # Use get_valid_dir_path but don't create or check emptiness for INPUT dir
        input_path = get_valid_dir_path(
            "Enter the path to the input image directory: ",
            create_if_missing=False,
            check_emptiness=False # Don't check emptiness for input
        )
        input_type = "directory"
        fps = 0 # Set fps to 0 for directory input

    output_dir = get_valid_dir_path(
        "Enter the output directory path (will be created if needed): ",
        create_if_missing=True,
        check_emptiness=True # Check emptiness for output
    )

    # Common options
    selection_method = get_choice(
        "Choose frame/image selection method (or first 3 letters)",
        choices=["best-n", "batched", "outlier-removal"],
        default="best-n"
    )

    # Set defaults first
    num_frames = 300
    min_buffer = 3
    batch_size = 5
    batch_buffer = 0
    outlier_window_size = 15
    outlier_sensitivity = 50

    # Get method-specific parameters
    if selection_method == "best-n":
        num_frames = get_valid_int("Enter number of frames/images to select", min_value=1, default=300)
        min_buffer = get_valid_int("Enter minimum buffer between frames/images", min_value=0, default=3)
    elif selection_method == "batched":
        batch_size = get_valid_int("Enter batch size", min_value=1, default=5)
        batch_buffer = get_valid_int("Enter batch buffer (frames/images to skip between batches)", min_value=0, default=0)
    elif selection_method == "outlier-removal":
        outlier_window_size = get_valid_int("Enter window size for comparison", min_value=3, max_value=30, default=15)
        outlier_sensitivity = get_valid_int("Enter sensitivity (0-100, higher is more aggressive)", min_value=0, max_value=100, default=50)

    output_format = get_choice(
        "Choose output format for saved images (or first 3 letters)",
        choices=["jpg", "png"],
        default="jpg"
    )
    force_overwrite = get_yes_no("Force overwrite existing files in output directory without confirmation?", default=False)

    # Print summary
    print("\n=== Configuration Summary ===")
    print(f"Input path: {input_path} (Type: {input_type})")
    print(f"Output directory: {output_dir}")
    if input_type == "video":
        print(f"FPS for extraction: {fps}")
    print(f"Selection method: {selection_method}")

    if selection_method == "best-n":
        print(f"Number of frames/images: {num_frames}")
        print(f"Minimum buffer: {min_buffer}")
    elif selection_method == "batched":
        print(f"Batch size: {batch_size}")
        print(f"Batch buffer: {batch_buffer}")
    elif selection_method == "outlier-removal":
        print(f"Window size: {outlier_window_size}")
        print(f"Sensitivity: {outlier_sensitivity}")

    print(f"Output format: {output_format}")
    print(f"Force overwrite: {'Yes' if force_overwrite else 'No'}")

    # Confirm before proceeding
    proceed = get_yes_no("\nProceed with these settings?", default=True)
    if not proceed:
        print("Operation cancelled by user.")
        return 1

    # Process the video or directory
    processor = SharpFrames(
        input_path=input_path,
        input_type=input_type,
        output_dir=output_dir,
        fps=fps,
        num_frames=num_frames,
        min_buffer=min_buffer,
        output_format=output_format,
        force_overwrite=force_overwrite,
        selection_method=selection_method,
        batch_size=batch_size,
        batch_buffer=batch_buffer,
        outlier_window_size=outlier_window_size,
        outlier_sensitivity=outlier_sensitivity
    )

    success = processor.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())