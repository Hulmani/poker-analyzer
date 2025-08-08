#!/usr/bin/env python3
"""
Frame Capture Module - Issue #1
Captures frames from scrcpy video stream using OpenCV
"""

import cv2
import time
import logging
from typing import Optional, Generator
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrcpyCapture:
    """Handles frame capture from scrcpy video stream"""
    
    def __init__(self, device_index: int = 0, target_fps: int = 30):
        """
        Initialize scrcpy capture
        
        Args:
            device_index: Camera/capture device index (0 for first device)
            target_fps: Target frames per second
        """
        self.device_index = device_index
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        
    def connect(self) -> bool:
        """
        Connect to scrcpy video stream
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Try to connect to video capture device
            self.cap = cv2.VideoCapture(self.device_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video capture device {self.device_index}")
                return False
            
            # Set capture properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("Failed to capture test frame")
                self.disconnect()
                return False
            
            logger.info(f"Successfully connected to device {self.device_index}")
            logger.info(f"Frame dimensions: {test_frame.shape}")
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to capture device: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from video stream"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Disconnected from capture device")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame
        
        Returns:
            Optional[np.ndarray]: Captured frame or None if failed
        """
        if not self.cap or not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to capture frame")
            return None
        
        return frame
    
    def frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames at target FPS
        
        Yields:
            np.ndarray: Captured frames
        """
        last_frame_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time >= self.frame_interval:
                frame = self.get_frame()
                if frame is not None:
                    last_frame_time = current_time
                    yield frame
            else:
                # Sleep briefly to avoid busy waiting
                time.sleep(0.001)

def main():
    """Test the frame capture functionality"""
    
    print("Starting scrcpy frame capture test...")
    print("Make sure scrcpy is running and showing your phone screen!")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Initialize capture
    capture = ScrcpyCapture(device_index=0, target_fps=30)
    
    # Connect to video stream
    if not capture.connect():
        print("Failed to connect to video stream")
        print("Make sure:")
        print("1. scrcpy is running")
        print("2. Phone screen is being mirrored")
        print("3. No other app is using the camera")
        return
    
    try:
        # Main capture loop
        frame_count = 0
        start_time = time.time()
        
        for frame in capture.frame_generator():
            # Display frame
            cv2.imshow('Scrcpy Capture - Issue #1', frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"FPS: {current_fps:.1f}, Frames: {frame_count}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during capture: {e}")
    
    finally:
        # Cleanup
        capture.disconnect()
        cv2.destroyAllWindows()
        print("Capture stopped")

if __name__ == "__main__":
    main()