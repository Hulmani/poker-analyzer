#!/usr/bin/env python3
"""
Frame Capture Module - Issue #1
Captures frames from scrcpy window using screen region capture
"""

import cv2
import time
import logging
import pyautogui
import numpy as np
from typing import Optional, Tuple, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrcpyCapture:
    """Handles screen region capture from scrcpy window"""
    
    def __init__(self, target_fps: int = 30):
        """
        Initialize scrcpy screen capture
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.capture_region: Optional[Tuple[int, int, int, int]] = None
        self.is_running = False
        
        # Disable pyautogui failsafe (prevents mouse corner exit)
        pyautogui.FAILSAFE = False
        
    def find_scrcpy_window(self) -> bool:
        """
        Find scrcpy window and set capture region
        
        Returns:
            bool: True if scrcpy window found
        """
        try:
            # Take a screenshot to work with
            screenshot = pyautogui.screenshot()
            
            print("Please position mouse at TOP-LEFT corner of scrcpy window...")
            print("You have 5 seconds to position mouse, then it will auto-capture position")
            
            # Countdown
            for i in range(5, 0, -1):
                print(f"Capturing top-left in {i}...")
                time.sleep(1)
            
            top_left = pyautogui.position()
            print(f"Top-left corner captured: {top_left}")
            
            print("\nNow move mouse to BOTTOM-RIGHT corner of scrcpy window...")
            print("You have 5 seconds...")
            
            # Countdown
            for i in range(5, 0, -1):
                print(f"Capturing bottom-right in {i}...")
                time.sleep(1)
            
            bottom_right = pyautogui.position()
            print(f"Bottom-right corner captured: {bottom_right}")
            
            # Calculate capture region (x, y, width, height)
            x = top_left.x
            y = top_left.y
            width = bottom_right.x - top_left.x
            height = bottom_right.y - top_left.y
            
            if width <= 0 or height <= 0:
                logger.error("Invalid capture region dimensions")
                return False
            
            self.capture_region = (x, y, width, height)
            logger.info(f"Capture region set: {self.capture_region}")
            
            # Test capture
            test_frame = self.get_frame()
            if test_frame is None:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info(f"Screen capture dimensions: {test_frame.shape}")
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Error finding scrcpy window: {e}")
            return False
    
    def set_capture_region(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Manually set capture region
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
            
        Returns:
            bool: True if region is valid
        """
        if width <= 0 or height <= 0:
            logger.error("Invalid dimensions")
            return False
            
        self.capture_region = (x, y, width, height)
        logger.info(f"Capture region set manually: {self.capture_region}")
        
        # Set running to True before testing capture
        self.is_running = True
        
        # Test capture
        test_frame = self.get_frame()
        if test_frame is None:
            logger.error("Failed to capture test frame")
            self.is_running = False  # Reset if test fails
            return False
        
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from scrcpy window region
        
        Returns:
            Optional[np.ndarray]: Captured frame or None if failed
        """
        if not self.capture_region or not self.is_running:
            print(f"Debug: capture_region={self.capture_region}, is_running={self.is_running}")
            return None
        
        try:
            print(f"Debug: Attempting capture with region {self.capture_region}")
            # Capture screen region
            screenshot = pyautogui.screenshot(region=self.capture_region)
            print(f"Debug: Screenshot captured, size: {screenshot.size}")
            
            # Convert PIL image to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            print(f"Debug: Frame converted, shape: {frame.shape}")
            
            return frame
            
        except Exception as e:
            print(f"DEBUG ERROR: Failed to capture frame: {e}")
            print(f"ERROR TYPE: {type(e)}")
            import traceback
            traceback.print_exc()
            return None
    
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
    
    def stop(self) -> None:
        """Stop frame capture"""
        self.is_running = False
        logger.info("Frame capture stopped")

def main():
    """Test the scrcpy screen region capture functionality"""
    
    print("=== Scrcpy Screen Region Capture Test ===")
    print("Make sure scrcpy is running and showing your OnePlus 7 screen!")
    print()
    
    # Initialize capture
    capture = ScrcpyCapture(target_fps=30)
    
    # Use fixed coordinates instead of interactive setup
    print("Step 1: Setting up capture region with fixed coordinates...")
    if not capture.set_capture_region(825, 77, 418, 918):
        print("Failed to set up capture region")
        print("Check debug output above for details")
        return
    
    print("\nStep 2: Starting capture...")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reconfigure region")
    
    try:
        # Main capture loop
        frame_count = 0
        start_time = time.time()
        
        for frame in capture.frame_generator():
            # Display frame
            cv2.imshow('Scrcpy Screen Capture - Issue #1', frame)
            
            # Calculate and display FPS every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
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
            elif key == ord('r'):
                print("Reconfiguring capture region...")
                cv2.destroyAllWindows()
                if capture.find_scrcpy_window():
                    print("Region reconfigured successfully")
                else:
                    print("Failed to reconfigure region")
                    break
    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during capture: {e}")
    
    finally:
        # Cleanup
        capture.stop()
        cv2.destroyAllWindows()
        print("Capture stopped")

if __name__ == "__main__":
    main()