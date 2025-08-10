#!/usr/bin/env python3
"""
Integrated Frame Capture + Preprocessing
Combines Issue #1 and Issue #2
"""

import cv2
import time
import logging
import pyautogui
import numpy as np
import sys
import os

from typing import Optional, Tuple, Generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_pipeline import PokerFrameProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedScrcpyCapture:
    """Enhanced scrcpy capture with preprocessing pipeline"""
    
    def __init__(self, target_fps: int = 30):
        """
        Initialize enhanced capture with preprocessing
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.capture_region: Optional[Tuple[int, int, int, int]] = None
        self.is_running = False
        
        # Initialize preprocessing pipeline
        self.processor = PokerFrameProcessor()
        
        # Processing options (can be toggled)
        self.enable_preprocessing = True
        self.enable_noise_reduction = True
        self.enable_contrast_enhancement = True
        self.enable_sharpening = True
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
    
    def set_capture_region(self, x: int, y: int, width: int, height: int) -> bool:
        """
        Set capture region coordinates
        
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
        logger.info(f"Capture region set: {self.capture_region}")
        
        # Set running to True before testing capture
        self.is_running = True
        
        # Test capture
        test_frame = self.get_raw_frame()
        if test_frame is None:
            logger.error("Failed to capture test frame")
            self.is_running = False
            return False
        
        return True
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """
        Capture raw frame without preprocessing
        
        Returns:
            Optional[np.ndarray]: Raw captured frame or None if failed
        """
        if not self.capture_region or not self.is_running:
            return None
        
        try:
            # Capture screen region
            screenshot = pyautogui.screenshot(region=self.capture_region)
            
            # Convert PIL image to OpenCV format (BGR)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            logger.warning(f"Failed to capture frame: {e}")
            return None
    
    def get_processed_frame(self) -> Optional[dict]:
        """
        Capture and process frame through preprocessing pipeline
        
        Returns:
            Optional[dict]: Dictionary with processed frames or None if failed
        """
        raw_frame = self.get_raw_frame()
        if raw_frame is None:
            return None
        
        if not self.enable_preprocessing:
            # Return raw frame in expected format
            return {
                'original': raw_frame,
                'processed': raw_frame,
                'text_optimized': raw_frame,
                'card_mask': raw_frame
            }
        
        # Process through pipeline
        processed_frames = self.processor.process_frame(
            raw_frame,
            enable_noise_reduction=self.enable_noise_reduction,
            enable_contrast_enhancement=self.enable_contrast_enhancement,
            enable_sharpening=self.enable_sharpening
        )
        
        return processed_frames
    
    def frame_generator(self, return_processed: bool = True) -> Generator:
        """
        Generator that yields frames at target FPS
        
        Args:
            return_processed: If True, return processed frames dict,
                           if False, return raw frames
        
        Yields:
            Frames (raw or processed dict)
        """
        last_frame_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time >= self.frame_interval:
                if return_processed:
                    frames = self.get_processed_frame()
                    if frames is not None:
                        last_frame_time = current_time
                        yield frames
                else:
                    frame = self.get_raw_frame()
                    if frame is not None:
                        last_frame_time = current_time
                        yield frame
            else:
                # Sleep briefly to avoid busy waiting
                time.sleep(0.001)
    
    def toggle_preprocessing(self, enable: bool = None) -> bool:
        """
        Toggle preprocessing on/off
        
        Args:
            enable: Force enable/disable, or None to toggle
            
        Returns:
            Current state after toggle
        """
        if enable is None:
            self.enable_preprocessing = not self.enable_preprocessing
        else:
            self.enable_preprocessing = enable
        
        logger.info(f"Preprocessing {'enabled' if self.enable_preprocessing else 'disabled'}")
        return self.enable_preprocessing
    
    def stop(self) -> None:
        """Stop frame capture"""
        self.is_running = False
        logger.info("Enhanced frame capture stopped")

def main():
    """Test the integrated capture + preprocessing system"""
    
    print("=== Enhanced Scrcpy Capture with Preprocessing ===")
    print("Make sure scrcpy is running and showing your OnePlus 7 poker app!")
    print()
    print("Controls:")
    print("- 'q': Quit")
    print("- 's': Save current frames")
    print("- 'p': Toggle preprocessing on/off")
    print("- '1': Show original frame")
    print("- '2': Show processed frame") 
    print("- '3': Show text-optimized frame")
    print("- '4': Show card mask frame")
    print()
    
    # Initialize enhanced capture
    capture = EnhancedScrcpyCapture(target_fps=30)
    
    # Set capture region (your coordinates from Issue #1)
    if not capture.set_capture_region(825, 77, 418, 918):
        print("Failed to set up capture region")
        return
    
    print("Capture started successfully!")
    print("Current view: Processed frame (press 1-4 to switch views)")
    
    current_view = 'processed'  # Default view
    
    try:
        # Main capture loop
        frame_count = 0
        start_time = time.time()
        
        for frames_dict in capture.frame_generator(return_processed=True):
            # Select which frame to display based on current view
            if current_view == 'original':
                display_frame = frames_dict['original']
                window_title = 'Original Frame'
            elif current_view == 'processed':
                display_frame = frames_dict['processed'] 
                window_title = 'Processed Frame (Enhanced)'
            elif current_view == 'text_optimized':
                display_frame = frames_dict['text_optimized']
                window_title = 'Text Optimized (OCR Ready)'
            elif current_view == 'card_mask':
                display_frame = frames_dict['card_mask']
                window_title = 'Card Mask (Cards Highlighted)'
            
            # Display selected frame
            cv2.imshow(window_title, display_frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                preprocessing_status = "ON" if capture.enable_preprocessing else "OFF"
                print(f"FPS: {current_fps:.1f}, Frames: {frame_count}, Preprocessing: {preprocessing_status}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                # Save all frame versions
                timestamp = int(time.time())
                cv2.imwrite(f"original_{timestamp}.jpg", frames_dict['original'])
                cv2.imwrite(f"processed_{timestamp}.jpg", frames_dict['processed'])
                cv2.imwrite(f"text_optimized_{timestamp}.jpg", frames_dict['text_optimized'])
                cv2.imwrite(f"card_mask_{timestamp}.jpg", frames_dict['card_mask'])
                print(f"All frame versions saved with timestamp {timestamp}")
            elif key == ord('p'):
                # Toggle preprocessing
                new_state = capture.toggle_preprocessing()
                print(f"Preprocessing {'enabled' if new_state else 'disabled'}")
            elif key == ord('1'):
                cv2.destroyAllWindows()
                current_view = 'original'
                print("Switched to: Original frame view")
            elif key == ord('2'):
                cv2.destroyAllWindows()
                current_view = 'processed'  
                print("Switched to: Processed frame view")
            elif key == ord('3'):
                cv2.destroyAllWindows()
                current_view = 'text_optimized'
                print("Switched to: Text-optimized frame view")
            elif key == ord('4'):
                cv2.destroyAllWindows()
                current_view = 'card_mask'
                print("Switched to: Card mask frame view")
    
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during capture: {e}")
    
    finally:
        # Cleanup
        capture.stop()
        cv2.destroyAllWindows()
        print("Enhanced capture stopped")

if __name__ == "__main__":
    main()