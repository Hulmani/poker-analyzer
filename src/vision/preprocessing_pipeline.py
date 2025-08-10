#!/usr/bin/env python3
"""
Preprocessing Pipeline - Issue #2
Enhances captured frames for better poker analysis
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PokerFrameProcessor:
    """Handles frame preprocessing for poker analysis"""
    
    def __init__(self):
        """Initialize preprocessing pipeline"""
        # Gaussian blur kernel for noise reduction
        self.blur_kernel_size = (3, 3)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid_size = (8, 8)
        
        # Sharpening kernel
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        
        # Initialize CLAHE object
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, 
                                   tileGridSize=self.clahe_tile_grid_size)
    
    def reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to frame
        
        Args:
            frame: Input frame
            
        Returns:
            Denoised frame
        """
        # Apply bilateral filter - reduces noise while preserving edges
        denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Optional: Additional Gaussian blur for extreme noise
        # denoised = cv2.GaussianBlur(denoised, self.blur_kernel_size, 0)
        
        return denoised
    
    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast for better text/card visibility
        
        Args:
            frame: Input frame
            
        Returns:
            Contrast enhanced frame
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split LAB channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel (lightness)
        l_channel_clahe = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def sharpen_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply sharpening to improve text readability
        
        Args:
            frame: Input frame
            
        Returns:
            Sharpened frame
        """
        # Apply sharpening kernel
        sharpened = cv2.filter2D(frame, -1, self.sharpen_kernel)
        
        # Blend original and sharpened (50/50 mix)
        result = cv2.addWeighted(frame, 0.5, sharpened, 0.5, 0)
        
        return result
    
    def correct_perspective(self, frame: np.ndarray, 
                          corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply perspective correction if screen appears tilted
        
        Args:
            frame: Input frame
            corners: Optional corner points for perspective correction
                   Format: [[top_left], [top_right], [bottom_right], [bottom_left]]
            
        Returns:
            Perspective corrected frame
        """
        if corners is None:
            # Auto-detect perspective correction (basic implementation)
            # For now, return original frame
            # TODO: Implement automatic corner detection for tilted screens
            return frame
        
        height, width = frame.shape[:2]
        
        # Define destination points (rectangle)
        dst_points = np.float32([
            [0, 0],                    # top-left
            [width, 0],                # top-right  
            [width, height],           # bottom-right
            [0, height]                # bottom-left
        ])
        
        # Calculate perspective transformation matrix
        src_points = np.float32(corners)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective correction
        corrected = cv2.warpPerspective(frame, matrix, (width, height))
        
        return corrected
    
    def optimize_for_text(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimize frame specifically for OCR text recognition
        
        Args:
            frame: Input frame
            
        Returns:
            Text-optimized frame
        """
        # Convert to grayscale for OCR
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply adaptive threshold for better text contrast
        # This makes text stand out against varying backgrounds
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to BGR for consistency
        text_optimized = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
        
        return text_optimized
    
    def create_card_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask to focus on card regions
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with card regions enhanced
        """
        # Convert to HSV for better color-based segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for white/light colors (typical card backgrounds)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=white_mask)
        
        return result
    
    def process_frame(self, frame: np.ndarray, 
                     enable_noise_reduction: bool = True,
                     enable_contrast_enhancement: bool = True, 
                     enable_sharpening: bool = True,
                     enable_perspective_correction: bool = False,
                     perspective_corners: Optional[np.ndarray] = None) -> dict:
        """
        Complete preprocessing pipeline
        
        Args:
            frame: Input frame
            enable_noise_reduction: Apply noise reduction
            enable_contrast_enhancement: Apply contrast enhancement  
            enable_sharpening: Apply sharpening
            enable_perspective_correction: Apply perspective correction
            perspective_corners: Corner points for perspective correction
            
        Returns:
            Dictionary with processed frames:
            - 'original': Original frame
            - 'processed': Fully processed frame
            - 'text_optimized': Frame optimized for OCR
            - 'card_mask': Frame with card regions highlighted
        """
        result = {'original': frame.copy()}
        processed = frame.copy()
        
        try:
            # Step 1: Noise reduction
            if enable_noise_reduction:
                processed = self.reduce_noise(processed)
                logger.debug("Applied noise reduction")
            
            # Step 2: Perspective correction (if needed)
            if enable_perspective_correction and perspective_corners is not None:
                processed = self.correct_perspective(processed, perspective_corners)
                logger.debug("Applied perspective correction")
            
            # Step 3: Contrast enhancement
            if enable_contrast_enhancement:
                processed = self.enhance_contrast(processed)
                logger.debug("Applied contrast enhancement")
            
            # Step 4: Sharpening
            if enable_sharpening:
                processed = self.sharpen_image(processed)
                logger.debug("Applied sharpening")
            
            # Store main processed frame
            result['processed'] = processed
            
            # Create specialized versions
            result['text_optimized'] = self.optimize_for_text(processed)
            result['card_mask'] = self.create_card_mask(processed)
            
            logger.debug("Frame processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            # Return original frame if processing fails
            result['processed'] = frame
            result['text_optimized'] = frame
            result['card_mask'] = frame
        
        return result

def test_preprocessing():
    """Test the preprocessing pipeline with sample frame"""
    
    # For testing, create a simple test pattern
    # In real use, this would come from your frame capture
    test_frame = np.ones((600, 400, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some noise for testing
    noise = np.random.randint(0, 50, test_frame.shape, dtype=np.uint8)
    noisy_frame = cv2.add(test_frame, noise)
    
    # Add some "cards" (white rectangles)
    cv2.rectangle(noisy_frame, (50, 100), (150, 200), (255, 255, 255), -1)   # Card 1
    cv2.rectangle(noisy_frame, (200, 100), (300, 200), (255, 255, 255), -1)  # Card 2
    
    # Add some text simulation
    cv2.putText(noisy_frame, "A♠", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(noisy_frame, "K♥", (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    print("Testing preprocessing pipeline...")
    
    # Initialize processor
    processor = PokerFrameProcessor()
    
    # Process frame
    results = processor.process_frame(noisy_frame)
    
    # Display results
    cv2.imshow('Original (Noisy)', results['original'])
    cv2.imshow('Processed', results['processed'])  
    cv2.imshow('Text Optimized', results['text_optimized'])
    cv2.imshow('Card Mask', results['card_mask'])
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Preprocessing test completed!")

if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    test_preprocessing()