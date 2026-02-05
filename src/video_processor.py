# Video processor - extracts text from video frames
# Processes videos frame by frame and combines the extracted text

import os
import sys
import cv2
import numpy as np
from collections import Counter

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ocr_engine import OCREngine


class VideoProcessor:
    """Process videos to extract text from frames"""
    
    def __init__(self, ocr_engine=None):
        """
        Initialize video processor
        
        Args:
            ocr_engine: OCR engine to use (creates new one if not provided)
        """
        self.ocr_engine = ocr_engine or OCREngine(use_tesseract=True)
    
    def extract_frames(self, video_path, frame_interval=30, max_frames=100):
        """
        Extract frames from video at regular intervals
        
        Args:
            video_path: path to video file
            frame_interval: extract every Nth frame
            max_frames: maximum number of frames to extract
            
        Returns:
            list of frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # extract every Nth frame
            if frame_count % frame_interval == 0:
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        
        return frames
    
    def process_frame(self, frame):
        """
        Extract text from a single frame
        
        Args:
            frame: video frame as numpy array
            
        Returns:
            tuple of (text, confidence)
        """
        # save frame temporarily
        temp_path = os.path.join(config.OUTPUT_DIR, 'temp_frame.jpg')
        cv2.imwrite(temp_path, frame)
        
        # extract text
        text, confidence = self.ocr_engine.extract_text(temp_path)
        
        # clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return text, confidence
    
    def process_video(self, video_path, frame_interval=None, max_frames=None, 
                     deduplicate=True, min_confidence=0.5):
        """
        Process entire video to extract text
        
        Args:
            video_path: path to video file
            frame_interval: extract every Nth frame
            max_frames: maximum frames to process
            deduplicate: whether to remove duplicate text
            min_confidence: minimum confidence threshold
            
        Returns:
            dictionary with extracted text and metadata
        """
        print(f"\nProcessing video: {video_path}")
        
        # use config defaults if not specified
        frame_interval = frame_interval or config.VIDEO_FRAME_INTERVAL
        max_frames = max_frames or config.MAX_FRAMES_TO_PROCESS
        
        # extract frames
        frames = self.extract_frames(video_path, frame_interval, max_frames)
        
        if not frames:
            return {
                'text': '',
                'all_text': [],
                'frame_count': 0,
                'confidence': 0.0
            }
        
        # process each frame
        all_text = []
        all_confidences = []
        
        print(f"Processing {len(frames)} frames...")
        for i, frame in enumerate(frames):
            print(f"  Frame {i+1}/{len(frames)}...", end='\r')
            
            text, confidence = self.process_frame(frame)
            
            if text and confidence >= min_confidence:
                all_text.append(text.strip())
                all_confidences.append(confidence)
        
        print(f"\nExtracted text from {len(all_text)} frames")
        
        # remove duplicates if requested
        if deduplicate:
            text_counter = Counter(all_text)
            unique_texts = [text for text, count in text_counter.most_common()]
        else:
            unique_texts = all_text
        
        # combine all text
        combined_text = ' '.join(unique_texts)
        
        # calculate average confidence
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'text': combined_text,
            'all_text': unique_texts,
            'frame_count': len(frames),
            'processed_count': len(all_text),
            'confidence': avg_confidence
        }
    
    def extract_text_from_video(self, video_path):
        """Simple method to extract text from video"""
        result = self.process_video(video_path)
        return result['text']


if __name__ == "__main__":
    # test video processor
    processor = VideoProcessor()
    
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        result = processor.process_video(test_video)
        print(f"\nExtracted text: {result['text'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Frames processed: {result['processed_count']}/{result['frame_count']}")
