"""
GUI Window for Translation from Image Application
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from gui.components import (FileUploadButton, ModernProgressBar, 
                            TextDisplayPanel, ImagePreviewPanel, ModernButton)
from ocr_engine import OCREngine
from translator import Translator
from video_processor import VideoProcessor


class MainWindow:
    """Modern main application window"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        
        # Set modern theme colors
        self.root.configure(bg="#f1f5f9")
        
        # make window resizable with minimum size
        self.root.minsize(1100, 700)
        
        # initialize placeholders (actual loading happens in background)
        self.ocr_engine = None
        self.translator = None
        self.video_processor = None
        
        self.current_file = None
        self.extracted_text = ""
        self.translated_text = ""
        self.models_loaded = False
        
        # create UI first (shows immediately)
        self.create_ui()
        
        # load models in background after GUI is displayed
        self.root.after(100, self._load_models_background)
    
    def create_ui(self):
        """Create modern user interface"""
        
        # ===== HEADER SECTION - Compact =====
        header_frame = tk.Frame(self.root, bg="#667eea", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        # Gradient overlay simulation
        gradient_bar = tk.Frame(header_frame, bg="#764ba2", height=3)
        gradient_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Icon and title container
        title_container = tk.Frame(header_frame, bg="#667eea")
        title_container.pack(expand=True)
        
        # Main title with modern font
        title_label = tk.Label(
            title_container,
            text="Translation from Image",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#667eea"
        )
        title_label.pack(pady=(10, 0))
        
        # Subtitle
        subtitle_label = tk.Label(
            title_container,
            text="Extract text from images and translate to Hindi",
            font=("Segoe UI", 10),
            fg="#e0e7ff",
            bg="#667eea"
        )
        subtitle_label.pack(pady=(3, 0))
        
        # ===== MAIN CONTENT AREA =====
        main_container = tk.Frame(self.root, bg="#f1f5f9")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # ===== LEFT PANEL - Upload & Process =====
        left_panel = tk.Frame(main_container, bg="#ffffff", relief=tk.FLAT, bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
        
        left_content = tk.Frame(left_panel, bg="#ffffff")
        left_content.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Step 1 label with accent - compact
        step1_frame = tk.Frame(left_content, bg="#ffffff")
        step1_frame.pack(fill=tk.X, padx=15, pady=(12, 5))
        
        step1_accent = tk.Frame(step1_frame, bg="#667eea", width=4)
        step1_accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        step1_label = tk.Label(
            step1_frame,
            text="Step 1: Upload File",
            font=("Segoe UI", 13, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        step1_label.pack(anchor='w', pady=3)
        
        # File upload section
        file_types = [
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),
            ("Video Files", "*.mp4;*.avi;*.mov;*.mkv"),
            ("All Files", "*.*")
        ]
        
        self.upload_button = FileUploadButton(
            left_content,
            command=self.on_file_selected,
            file_types=file_types
        )
        self.upload_button.pack(fill=tk.X, padx=15, pady=5)
        
        # Image preview - compact size
        self.image_preview = ImagePreviewPanel(left_content, width=380, height=170)
        self.image_preview.pack(padx=15, pady=(3, 8))
        
        # Button container - side by side layout
        button_frame = tk.Frame(left_content, bg="#ffffff")
        button_frame.pack(pady=(5, 8), padx=20)
        
        # Extract button - compact purple style
        self.extract_button = ModernButton(
            button_frame,
            text="Extract",
            bg_color="#8b5cf6",
            hover_color="#7c3aed",
            width=150,
            height=42,
            corner_radius=21,
            font_size=12,
            command=self.extract_text,
            state=tk.DISABLED
        )
        self.extract_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Translate button - compact coral style
        self.process_button = ModernButton(
            button_frame,
            text="Translate",
            bg_color="#f43f5e",
            hover_color="#e11d48",
            width=150,
            height=42,
            corner_radius=21,
            font_size=12,
            command=self.translate_text,
            state=tk.DISABLED
        )
        self.process_button.pack(side=tk.LEFT)
        
        # Progress bar - compact
        self.progress_bar = ModernProgressBar(left_content)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=(5, 10))
        
        # ===== RIGHT PANEL - Results =====
        right_panel = tk.Frame(main_container, bg="#ffffff", relief=tk.FLAT)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        
        right_content = tk.Frame(right_panel, bg="#ffffff")
        right_content.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Step 2 label with accent - compact
        step2_frame = tk.Frame(right_content, bg="#ffffff")
        step2_frame.pack(fill=tk.X, padx=15, pady=(12, 8))
        
        step2_accent = tk.Frame(step2_frame, bg="#10b981", width=4)
        step2_accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        step2_label = tk.Label(
            step2_frame,
            text="Step 2: Results",
            font=("Segoe UI", 13, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        step2_label.pack(anchor='w', pady=3)
        
        # Original text panel - compact
        self.original_text_panel = TextDisplayPanel(
            right_content,
            title="Extracted English Text",
            height=6
        )
        self.original_text_panel.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 8))
        
        # Translated text panel - compact
        self.translated_text_panel = TextDisplayPanel(
            right_content,
            title="Translated Text (Hindi)",
            height=6
        )
        self.translated_text_panel.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 8))
        
        # Info panel with modern styling - more compact
        info_frame = tk.Frame(right_content, bg="#f0fdf4", relief=tk.FLAT)
        info_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.info_label = tk.Label(
            info_frame,
            text="Loading OCR engines...",
            font=("Segoe UI", 9),
            fg="#166534",
            bg="#f0fdf4"
        )
        self.info_label.pack(padx=10, pady=8)
    
    def _load_models_background(self):
        """Load OCR models in background thread"""
        def load():
            try:
                # disable TrOCR for faster loading (EasyOCR is enough for documents)
                self.ocr_engine = OCREngine(use_custom_model=False, use_tesseract=True, use_easyocr=True, use_trocr=False)
                self.translator = Translator()
                self.video_processor = VideoProcessor(self.ocr_engine)
                self.models_loaded = True
                
                # update status on main thread
                self.root.after(0, self._on_models_loaded)
            except Exception as e:
                print(f"Error loading models: {e}")
                self.root.after(0, lambda: self._on_models_error(str(e)))
        
        # run in background thread
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_models_loaded(self):
        """Called when models finish loading"""
        self.info_label.config(text="Powered by EasyOCR & Google Translate | English to Hindi")
        self.progress_bar.stop("Ready")
    
    def _on_models_error(self, error):
        """Called if model loading fails"""
        self.info_label.config(text=f"Error loading models: {error}")
    
    def on_file_selected(self, file_path):
        self.current_file = file_path
        self.extract_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.DISABLED)
        
        # Clear previous results
        self.original_text_panel.set_text("")
        self.translated_text_panel.set_text("")
        self.extracted_text = ""
        self.translated_text = ""
        
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            self.image_preview.display_image(file_path)
        else:
            self.image_preview.show_placeholder("Video file selected")
    
    def extract_text(self):
        """Extract text from the uploaded image/video"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please select a file first")
            return
        
        # check if models are loaded
        if not self.models_loaded or self.ocr_engine is None:
            messagebox.showinfo("Please Wait", "OCR engines are still loading. Please wait a moment.")
            return
        
        self.extract_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._extract_text_thread, daemon=True)
        thread.start()
    
    def _extract_text_thread(self):
        """Thread for text extraction"""
        try:
            self.root.after(0, lambda: self.progress_bar.start("Extracting text..."))
            
            is_video = self.current_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            if is_video:
                self.root.after(0, lambda: self.progress_bar.update_status("Processing video frames..."))
                result = self.video_processor.process_video(self.current_file)
                self.extracted_text = result['text']
            else:
                print(f"Processing: {self.current_file}")
                self.extracted_text, confidence = self.ocr_engine.extract_text(self.current_file)
                print(f"Extracted: {self.extracted_text}")
            
            if self.extracted_text and self.extracted_text.strip():
                self.root.after(0, lambda: self.original_text_panel.set_text(self.extracted_text))
                self.root.after(0, lambda: self.progress_bar.stop("Text Extracted Successfully!"))
                # Enable translate button
                self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))
            else:
                self.root.after(0, lambda: self.original_text_panel.set_text("No text detected in the image"))
                self.root.after(0, lambda: self.progress_bar.stop("No text found"))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.progress_bar.stop("Extraction Error"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            self.root.after(0, lambda: self.extract_button.config(state=tk.NORMAL))
    
    def translate_text(self):
        """Translate the extracted text to Hindi"""
        if not self.extracted_text or not self.extracted_text.strip():
            messagebox.showwarning("No Text", "Please extract text first")
            return
        
        self.process_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._translate_text_thread, daemon=True)
        thread.start()
    
    def _translate_text_thread(self):
        """Thread for translation"""
        try:
            self.root.after(0, lambda: self.progress_bar.start("Translating to Hindi..."))
            
            is_english = self.ocr_engine.is_english(self.extracted_text)
            print(f"Is English: {is_english}")
            
            if is_english:
                try:
                    self.translated_text = self.translator.translate(self.extracted_text)
                    print(f"Translated: {self.translated_text}")
                    
                    self.root.after(0, lambda: self.translated_text_panel.set_text(self.translated_text))
                    self.root.after(0, lambda: self.progress_bar.stop("Translation Complete!"))
                    
                except Exception as trans_error:
                    print(f"Translation error: {trans_error}")
                    self.root.after(0, lambda: self.translated_text_panel.set_text(f"Translation error: {str(trans_error)}"))
                    self.root.after(0, lambda: self.progress_bar.stop("Translation failed"))
            else:
                self.root.after(0, lambda: self.translated_text_panel.set_text("Text detected is not English"))
                self.root.after(0, lambda: self.progress_bar.stop("Not English text"))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.progress_bar.stop("Translation Error"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))

    def process_file(self):
        if not self.current_file:
            messagebox.showwarning("No File", "Please select a file first")
            return
        
        self.process_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._process_file_thread, daemon=True)
        thread.start()
    
    def _process_file_thread(self):
        try:
            self.root.after(0, lambda: self.progress_bar.start("Extracting text..."))
            
            is_video = self.current_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            if is_video:
                self.root.after(0, lambda: self.progress_bar.update_status("Processing video frames..."))
                result = self.video_processor.process_video(self.current_file)
                self.extracted_text = result['text']
            else:
                print(f"Processing: {self.current_file}")
                self.extracted_text, confidence = self.ocr_engine.extract_text(self.current_file)
                print(f"Extracted: {self.extracted_text}")
            
            if self.extracted_text and self.extracted_text.strip():
                self.root.after(0, lambda: self.original_text_panel.set_text(self.extracted_text))
                
                is_english = self.ocr_engine.is_english(self.extracted_text)
                print(f"Is English: {is_english}")
                
                if is_english:
                    self.root.after(0, lambda: self.progress_bar.update_status("Translating to Hindi..."))
                    
                    try:
                        self.translated_text = self.translator.translate(self.extracted_text)
                        print(f"Translated: {self.translated_text}")
                        
                        self.root.after(0, lambda: self.translated_text_panel.set_text(self.translated_text))
                        self.root.after(0, lambda: self.progress_bar.stop("Translation Complete!"))
                        
                    except Exception as trans_error:
                        print(f"Translation error: {trans_error}")
                        self.root.after(0, lambda: self.translated_text_panel.set_text(f"Translation error: {str(trans_error)}"))
                        self.root.after(0, lambda: self.progress_bar.stop("Translation failed"))
                else:
                    self.root.after(0, lambda: self.translated_text_panel.set_text(""))
                    self.root.after(0, lambda: self.progress_bar.stop("Not English text"))
            else:
                self.root.after(0, lambda: self.original_text_panel.set_text("No text detected"))
                self.root.after(0, lambda: self.translated_text_panel.set_text(""))
                self.root.after(0, lambda: self.progress_bar.stop("No text found"))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.progress_bar.stop("Error occurred"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))


def main():
    """Main entry point"""
    
    # Enable DPI awareness for crisp, sharp rendering on Windows
    try:
        from ctypes import windll
        # SetProcessDPIAware for Windows Vista+
        windll.user32.SetProcessDPIAware()
        # For Windows 10 version 1607+, use Per-Monitor DPI awareness
        try:
            windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except:
            try:
                windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
            except:
                pass
    except:
        pass  # Not on Windows, skip DPI settings
    
    root = tk.Tk()
    
    # Enable TkAgg DPI scaling
    try:
        root.tk.call('tk', 'scaling', 1.5)  # Adjust scaling for sharper text
    except:
        pass
    
    # Set window icon if available
    try:
        # root.iconbitmap('icon.ico')  # Add if you have an icon
        pass
    except:
        pass
    
    app = MainWindow(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
