"""
GUI Components for Translation from Image Application
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from PIL import Image, ImageTk
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class ModernButton(tk.Canvas):
    """Beautiful modern button with rounded corners and smooth hover effects"""
    
    def __init__(self, parent, text="", bg_color="#667eea", hover_color="#5a67d8", 
                 fg_color="white", width=200, height=45, corner_radius=22, 
                 font_size=12, command=None, state=tk.NORMAL, **kwargs):
        
        # Remove unsupported kwargs
        kwargs.pop('padx', None)
        kwargs.pop('pady', None)
        kwargs.pop('font', None)
        
        super().__init__(parent, width=width, height=height, 
                        bg=parent.cget('bg'), highlightthickness=0, **kwargs)
        
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.fg_color = fg_color
        self.text = text
        self.btn_width = width
        self.btn_height = height
        self.corner_radius = corner_radius
        self.font_size = font_size
        self.command = command
        self._state = state
        self.current_color = bg_color
        
        self._draw_button()
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
    
    def _draw_button(self):
        self.delete("all")
        
        color = self.current_color if self._state != tk.DISABLED else "#cbd5e1"
        
        # Draw rounded rectangle
        self._create_rounded_rect(2, 2, self.btn_width-2, self.btn_height-2, 
                                  self.corner_radius, fill=color, outline="")
        
        # Draw text
        text_color = self.fg_color if self._state != tk.DISABLED else "#94a3b8"
        self.create_text(self.btn_width//2, self.btn_height//2, text=self.text,
                        fill=text_color, font=("Segoe UI", self.font_size, "bold"))
    
    def _create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_enter(self, e):
        if self._state != tk.DISABLED:
            self.current_color = self.hover_color
            self._draw_button()
            self.config(cursor="hand2")
    
    def _on_leave(self, e):
        self.current_color = self.bg_color
        self._draw_button()
        self.config(cursor="")
    
    def _on_click(self, e):
        if self._state != tk.DISABLED and self.command:
            self.command()
    
    def config(self, **kwargs):
        if 'state' in kwargs:
            self._state = kwargs.pop('state')
            self._draw_button()
        if 'text' in kwargs:
            self.text = kwargs.pop('text')
            self._draw_button()
        if 'bg' in kwargs:
            self.bg_color = kwargs.pop('bg')
            self.current_color = self.bg_color
            self._draw_button()
        super().config(**kwargs)
    
    def configure(self, **kwargs):
        self.config(**kwargs)


class FileUploadButton(tk.Frame):
    """Modern file upload area with compact design"""
    
    def __init__(self, parent, command=None, file_types=None, **kwargs):
        super().__init__(parent, bg="#ffffff", **kwargs)
        
        self.command = command
        self.file_types = file_types or [("All Files", "*.*")]
        self.selected_file = None
        
        # Main container - compact
        container = tk.Frame(self, bg="#f1f5f9", relief=tk.FLAT)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # Inner container
        inner = tk.Frame(container, bg="#f1f5f9")
        inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        # Upload button - compact
        self.button = ModernButton(
            inner,
            text="Choose File",
            bg_color="#3b82f6",
            hover_color="#2563eb",
            width=150,
            height=40,
            corner_radius=20,
            font_size=11,
            command=self.browse_file
        )
        self.button.pack(pady=(5, 8))
        
        # File label
        self.file_label = tk.Label(
            inner,
            text="No file selected",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#f1f5f9"
        )
        self.file_label.pack(pady=(0, 5))
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=self.file_types
        )
        
        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            if len(filename) > 35:
                filename = filename[:32] + "..."
            self.file_label.config(
                text=f"{filename}", 
                fg="#10b981", 
                font=("Segoe UI", 10, "bold")
            )
            
            if self.command:
                self.command(file_path)
    
    def get_file(self):
        return self.selected_file


class ModernProgressBar(tk.Frame):
    """Beautiful progress bar with modern styling"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#ffffff", **kwargs)
        
        # Status label
        self.status_label = tk.Label(
            self,
            text="Ready to process",
            font=("Segoe UI", 11),
            fg="#64748b",
            bg="#ffffff"
        )
        self.status_label.pack(pady=(5, 8))
        
        # Progress bar with custom style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Modern.Horizontal.TProgressbar",
                       troughcolor='#e2e8f0',
                       bordercolor='#e2e8f0',
                       background='#667eea',
                       lightcolor='#667eea',
                       darkcolor='#764ba2',
                       borderwidth=0,
                       thickness=8)
        
        # Use determinate mode so we can control the fill
        self.progress = ttk.Progressbar(
            self,
            mode='determinate',
            length=350,
            maximum=100,
            value=0,
            style="Modern.Horizontal.TProgressbar"
        )
        self.progress.pack(pady=(0, 5))
        
        self._animating = False
        self._animation_id = None
    
    def start(self, status="Processing..."):
        """Start animated progress (simulates indeterminate with smooth animation)"""
        self.status_label.config(text=status, fg="#3b82f6", font=("Segoe UI", 11))
        self.progress['value'] = 0
        self._animating = True
        self._animate_progress()
    
    def _animate_progress(self):
        """Animate progress bar smoothly up to 90% while processing"""
        if self._animating:
            current = self.progress['value']
            # Slowly fill to 90%, leaving room for completion
            if current < 90:
                # Slower as it gets closer to 90%
                increment = max(0.5, (90 - current) / 50)
                self.progress['value'] = current + increment
            self._animation_id = self.after(50, self._animate_progress)
    
    def stop(self, status="Complete"):
        """Stop animation and set final state"""
        self._animating = False
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        
        # Set progress based on status
        if "Complete" in status or "Successfully" in status:
            # Animate to 100% on success
            self._animate_to_complete()
            self.status_label.config(text=status, fg="#10b981", font=("Segoe UI", 11, "bold"))
        elif "Error" in status or "failed" in status or "No text" in status:
            # Reset to 0 on error
            self.progress['value'] = 0
            self.status_label.config(text=status, fg="#ef4444", font=("Segoe UI", 11))
        else:
            # Reset to 0 for other statuses
            self.progress['value'] = 0
            self.status_label.config(text=status, fg="#64748b", font=("Segoe UI", 11))
    
    def _animate_to_complete(self):
        """Smoothly animate to 100%"""
        current = self.progress['value']
        if current < 100:
            self.progress['value'] = min(100, current + 5)
            self.after(20, self._animate_to_complete)
    
    def update_status(self, status):
        self.status_label.config(text=status, fg="#3b82f6")
    
    def reset(self):
        """Reset progress bar to initial state"""
        self._animating = False
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        self.progress['value'] = 0
        self.status_label.config(text="Ready to process", fg="#64748b", font=("Segoe UI", 11))


class TextDisplayPanel(tk.Frame):
    """Text display panel with copy button"""
    
    def __init__(self, parent, title="Text", height=6, **kwargs):
        super().__init__(parent, bg="#ffffff", **kwargs)
        
        # Title row with copy button
        title_frame = tk.Frame(self, bg="#ffffff")
        title_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_label = tk.Label(
            title_frame,
            text=title,
            font=("Segoe UI", 11, "bold"),
            fg="#1e293b",
            bg="#ffffff"
        )
        title_label.pack(side=tk.LEFT)
        
        # Small copy button on the right side of title
        self.copy_button = ModernButton(
            title_frame,
            text="Copy",
            bg_color="#8b5cf6",
            hover_color="#7c3aed",
            width=90,
            height=30,
            corner_radius=15,
            font_size=10,
            command=self.copy_text
        )
        self.copy_button.pack(side=tk.RIGHT)
        
        # Text container with border
        text_container = tk.Frame(self, bg="#e2e8f0", relief=tk.FLAT)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        # Text widget - compact
        self.text_widget = scrolledtext.ScrolledText(
            text_container,
            height=height,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#334155",
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=8,
            insertbackground="#667eea"
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    def set_text(self, text):
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.insert('1.0', text)
    
    def get_text(self):
        return self.text_widget.get('1.0', tk.END).strip()
    
    def copy_text(self):
        text = self.get_text()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            original_text = self.copy_button.text
            self.copy_button.text = "Copied!"
            self.copy_button.bg_color = "#10b981"
            self.copy_button.current_color = "#10b981"
            self.copy_button._draw_button()
            self.after(1500, lambda: self._reset_copy_button(original_text))
    
    def _reset_copy_button(self, original_text):
        self.copy_button.text = original_text
        self.copy_button.bg_color = "#8b5cf6"
        self.copy_button.current_color = "#8b5cf6"
        self.copy_button._draw_button()


class ImagePreviewPanel(tk.Frame):
    """Image preview panel"""
    
    def __init__(self, parent, width=400, height=250, **kwargs):
        super().__init__(parent, bg="#ffffff", **kwargs)
        
        self.preview_width = width
        self.preview_height = height
        
        # Title
        title_label = tk.Label(
            self,
            text="Preview",
            font=("Segoe UI", 12, "bold"),
            fg="#1e293b",
            bg="#ffffff"
        )
        title_label.pack(anchor='w', pady=(0, 8))
        
        # Canvas container with subtle border
        canvas_frame = tk.Frame(self, bg="#e2e8f0", relief=tk.FLAT)
        canvas_frame.pack()
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=width,
            height=height,
            bg="#f8fafc",
            relief=tk.FLAT,
            highlightthickness=0
        )
        self.canvas.pack(padx=2, pady=2)
        
        # Placeholder
        self._show_initial_placeholder()
        
        self.current_image = None
    
    def _show_initial_placeholder(self):
        self.canvas.create_text(
            self.preview_width // 2,
            self.preview_height // 2 - 15,
            text="No image loaded",
            font=("Segoe UI", 12),
            fill="#94a3b8"
        )
        self.canvas.create_text(
            self.preview_width // 2,
            self.preview_height // 2 + 15,
            text="Upload an image to preview",
            font=("Segoe UI", 10),
            fill="#cbd5e1"
        )
    
    def display_image(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((self.preview_width - 10, self.preview_height - 10), Image.Resampling.LANCZOS)
            
            self.current_image = ImageTk.PhotoImage(img)
            
            self.canvas.delete("all")
            
            x = self.preview_width // 2
            y = self.preview_height // 2
            self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.current_image)
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.show_placeholder("Error loading image")
    
    def show_placeholder(self, text="No image loaded"):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.preview_width // 2,
            self.preview_height // 2,
            text=text,
            font=("Segoe UI", 12),
            fill="#94a3b8",
            justify=tk.CENTER
        )
