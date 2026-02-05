# Translation from Image
# Main entry point for the application

import sys
import os

# add src folder to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import main

if __name__ == "__main__":
    print("=" * 60)
    print("Translation from Image - Starting Application")
    print("=" * 60)
    print()
    
    # launch the GUI
    main()
