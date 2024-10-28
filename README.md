# Manga-Translator
Python script to assist with translating manga by drawing bounding boxes over Japanese text in images, extract and translate it, and overlay the translation within the selection area.

## Features
- Select a region in the image to translate from Japanese to English.
- Display translated text within the selection box.
- Navigate through images in the directory.
- Resize and toggle fullscreen for flexibility across monitors.

## Keybinds
- **A / Left Arrow**: Go to the previous image.
- **D / Right Arrow**: Go to the next image.
- **F**: Toggle fullscreen.
- **ESC**: Exit the application.

## Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/Bryson-Rice/Manga-Translator
    cd Manga-Translator
    ```
2. Install required dependencies:
	```bash
    pip install -r requirements.txt
    ```
3. Run the application:
	```bash
    python translator.py <directory-path>
    ```

## Python Version
- This was built using Python 3.12.4, other version of python have not been tested.

## Notes
- The default font is set to Arial; ensure arial.ttf is available, or replace it with a preferred font path.
- This tool is optimized for images with Japanese text and may not perform as well with other languages.