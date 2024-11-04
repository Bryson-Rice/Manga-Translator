import cv2
import os
import re
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
import numpy as np
import argparse

# TODO: When running with -h script will load ocr model
# Initialize the manga-ocr model and translator
ocr = MangaOcr()
translator = GoogleTranslator(source="ja", target="en")

# Initialize global variables
drawing = False
x_init, y_init = -1, -1
current_img_index = 0
img = None
img_copy = None
bounding_boxes = []
fullscreen = False
window_position = (0, 0)
window_size = (1920, 1080)


# Needed to load images in natural order | Python would try to load in the following order 010 100 101 instead of 001 002 003
def natural_sort_key(filename):
    return [
        int(text) if text.isdigit() else text for text in re.split(r"(\d+)", filename)
    ]


# Add new lines to fit within the RoI
def wrap_text(draw, text, font, max_width):
    lines = []
    words = text.split()
    current_line = []
    for word in words:
        current_line.append(word)
        bbox = draw.textbbox((0, 0), " ".join(current_line), font=font)
        width = bbox[2] - bbox[0]
        if width > max_width:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))
    return lines


# Save the Region of Interest (RoI) as an image
def save_roi(img, x1, y1, x2, y2, output_path="cropped_image.png"):
    roi = img[y1:y2, x1:x2]
    cv2.imwrite(output_path, roi)
    return output_path


def extract_and_translate_text(image_path):
    pil_image = Image.open(image_path)
    text = ocr(pil_image)
    translated_text = translator.translate(text)
    return translated_text


# Draw a white rectangle as a background for the translated text | see https://mangafonts.carrd.co/ for a list of manga fonts
def draw_translated_text(
    img, x1, y1, x2, y2, translated_text, font_path="arial.ttf", font_size=20
):
    font = ImageFont.truetype(font_path, font_size)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255))

    # Wrap the translated text
    box_width = x2 - x1
    lines = wrap_text(draw, translated_text, font, box_width)

    # Calculate line height and total text height for vertical centering
    line_height = (
        draw.textbbox((0, 0), "Test", font=font)[3]
        - draw.textbbox((0, 0), "Test", font=font)[1]
    )
    total_text_height = line_height * len(lines)
    text_y = y1 + (y2 - y1 - total_text_height) // 2

    # Draw each line of text inside the selection box, centered horizontal and vertical
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        text_x = x1 + (box_width - line_width) // 2
        draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
        text_y += line_height  # Move to the next line

    # Convert back to OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# save the RoI, perform OCR and translation and draw the result
def draw_rectangle(event, x, y, flags, param):
    global x_init, y_init, drawing, img_copy, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(x_init, x), min(y_init, y)
        x2, y2 = max(x_init, x), max(y_init, y)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Image", img_copy)
        param.append((x1, y1, x2, y2))

        if param:
            roi_path = save_roi(img, x1, y1, x2, y2)
            translated_text = extract_and_translate_text(roi_path)
            img = draw_translated_text(img, x1, y1, x2, y2, translated_text)
            cv2.imshow("Image", img)


def load_images_from_directory(directory):
    # Convert relative path to absolute path | weird errors if trying to use relative
    directory = os.path.abspath(directory)
    
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    image_files.sort(key=natural_sort_key)
    return image_files



def display_image(index, image_files):
    global img, img_copy, bounding_boxes
    img = cv2.imread(image_files[index])
    img_copy = img.copy()
    bounding_boxes.clear()
    window_title = f"Image {index + 1}/{len(image_files)}"
    cv2.imshow("Image", img_copy)
    cv2.setWindowTitle("Image", window_title)


def main(directory):
    global current_img_index, fullscreen, window_position, window_size
    
    # Normalize the directory path to avoid trailing backslashes causing issues
    directory = os.path.normpath(directory.strip('"'))
    
    image_files = load_images_from_directory(directory)
    total_images = len(image_files)

    if total_images == 0:
        print("No images found in the directory.")
        return

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", draw_rectangle, bounding_boxes)
    display_image(current_img_index, image_files)
    cv2.resizeWindow("Image", *window_size)
    cv2.moveWindow("Image", *window_position)

    while True:
        key = cv2.waitKey(0) & 0xFF
        char_key = chr(key).lower()  # Convert key press to lowercase as having capslock on prevents key binds from working

        # Left arrow or 'a' | Depending on the keyboard and OS using the left and right arrows may not work
        if char_key == 'a' or key == 81:
            if current_img_index > 0:
                current_img_index -= 1
                display_image(current_img_index, image_files)
        # Right arrow or 'd'
        elif char_key == 'd' or key == 83:
            if current_img_index < total_images - 1:
                current_img_index += 1
                display_image(current_img_index, image_files)
        # Toggle fullscreen with 'f'
        elif char_key == 'f':
            fullscreen = not fullscreen
            if fullscreen:
                window_size = (cv2.getWindowImageRect("Image")[2], cv2.getWindowImageRect("Image")[3])
                window_position = (cv2.getWindowImageRect("Image")[0], cv2.getWindowImageRect("Image")[1])
                cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Image", *window_size)
                cv2.moveWindow("Image", *window_position)
        # Esc key to exit
        elif key == 27:
            break

    cv2.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image OCR and Translation")
    parser.add_argument("directory", type=str, help="Directory containing images")
    args = parser.parse_args()
    main(args.directory)
