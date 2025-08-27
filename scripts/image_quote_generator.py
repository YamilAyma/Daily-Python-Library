#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_quote_generator.py: A script to generate images with text quotes overlaid.

This utility automates the creation of quote images for social media, presentations,
or personal use by combining text with background images.
"""
from __future__ import annotations

import sys
import os
import argparse
import csv
import textwrap
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from PIL import Image, ImageDraw, ImageFont

# Allow running as a script or as a module
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from .scripttypes import ScriptMetadata, function_metadata, FUNCTIONS_METADATA
FUNCTIONS_METADATA.clear()

# --- Script Metadata ---
metadata = ScriptMetadata(
    title="Image Quote Generator",
    description="Generates a batch of images by overlaying text quotes onto background images, with text wrapping and customization options.",
    version="0.1.0",
    author="AI",
    email="No Email",
    license="MIT",
    status="development",
    dependencies=["Pillow"],
    tags=["image-processing", "automation", "content-creation", "pillow", "text-wrapping"],
    cli=True,
    links=[
        {"Pillow Documentation": "https://pillow.readthedocs.io/en/stable/"},
        {"Google Fonts (for .ttf files)": "https://fonts.google.com/"}
    ],
    note="SETUP: pip install Pillow. You also need a .ttf or .otf font file."
)

##########################
#    HELPER FUNCTIONS
##########################

def _draw_wrapped_text(draw: ImageDraw, text: str, font: ImageFont, text_color: str, width_ratio: float = 0.8, image_width: int = 0, image_height: int = 0):
    """A helper function to draw centered, word-wrapped text on an image."""
    
    # Estimate a wrap width based on the image width
    char_width_avg = font.getlength("a")
    wrap_width_chars = int((image_width * width_ratio) / char_width_avg)
    
    # Wrap the text
    lines = textwrap.wrap(text, width=wrap_width_chars)
    
    # Calculate the total height of the text block
    total_text_height = sum([font.getbbox(line)[3] for line in lines])
    
    # Calculate the starting y-coordinate to center the text block
    current_y = (image_height - total_text_height) / 2
    
    # Draw each line of text
    for line in lines:
        line_width = font.getlength(line)
        # Calculate x-coordinate to center the line
        line_x = (image_width - line_width) / 2
        
        draw.text((line_x, current_y), line, font=font, fill=text_color)
        current_y += font.getbbox(line)[3] + 5  # Add a small padding between lines

##########################
#    SCRIPT FUNCTIONS
##########################

@function_metadata(status="development", note="Generate a series of images with quotes overlaid", category="generation", tags=["image", "text", "quotes"])
def generate_quote_images(
    quotes: List[str],
    image_paths: List[str],
    output_folder: str,
    font_path: str,
    font_size: int = 50,
    font_color: str = "white"
):
    """
    Generates a series of images with quotes overlaid.

    Args:
        quotes (List[str]): A list of quote strings to write on the images.
        image_paths (List[str]): A list of paths to the background images.
        output_folder (str): The folder where the generated images will be saved.
        font_path (str): The path to the .ttf or .otf font file.
        font_size (int, optional): The font size. Defaults to 50.
        font_color (str, optional): The color of the text (e.g., "white", "#FF0000"). Defaults to "white".
    
    Examples:
        >>> generate_quote_images(
        ...     quotes=["Life is what happens when you're busy making other plans."],
        ...     image_paths=["background1.jpg", "background2.jpg"],
        ...     output_folder="output_images",
        ...     font_path="path/to/font.ttf"
        ... )
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise ImportError("Pillow is not installed. Please run: 'pip install Pillow'") from e

    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found at '{font_path}'. Download a .ttf file (e.g., from Google Fonts).")
        
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output directory '{output_folder}' is ready.")

    font = ImageFont.truetype(font_path, font_size)
    
    for i, quote in enumerate(quotes):
        # Cycle through the available images
        image_path = image_paths[i % len(image_paths)]
        
        print(f"Processing quote {i+1}/{len(quotes)} on image '{os.path.basename(image_path)}'...")
        
        with Image.open(image_path).convert("RGBA") as img:
            draw = ImageDraw.Draw(img)
            
            _draw_wrapped_text(
                draw,
                quote,
                font,
                font_color,
                image_width=img.width,
                image_height=img.height
            )
            
            output_path = os.path.join(output_folder, f"quote_image_{i+1:03d}.png")
            img.save(output_path, "PNG")

    print(f"\nSuccessfully generated {len(quotes)} images in '{output_folder}'.")

##########################
#    CLI FUNCTIONS
##########################

def _handle_cli():
    parser = argparse.ArgumentParser(description=metadata.description)
    
    # --- Input Sources (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", type=str, help="Path to a CSV file with quotes. Assumes the first column contains the quotes.")
    input_group.add_argument("--quotes", nargs='+', help="A list of quotes provided directly as arguments.")
    
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image-folder", type=str, help="Path to a folder containing background images.")
    image_group.add_argument("--images", nargs='+', help="A list of paths to specific background images.")

    # --- Customization Options ---
    parser.add_argument("--font-path", type=str, required=True, help="Path to the .ttf or .otf font file.")
    parser.add_argument("--font-size", type=int, default=50, help="Font size for the text.")
    parser.add_argument("--font-color", type=str, default="white", help="Color of the text (e.g., 'white', '#000000').")
    parser.add_argument("--output-folder", type=str, default="generated_quotes", help="Folder to save the generated images.")

    args = parser.parse_args()

    # --- Prepare quotes ---
    quotes_list = []
    if args.csv:
        try:
            with open(args.csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                quotes_list = [row[0] for row in reader]
        except FileNotFoundError:
            print(f"Error: CSV file not found at '{args.csv}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
    else:
        quotes_list = args.quotes

    # --- Prepare image paths ---
    image_paths_list = []
    if args.image_folder:
        try:
            image_paths_list = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_paths_list:
                print(f"Error: No images found in folder '{args.image_folder}'")
                sys.exit(1)
        except FileNotFoundError:
            print(f"Error: Image folder not found at '{args.image_folder}'")
            sys.exit(1)
    else:
        image_paths_list = args.images

    # --- Run the main function ---
    try:
        generate_quote_images(
            quotes=quotes_list,
            image_paths=image_paths_list,
            output_folder=args.output_folder,
            font_path=args.font_path,
            font_size=args.font_size,
            font_color=args.font_color,
        )
    except (FileNotFoundError, ImportError, Exception) as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

##########################
#    EXECUTE
##########################

if __name__ == "__main__":
    _handle_cli()