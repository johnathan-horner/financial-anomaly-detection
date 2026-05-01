#!/usr/bin/env python3
"""Create Open Graph image for portfolio site"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_og_image():
    # Create image with dark background matching portfolio theme
    width, height = 1200, 630
    bg_color = "#0a0e17"  # Dark background from portfolio
    text_color = "#ffffff"  # White text

    # Create image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default if not available
    try:
        # Try to use a system font
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        footer_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except (IOError, OSError):
        # Fall back to default font
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    # Text content
    title = "Johnathan Horner"
    subtitle = "AI Systems Portfolio"
    footer = "6 Production Systems | 4 Verticals | 4 AWS Certifications"

    # Get text dimensions and center them
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]

    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_height = subtitle_bbox[3] - subtitle_bbox[1]

    footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
    footer_width = footer_bbox[2] - footer_bbox[0]
    footer_height = footer_bbox[3] - footer_bbox[1]

    # Calculate positions to center text
    title_x = (width - title_width) // 2
    title_y = height // 2 - title_height - 40

    subtitle_x = (width - subtitle_width) // 2
    subtitle_y = height // 2 + 10

    footer_x = (width - footer_width) // 2
    footer_y = height - footer_height - 80

    # Draw text
    draw.text((title_x, title_y), title, fill=text_color, font=title_font)
    draw.text((subtitle_x, subtitle_y), subtitle, fill="#d8dce6", font=subtitle_font)  # Slightly muted
    draw.text((footer_x, footer_y), footer, fill="#6b7a94", font=footer_font)  # More muted

    # Save image
    img.save('og-image.png', 'PNG', optimize=True)
    print("Created og-image.png (1200x630)")

if __name__ == "__main__":
    create_og_image()