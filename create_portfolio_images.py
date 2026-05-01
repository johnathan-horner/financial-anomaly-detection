#!/usr/bin/env python3
"""Create favicon and OG image from memoji for portfolio site"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_favicon_files(memoji_path):
    """Create favicon files from memoji"""
    # Open the original memoji image
    memoji = Image.open(memoji_path).convert("RGBA")

    # Create 32x32 favicon
    favicon_32 = memoji.resize((32, 32), Image.Resampling.LANCZOS)
    favicon_32.save('favicon.png', 'PNG')

    # Create 180x180 apple touch icon
    apple_icon = memoji.resize((180, 180), Image.Resampling.LANCZOS)
    apple_icon.save('apple-touch-icon.png', 'PNG')

    # Convert 32x32 to ICO format
    favicon_32.save('favicon.ico', format='ICO', sizes=[(32, 32)])

    print("Created favicon.png (32x32)")
    print("Created apple-touch-icon.png (180x180)")
    print("Created favicon.ico (32x32)")

def create_og_image(memoji_path):
    """Create Open Graph image with memoji"""
    # Create image with dark background
    width, height = 1200, 630
    bg_color = "#0a0e17"  # Dark background from portfolio

    # Create image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Open and resize memoji
    memoji = Image.open(memoji_path).convert("RGBA")
    # Make memoji larger - about 300px
    memoji_size = 300
    memoji = memoji.resize((memoji_size, memoji_size), Image.Resampling.LANCZOS)

    # Position memoji in upper center
    memoji_x = (width - memoji_size) // 2
    memoji_y = 80

    # Create a new image to paste memoji with transparency
    temp_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    temp_img.paste(memoji, (memoji_x, memoji_y), memoji)

    # Composite with background
    img = Image.alpha_composite(img.convert('RGBA'), temp_img).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to use system fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    except (IOError, OSError):
        # Fallback fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            subtitle_font = ImageFont.truetype("arial.ttf", 48)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

    # Text content
    title = "Johnathan Horner"
    subtitle = "AI Systems Portfolio"

    # Get text dimensions
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]

    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]

    # Position text below memoji
    title_x = (width - title_width) // 2
    title_y = memoji_y + memoji_size + 40

    subtitle_x = (width - subtitle_width) // 2
    subtitle_y = title_y + 80

    # Draw text
    draw.text((title_x, title_y), title, fill="#ffffff", font=title_font)
    draw.text((subtitle_x, subtitle_y), subtitle, fill="#94a3b8", font=subtitle_font)

    # Save image
    img.save('og-image.png', 'PNG', optimize=True)
    print("Created og-image.png (1200x630)")

def main():
    memoji_path = "/Users/johnathanhorner/Desktop/CSS-My Site /cpu memoji.png"

    if not os.path.exists(memoji_path):
        print(f"Error: Memoji file not found at {memoji_path}")
        return

    print("Creating favicon files...")
    create_favicon_files(memoji_path)

    print("Creating OG image...")
    create_og_image(memoji_path)

    print("All images created successfully!")

if __name__ == "__main__":
    main()