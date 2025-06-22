#!/usr/bin/env python3
"""
ROM Screenshot Mosaic Generator
Creates a 10x15 grid mosaic from random interesting ROM screenshots
"""

import os
import glob
import random
from PIL import Image
import numpy as np

def create_rom_mosaic(input_dir="output/interesting_roms", output_file="rom_mosaic.png", 
                     num_samples=60, grid_cols=10, grid_rows=6):
    """
    Create a mosaic from random ROM screenshots
    
    Args:
        input_dir: Directory containing PNG screenshots
        output_file: Output mosaic filename
        num_samples: Number of images to sample (max 150)
        grid_cols: Number of columns in grid (10)
        grid_rows: Number of rows in grid (15)
    """
    
    # Validate grid size
    if grid_cols * grid_rows != num_samples:
        print(f"Warning: Grid size {grid_cols}x{grid_rows} = {grid_cols * grid_rows}, but requesting {num_samples} samples")
        num_samples = grid_cols * grid_rows
        print(f"Adjusted to {num_samples} samples")
    
    # Find all PNG files
    png_pattern = os.path.join(input_dir, "*.png")
    png_files = glob.glob(png_pattern)
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return None
    
    print(f"Found {len(png_files)} PNG files")
    
    # Randomly sample files
    if len(png_files) < num_samples:
        print(f"Warning: Only {len(png_files)} files available, using all of them")
        selected_files = png_files
        # Pad with repeats if needed
        while len(selected_files) < num_samples:
            selected_files.extend(png_files[:num_samples - len(selected_files)])
    else:
        selected_files = random.sample(png_files, num_samples)
    
    print(f"Selected {len(selected_files)} files for mosaic")
    
    # CHIP-8 display dimensions
    chip8_width = 64
    chip8_height = 32
    spacing = 1  # White pixel spacing between images
    
    # Calculate mosaic dimensions
    mosaic_width = (chip8_width * grid_cols) + (spacing * (grid_cols - 1))
    mosaic_height = (chip8_height * grid_rows) + (spacing * (grid_rows - 1))
    
    print(f"Creating mosaic: {mosaic_width}x{mosaic_height} pixels")
    print(f"Grid layout: {grid_cols} columns × {grid_rows} rows")
    
    # Create white background
    mosaic = Image.new('L', (mosaic_width, mosaic_height), 255)  # White background
    
    # Process each selected file
    for i, png_file in enumerate(selected_files):
        try:
            # Calculate grid position
            row = i // grid_cols
            col = i % grid_cols
            
            # Load and process image
            img = Image.open(png_file)
            
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize to CHIP-8 native resolution (should already be scaled)
            img_resized = img.resize((chip8_width, chip8_height), Image.NEAREST)
            
            # Calculate position in mosaic
            x_pos = col * (chip8_width + spacing)
            y_pos = row * (chip8_height + spacing)
            
            # Paste into mosaic
            mosaic.paste(img_resized, (x_pos, y_pos))
            
            if (i + 1) % 30 == 0:  # Progress every 30 images
                print(f"Processed {i + 1}/{num_samples} images...")
                
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
            # Continue with other images
    
    # Save mosaic
    mosaic.save(output_file)
    print(f"Mosaic saved as: {output_file}")
    print(f"Final dimensions: {mosaic.size[0]}x{mosaic.size[1]} pixels")
    
    return mosaic

def create_labeled_mosaic(input_dir="output/interesting_roms", output_file="rom_mosaic_labeled.png",
                         num_samples=60, grid_cols=10, grid_rows=6, font_size=8):
    """
    Create a mosaic with SHA-1 hash labels below each image
    """
    from PIL import ImageDraw, ImageFont
    import hashlib
    
    # Get the basic mosaic first
    selected_files = []
    png_pattern = os.path.join(input_dir, "*.png")
    png_files = glob.glob(png_pattern)
    
    if len(png_files) >= num_samples:
        selected_files = random.sample(png_files, num_samples)
    else:
        selected_files = png_files
        while len(selected_files) < num_samples:
            selected_files.extend(png_files[:num_samples - len(selected_files)])
    
    # CHIP-8 display dimensions
    chip8_width = 64
    chip8_height = 32
    spacing = 1
    label_height = 12  # Space for SHA-1 hash labels
    
    # Calculate mosaic dimensions (with space for labels)
    mosaic_width = (chip8_width * grid_cols) + (spacing * (grid_cols - 1))
    mosaic_height = ((chip8_height + label_height) * grid_rows) + (spacing * (grid_rows - 1))
    
    # Create white background
    mosaic = Image.new('L', (mosaic_width, mosaic_height), 255)
    draw = ImageDraw.Draw(mosaic)
    
    # Try to load a small font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    print(f"Creating labeled mosaic: {mosaic_width}x{mosaic_height} pixels")
    
    # Process each file
    for i, png_file in enumerate(selected_files):
        try:
            row = i // grid_cols
            col = i % grid_cols
            
            # Load and resize image
            img = Image.open(png_file)
            if img.mode != 'L':
                img = img.convert('L')
            img_resized = img.resize((chip8_width, chip8_height), Image.NEAREST)
            
            # Calculate position
            x_pos = col * (chip8_width + spacing)
            y_pos = row * (chip8_height + label_height + spacing)
            
            # Paste image at calculated position
            mosaic.paste(img_resized, (x_pos, y_pos))
            
            # Generate SHA-1 hash label from ROM file
            # Find corresponding .ch8 file
            png_basename = os.path.basename(png_file)
            ch8_filename = png_basename.replace('.png', '.ch8')
            ch8_path = os.path.join(input_dir, ch8_filename)
            
            sha1_label = "??????"  # Default if ROM file not found
            
            if os.path.exists(ch8_path):
                try:
                    # Calculate SHA-1 hash of ROM file
                    with open(ch8_path, 'rb') as f:
                        rom_data = f.read()
                    sha1_hash = hashlib.sha1(rom_data).hexdigest()
                    sha1_label = sha1_hash[:6].upper()  # First 6 characters, uppercase
                except Exception as e:
                    print(f"Error calculating hash for {ch8_path}: {e}")
            else:
                print(f"Warning: ROM file not found for {png_file}")
            
            # Draw SHA-1 hash label below the image
            text_x = x_pos + 1  # Small offset for better visibility
            text_y = y_pos + chip8_height + 1  # Below the image
            draw.text((text_x, text_y), sha1_label, fill=0, font=font)
            
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
    
    mosaic.save(output_file)
    print(f"Labeled mosaic saved as: {output_file}")
    return mosaic

def main():
    """Main function with options"""
    print("ROM Screenshot Mosaic Generator")
    print("=" * 40)
    
    # Check if interesting_roms directory exists
    input_dir = "output/interesting_roms"
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found!")
        print("Please make sure you have ROM screenshots in the correct location.")
        return
    
    # Count available PNG files
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(png_files)} PNG files in {input_dir}")
    
    if len(png_files) == 0:
        print("No PNG files found! Make sure you have generated screenshots.")
        return
    
    # Create basic mosaic
    print("\nCreating basic mosaic...")
    mosaic = create_rom_mosaic(
        input_dir=input_dir,
        output_file="rom_mosaic_10x6.png",
        num_samples=60,
        grid_cols=10,
        grid_rows=6
    )
    
    # Ask if user wants labeled version
    try:
        create_labeled = input("\nCreate labeled version? (y/n): ").lower().startswith('y')
        if create_labeled:
            print("Creating labeled mosaic...")
            labeled_mosaic = create_labeled_mosaic(
                input_dir=input_dir,
                output_file="rom_mosaic_10x6_labeled.png",
                num_samples=60,
                grid_cols=10,
                grid_rows=6
            )
    except KeyboardInterrupt:
        print("\nSkipping labeled version...")
    
    print("\nMosaic generation complete!")
    print("Files created:")
    print("- rom_mosaic_10x6.png (basic grid)")
    if 'labeled_mosaic' in locals():
        print("- rom_mosaic_10x6_labeled.png (with SHA-1 hashes)")

if __name__ == "__main__":
    main()