import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def string_map_to_image(string_map, font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', font_size = 30, index = 0):
    """
    Convert a string map to an image.

    Args:
        string_map (str): Text representation of the world.
        font_path (str): Path to a .ttf font file (optional).
        font_size (int): Font size for rendering the text.

    Returns:
        PIL.Image: Image object representing the string map.
    """
    lines = string_map.split('\n')
    max_width = max(len(line) for line in lines)
    line_height = font_size + 8
    img_width = max_width * font_size // 2 + 150  # Adjust width for padding
    img_height = len(lines) * line_height
    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    if font_path is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, font_size)
    for i, line in enumerate(lines):
        draw.text((10, i * line_height), line, fill="white", font=font)
    index_text = f"Frame {index + 1}"
    draw.text((10, img_height - line_height), index_text, fill="white", font=font)
    return image

def generate_gif_from_string_map(string_map_list, gif_path, duration=1000, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', font_size=30):
    """
    Generate a .gif file from a list of string maps.

    Args:
        string_map_list (list): List of string maps.
        gif_path (str): Path to save the .gif file.
        duration (float): Duration for each frame in the gif.
        font_path (str): Path to a .ttf font file (optional).
        font_size (int): Font size for rendering the text.
    """
    images = []
 
    for i, string_map in enumerate(string_map_list):
        image = string_map_to_image(string_map, font_path, font_size, i)
        images.append(np.array(image))

    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved at {gif_path}")
    return gif_path