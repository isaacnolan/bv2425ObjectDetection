import pygame
import math
from math import sin, cos, radians
import random
import os
from PIL import Image

pygame.init()
clock = pygame.time.Clock()

#colors
colors = [
    (0, 0, 0),  # Black
    (255, 0, 0),  # Red
    (0, 0, 255),  # Blue
    (0, 255, 0),  # Green
    (254, 254, 254),  # White
    (128, 0, 128),  # Purple
    (150, 75, 0),  # Brown
    (255, 165, 0),  # Orange
]

#set up window
wn_width = 500
wn_height = 386
transparent_color = (255, 0, 255)
wn = pygame.display.set_mode((wn_width, wn_height))
wn.fill(transparent_color) 
pygame.display.set_caption("Random Shape Drawer")

max_type = 7
num_shapes = 62418 #number of shapes to generate***

def main_loop():
    # Create the output folder if it doesn't exist
    output_folder = "outputShape"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    shape_count = 0
    for color_index, color in enumerate(colors):
        for shape_type in range(max_type + 1):
            wn.fill(transparent_color)  # Clear the window with the transparent color
            draw_shape(shape_type, color)

            # Save the image
            image_name = f"shape_{shape_count + 1}.png"
            output_path = os.path.join(output_folder, image_name)
            pygame.image.save(wn, output_path)

            # Convert the specific background color to transparency using PIL
            img = Image.open(output_path)
            img = img.convert("RGBA")
            datas = img.getdata()

            newData = []
            for item in datas:
                if item[0] == transparent_color[0] and item[1] == transparent_color[1] and item[2] == transparent_color[2]:
                    newData.append((0, 0, 0, 0))
                else:
                    newData.append(item)

            img.putdata(newData)
            img.save(output_path, "PNG")

            shape_count += 1

    pygame.quit()
    quit()

def draw_shape(type, color):
    radius = wn_width / 2
    center = [wn_width / 2, wn_height / 2]
    if type == 0:
        # Rectangle
        rect_width = 500
        rect_height = 300
        pygame.draw.rect(wn, color, (wn_width / 2 - rect_width / 2, wn_height / 2 - rect_height / 2, rect_width, rect_height))
    elif type == 1:
        # Circle
        pygame.draw.circle(wn, color, (wn_width / 2, wn_height / 2), wn_height / 2)
    elif type == 2:
        # Triangle
        pygame.draw.polygon(wn, color, [[wn_width / 2 - wn_height / math.tan(math.pi / 3), wn_height], [wn_width / 2, 0], [wn_width / 2 + wn_height / math.tan(math.pi / 3), wn_height]])
    elif type == 3:
        # Semicircle
        pie(wn, color, [center[0], center[1] - 120], radius, 0, 180)
    elif type == 4:
        # Quarter circle
        pie(wn, color, [70, 0], wn_height, 0, 90)
    elif type == 5:
        # Pentagon
        sides = 5
        fit_radius = radius / (1.283)
        pentagon_points = generate_ngon_points(sides, fit_radius, [center[0], center[1] + 10])
        pygame.draw.polygon(wn, color, pentagon_points)
    elif type == 6:
        # Star
        star_points = generate_star_points(radius, center)
        pygame.draw.polygon(wn, color, star_points)
    elif type == 7:
        # Cross
        plus_sign_points = generate_plus_sign_points(wn_height / 2, center)
        pygame.draw.polygon(wn, color, plus_sign_points[:4])  # Draw the horizontal part
        pygame.draw.polygon(wn, color, plus_sign_points[4:])  # Draw the vertical part

def generate_star_points(outer_radius, center):
    tip_angle = math.radians(36)
    inner_radius = outer_radius / 2.61

    points = []

    for i in range(5):
        outer_angle = 2 * math.pi * i / 5 - math.pi / 2
        outer_x = center[0] + outer_radius * math.cos(outer_angle)
        outer_y = center[1] + outer_radius * math.sin(outer_angle)
        points.append((outer_x, outer_y))

        inner_angle = outer_angle + math.pi / 5
        inner_x = center[0] + inner_radius * math.cos(inner_angle)
        inner_y = center[1] + inner_radius * math.sin(inner_angle)
        points.append((inner_x, inner_y))

    return points

def generate_ngon_points(sides, radius, center):
    points = []
    offset_angle = (3 * math.pi / 2) + (math.pi / sides)  #start at the top of the screen

    for k in range(sides):
        angle = 2 * math.pi * k / sides - offset_angle
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)  #pygame's y-axis increases downwards
        points.append((x, y))

    return points

def pie(scr, color, center, radius, start_angle, stop_angle):
    theta = start_angle
    while theta <= stop_angle:
        pygame.draw.line(scr, color, center, (center[0] + radius * cos(radians(theta)), center[1] + radius * sin(radians(theta))), 2)
        theta += 0.01

def generate_plus_sign_points(radius, center):
    arm_width = radius * 0.39

    horizontal_rect = [
        (center[0] - radius, center[1] - arm_width),  # Top-left
        (center[0] - radius, center[1] + arm_width),  # Bottom-left
        (center[0] + radius, center[1] + arm_width),  # Bottom-right
        (center[0] + radius, center[1] - arm_width),  # Top-right
    ]

    vertical_rect = [
        (center[0] - arm_width, center[1] - radius),  # Top-left
        (center[0] - arm_width, center[1] + radius),  # Bottom-left
        (center[0] + arm_width, center[1] + radius),  # Bottom-right
        (center[0] + arm_width, center[1] - radius),  # Top-right
    ]

    return horizontal_rect + vertical_rect

main_loop()
 