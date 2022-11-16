#!/usr/bin/env python3

import cv2
from math import floor, sqrt
import numpy as np
from typing import Tuple, Any, List

Cv2Image = Any


def compute_euclidean_distance(point1: List[float], point2: List[float]) -> float:
    total_difference = 0
    for val1, val2 in zip(point1, point2):
        total_difference += (val1 - val2) ** 2

    return sqrt(total_difference)


def ray_generator(start_y: int, start_x: int, y_delta: float, x_delta: float, positive_direction=True) -> Tuple[int, int]:
    vector_length = sqrt(y_delta ** 2 + x_delta ** 2)
    if vector_length == 0:
        return

    # Use an additional scaling factor to avoid skipping
    # cells that are only partially pierced.
    y_delta = y_delta / (vector_length * 4)
    x_delta = x_delta / (vector_length * 4)

    if not positive_direction:
        y_delta = -1 * y_delta
        x_delta = -1 * x_delta

    current_y = start_y + 0.5
    current_x = start_x + 0.5

    while True:
        current_y += y_delta
        current_x += x_delta

        y = floor(current_y)
        x = floor(current_x)

        if y == start_y and x == start_x:
            continue

        yield (y, x)


def apply_stroke_width_transform(image: Cv2Image) -> Cv2Image:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y_max, x_max = image.shape

    edges = cv2.Canny(image, 175, 320, apertureSize=3)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    transformed_image = np.zeros(shape=image.shape, dtype=np.float64)

    for y, row in enumerate(edges):
        for x, value in enumerate(row):
            if value == 0:
                continue

            min_stroke_width = np.Inf
            stroke_width_ray = None

            current_ray = [(y, x)]
            for ray_y, ray_x in ray_generator(y, x, sobel_y[y][x], sobel_x[y][x], positive_direction=True):
                current_ray.append((ray_y, ray_x))

                if ray_y >= y_max or ray_y < 0 or \
                        ray_x >= x_max or ray_x < 0:
                    break

                if edges[ray_y][ray_x] == 0:
                    continue

                # cosine similarity < 0 => opposite vectors
                if (sobel_y[y][x] * sobel_y[ray_y][ray_x]) + \
                        (sobel_x[y][x] * sobel_x[ray_y][ray_x]) < 0:
                    stroke_width = compute_euclidean_distance(
                        (y, x), (ray_y, ray_x))

                    if stroke_width < min_stroke_width:
                        min_stroke_width = stroke_width
                        stroke_width_ray = current_ray

                break

            current_ray = [(y, x)]
            for ray_y, ray_x in ray_generator(y, x, sobel_y[y][x], sobel_x[y][x], positive_direction=False):

                if ray_y >= y_max or ray_y < 0 or \
                        ray_x >= x_max or ray_x < 0:
                    break

                if edges[ray_y][ray_x] == 0:
                    continue

                # cosine similarity < 0 => opposite vectors
                if (sobel_y[y][x] * sobel_y[ray_y][ray_x]) + \
                        (sobel_x[y][x] * sobel_x[ray_y][ray_x]) < 0:
                    stroke_width = compute_euclidean_distance(
                        (y, x), (ray_y, ray_x))

                    if stroke_width < min_stroke_width:
                        min_stroke_width = stroke_width
                        stroke_width_ray = current_ray

                break

            if min_stroke_width != np.Inf:
                for ray_y, ray_x in stroke_width_ray:

                    if transformed_image[ray_y][ray_x] == 0 or \
                            min_stroke_width < transformed_image[ray_y][ray_x]:
                        transformed_image[ray_y][ray_x] = min_stroke_width

    return transformed_image


if __name__ == "__main__":
    image_path = ""

    image = cv2.imread(image_path)

    transformed_image = apply_stroke_width_transform(image)
