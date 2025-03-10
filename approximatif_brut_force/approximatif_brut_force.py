import cv2
import numpy as np
import random
from statistics import mean
import matplotlib.pyplot as plt

class Germ:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.pixels = []

    def add_pixel(self, color):
        self.pixels.append(color)

    def compute_average_color(self):
        if self.pixels:
            c = list(zip(*self.pixels))
            self.color = (
                round(mean(c[0])),
                round(mean(c[1])),
                round(mean(c[2]))
            )

class VoronoiImage:
    def __init__(self, path, num_germs):
        self.image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.height, self.width, _ = self.image.shape
        self.germs = self.generate_germs(num_germs)
        self.voronoi_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def generate_germs(self, num_germs):
        germs = []
        for _ in range(num_germs):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            color = tuple(self.image[y, x])  # Prendre la couleur de l'image originale
            germs.append(Germ(x, y, color))
        return germs

    def compute_voronoi(self):
        # Étape 2 : Attribution de chaque pixel au germe le plus proche (force brute)
        for y in range(self.height):
            for x in range(self.width):
                min_distance = float('inf')
                closest_germ = None

                for germ in self.germs:
                    dist = (x - germ.x) ** 2 + (y - germ.y) ** 2  # Distance euclidienne au carré
                    if dist < min_distance:
                        min_distance = dist
                        closest_germ = germ
                
                # Ajouter la couleur du pixel au germe correspondant
                closest_germ.add_pixel(tuple(self.image[y, x]))
                # Colorier la carte de Voronoï avec la couleur du germe
                self.voronoi_map[y, x] = closest_germ.color

        # Étape 3 : Moyenne des couleurs pour chaque région
        for germ in self.germs:
            germ.compute_average_color()

        # Mise à jour de l'image avec les couleurs moyennes
        for y in range(self.height):
            for x in range(self.width):
                for germ in self.germs:
                    if np.array_equal(self.voronoi_map[y, x], germ.color):
                        self.voronoi_map[y, x] = germ.color
                        break

    def show_and_save(self, output_path):
        cv2.imshow('Voronoi Approximation', self.voronoi_map)
        cv2.imwrite(output_path, self.voronoi_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Utilisation
voronoi = VoronoiImage("./tigre.jpg", num_germs=1000)
voronoi.compute_voronoi()
voronoi.show_and_save("voronoi_result.png")
