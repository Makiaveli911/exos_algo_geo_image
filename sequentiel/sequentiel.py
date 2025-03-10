import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'image (taille NxM)
N, M = 200, 200  # Taille de l'image

# Initialisation de l'image avec des valeurs infinies
voronoi_map = np.full((N, M), -1)

# Points de départ (sites du diagramme de Voronoï)
germmes = 20  # Nombre de germmes
np.random.seed(42)
sites = np.random.randint(0, (N, M), (germmes, 2))

# Affecter un identifiant unique à chaque site
for idx, (x, y) in enumerate(sites):
    voronoi_map[x, y] = idx

# Balayage avant: de haut en bas, gauche à droite
for i in range(N):
    for j in range(M):
        if voronoi_map[i, j] == -1:
            min_dist = float('inf')
            closest_site = -1
            for idx, (sx, sy) in enumerate(sites):
                dist = abs(i - sx) + abs(j - sy)
                if dist < min_dist:
                    min_dist = dist
                    closest_site = idx
            voronoi_map[i, j] = closest_site

# Balayage arrière: de bas en haut, droite à gauche
for i in range(N-1, -1, -1):
    for j in range(M-1, -1, -1):
        if voronoi_map[i, j] == -1:
            min_dist = float('inf')
            closest_site = -1
            for idx, (sx, sy) in enumerate(sites):
                dist = abs(i - sx) + abs(j - sy)
                if dist < min_dist:
                    min_dist = dist
                    closest_site = idx
            voronoi_map[i, j] = closest_site

# Affichage du résultat avec couleurs distinctes
plt.figure(figsize=(8, 8))
plt.imshow(voronoi_map, cmap='tab10', origin='upper')
plt.scatter(sites[:, 1], sites[:, 0], c='red', marker='o', edgecolors='black', s=100, label="Sites")
plt.title("Diagramme de Voronoï Discret - Algorithme Séquentiel")
plt.legend()
plt.axis("off")
plt.show()
