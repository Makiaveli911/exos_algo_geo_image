import numpy as np
import matplotlib.pyplot as plt

# Définition de la grille et des points
N, M = 100, 100  # Dimensions de la grille (NxM)
num_points = 10  # Nombre de points générateurs

# Génération aléatoire de points (coordonnées x, y dans la grille)
points = np.random.randint(0, N, size=(num_points, 2))

# Création de la matrice d'appartenance
ownership = np.zeros((N, M), dtype=int)

# Fonction pour calculer la distance entre deux points (norme Euclidienne par défaut)
def euclidean_distance(p, q):
    return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

# Algorithme de force brute
for x in range(N):  # Parcours des lignes
    for y in range(M):  # Parcours des colonnes
        P = (x, y)  # Pixel courant
        # Trouver l'indice du générateur le plus proche
        distances = [euclidean_distance(P, (px, py)) for px, py in points]
        m = np.argmin(distances)  # Index du générateur le plus proche
        ownership[x, y] = m

# Affichage du diagramme
plt.figure(figsize=(6, 6))
plt.imshow(ownership.T, cmap="tab10", origin="lower")  # .T pour l'orientation des axes
plt.scatter(points[:, 0], points[:, 1], color="red", marker="o", edgecolor="black", s=50)  # Points générateurs
plt.title("Diagramme de Voronoï discret (force brute)")
plt.show()
