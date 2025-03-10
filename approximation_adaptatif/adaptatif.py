import cv2
import numpy as np
import random
import math
import os

# INITIALISATION
imgName = "./tigre.jpg"
imgOriginal = cv2.imread(imgName, 1)

# Créaion automatique des dossier de sauvegarde
os.makedirs('./adaptative', exist_ok=True)
os.makedirs('./images_result',exist_ok=True)

# 2% de germes
pourcent = 2
nbGerme = int((pourcent * imgOriginal.shape[0]) / 100)
print(nbGerme)

germes = []
seuil = 100

print("OpenCV version: " + cv2.__version__)

def generateDiscreteVoronoi(imagedist, voronoiZoneNumber, d_height, d_width):
    # On utilise directement imagedist sous forme float
    distanceFromClosestSite = imagedist.astype(float, copy=True)
    sqrt2 = math.sqrt(2)

    # Directions (dx, dy, poids) pour les deux balayages
    forward_dirs = [(-1, -1, sqrt2), (-1, 0, 1), (-1, 1, sqrt2), (0, -1, 1)]
    backward_dirs = [(1, 1, sqrt2), (1, 0, 1), (1, -1, sqrt2), (0, 1, 1)]

    # Balayage vers l'avant (top-left -> bottom-right)
    for i in range(d_height):
        for j in range(d_width):
            for dx, dy, cost in forward_dirs:
                x, y = i + dx, j + dy
                if 0 <= x < d_height and 0 <= y < d_width:
                    new_dist = distanceFromClosestSite[x, y] + cost
                    if new_dist < distanceFromClosestSite[i, j]:
                        distanceFromClosestSite[i, j] = new_dist
                        voronoiZoneNumber[i, j] = voronoiZoneNumber[x, y]

    # Balayage vers l'arrière (bottom-right -> top-left)
    for i in range(d_height - 1, -1, -1):
        for j in range(d_width - 1, -1, -1):
            for dx, dy, cost in backward_dirs:
                x, y = i + dx, j + dy
                if 0 <= x < d_height and 0 <= y < d_width:
                    new_dist = distanceFromClosestSite[x, y] + cost
                    if new_dist < distanceFromClosestSite[i, j]:
                        distanceFromClosestSite[i, j] = new_dist
                        voronoiZoneNumber[i, j] = voronoiZoneNumber[x, y]

    # Tu peux retourner distanceFromClosestSite si tu en as besoin
    return distanceFromClosestSite, voronoiZoneNumber

# Calculer couleurs moyenne d'une zone de Voronoi
def calculateAverageColor(image, voronoiZoneNumber, n):
    # Initialiser un tableau pour stocker la somme des couleurs et le nombre de pixels dans chaque zone
    zoneColorSum = np.zeros((n + 1, 3))
    zonePixelCounts = np.zeros(n + 1)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Récupérer la couleur du pixel de l'image originale
            pixelColor = image[i, j]

            # Récupérer le numéro de zone du pixel dans l'image Voronoi
            zone = voronoiZoneNumber[i][j]

            zoneColorSum[zone] += pixelColor
            zonePixelCounts[zone] += 1

    # Calculer la moyenne des couleurs pour chaque zone
    zoneColors = np.zeros((n + 1, 3), dtype=int)
    for zone in range(1, n + 1):
        if zonePixelCounts[zone] > 0:
            zoneColors[zone] = (zoneColorSum[zone] / zonePixelCounts[zone]).astype(int)

    return zoneColors

def calculateVariance(image, voronoiZoneNumber, n, zoneColors):
    variance = np.zeros(n + 1)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixelColor = image[i, j]
            zone = voronoiZoneNumber[i][j]
            variance[zone] += np.sum((pixelColor - zoneColors[zone]) ** 2)

    for zone in range(1, n + 1):
        if variance[zone] != 0:
            variance[zone] /= np.count_nonzero(voronoiZoneNumber == zone)

    return variance

def generateRandomPointsInZone(voronoiZoneNumber, zone, numPoints):
    zonePoints = np.argwhere(voronoiZoneNumber == zone)
    np.random.shuffle(zonePoints)

    selectedPoints = zonePoints[:numPoints]

    return selectedPoints

def VorDiscret(voronoi, n, imgOriginal, index, germes, seuil, imgName):
    random.seed()

    nombreLignes, nombreColonnes = voronoi.shape[:2]

    voronoiZoneNumber = np.zeros((nombreLignes, nombreColonnes), dtype=int)
    imagedist = np.full((nombreLignes, nombreColonnes), nombreLignes + nombreColonnes, dtype=int)

    # Initialiser les germes
    for idx, (x, y) in enumerate(germes, start=1):
        voronoiZoneNumber[x, y] = idx
        imagedist[x, y] = 0

    # Calculer le diagramme discret de Voronoi
    generateDiscreteVoronoi(imagedist, voronoiZoneNumber, nombreLignes, nombreColonnes)

    # Calculer les couleurs moyennes des zones
    zoneColors = calculateAverageColor(imgOriginal, voronoiZoneNumber, n)

    # Calculer la variance de couleur pour chaque zone
    variance = calculateVariance(imgOriginal, voronoiZoneNumber, len(germes), zoneColors)

    isAllHeterogenous = 0
    for zone in range(1, len(germes) + 1):
        if variance[zone] > seuil:
            isAllHeterogenous += 1
            selectedPoints = generateRandomPointsInZone(voronoiZoneNumber, zone, 3)
            germes.extend([point.tolist() for point in selectedPoints])

    # Colorer l'image par les couleurs moyennes des zones
    voronoi[:, :, :] = zoneColors[voronoiZoneNumber]

    # Sauvegarder l'image générée
    path1 = f"ADAPTATIF-{index}-{imgName.split('/')[-1]}"
    cv2.imwrite(f"./adaptative/{path1}", voronoi)
    print(f"Image enregistrée ici : adaptative/{path1}")

    return isAllHeterogenous

imgOriginal = cv2.imread(imgName, 1)
if imgOriginal is None:
    print("error: image not read from file\n\n")
    exit()

colonnes = imgOriginal.shape[1]
lignes = imgOriginal.shape[0]

# Tableau de zeros
voronoi = np.zeros((imgOriginal.shape[0], imgOriginal.shape[1], 3), dtype=np.uint8)

cpt = 1
while cpt < nbGerme:
    a = random.randint(0, voronoi.shape[0] - 1)
    b = random.randint(0, voronoi.shape[1] - 1)
    germes.append([a, b])
    cpt += 1

index = 1
while True:
    isAllHeterogenous = VorDiscret(
    voronoi=voronoi,
    n=len(germes),
    imgOriginal=imgOriginal,
    index=index,
    germes=germes,
    seuil=seuil,
    imgName=imgName
)
    if isAllHeterogenous == 0:
        break
    index += 1

# Sauvegarde
path1 = "ADAPTATIF" + "-" + str(nbGerme) + "-" + imgName.split('/')[1]
cv2.imwrite("./images_result/" + path1, voronoi)
print("Image enregistrée: " + "images_result/" + path1)

# Montrer résultat
cv2.namedWindow("resultat", cv2.WINDOW_AUTOSIZE)
cv2.imshow("resultat", voronoi)

# Montrer original
cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)
cv2.imshow("imgOriginal", imgOriginal)

cv2.waitKey(0)
cv2.destroyAllWindows()