import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carica l'immagine in scala di grigi
img = cv2.imread('immagine.jpg', cv2.IMREAD_GRAYSCALE)

# Definiamo un kernel per rilevare bordi orizzontali
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]])

# Applichiamo la convoluzione usando filter2D
filtered_img = cv2.filter2D(img, ddepth=-1, kernel=kernel)

# Visualizziamo immagine originale e filtrata
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Originale')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Filtrata (bordi orizzontali)')
plt.imshow(filtered_img, cmap='gray')
plt.axis('off')

plt.show()
