import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gelboy.jpg')
img2 = cv2.imread('gelboy.jpg',0)

height, width = img2.shape[:2]

new_height = int(height * 0.5)
new_width = int(width * 0.5)
new_size = (new_width, new_height)

mitad_tam = cv2.resize(img2, new_size)

top = bottom = (img2.shape[0] - mitad_tam.shape[0]) // 2
left = right = (img2.shape[1] - mitad_tam.shape[1]) // 2

bordered = cv2.copyMakeBorder(mitad_tam, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

borderedr = cv2.copyMakeBorder(bordered, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


flipped_image = cv2.flip(img2, 1)

db = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
img_db = cv2.filter2D(borderedr,-1,db)
img_dbi = cv2.filter2D(flipped_image,-1,db)


fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (8, 8))

axs[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0][0].axis("off")
axs[0][0].set_title('Imagen Original')

axs[0][1].imshow(img2, cmap="gray")
axs[0][1].axis("off")
axs[0][1].set_title('Imagen en blanco y negro')

axs[1][0].imshow(bordered, cmap="gray")
axs[1][0].axis("off")
axs[1][0].set_title('Imagen reducida a la mitad')

axs[1][1].imshow(flipped_image, cmap="gray")
axs[1][1].axis("off")
axs[1][1].set_title('Imagen Invertida de Izq a Der')

axs[2][0].imshow(img_db, cmap="gray")
axs[2][0].axis("off")
axs[2][0].set_title('Deteccion de bordes')

axs[2][1].imshow(img_dbi, cmap="gray")
axs[2][1].axis("off")
axs[2][1].set_title('Deteccion de bordes a imagen invetida')

plt.show()