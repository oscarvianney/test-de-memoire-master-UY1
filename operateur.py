
import numpy as np

def diagonal_symmetric_lbp(img):
	new_image=np.zeros(img.shape)
	new_image = new_image.astype("float")
	img = np.asarray(img)
	#Calcul du code DSLBP de chaque pixel
	for i in range(1,img.shape[0]-1) :
		for j in range(1,img.shape[1]-1) :

			b0 = ((img[i-1,j] >= img[i,j+1]).astype("int"))

			b1 = ((img[i-1,j-1] >= img[i+1,j+1]).astype("int"))*2
					
			b2 = ((img[i,j-1] >= img[i+1,j]).astype("int"))*4

			new_image[i,j]= b0 + b1 + b2
# on enleve les bords car ils valent 0 
	dslbpImg=new_image[1:new_image.shape[0]-1,1:new_image.shape[1]-1]
	return dslbpImg

def central_symmetric_lbp(img):
	new_image=np.zeros(img.shape)
	new_image = new_image.astype("float")
	img = np.asarray(img)
	#Calcul du code CSLBP de chaque pixel
	for i in range(1,img.shape[0]-1) :
		for j in range(1,img.shape[1]-1) :
			b0 = ((img[i-1,j-1] >= img[i+1,j+1]).astype("int")) 

			b1 = ((img[i-1,j] >= img[i+1,j]).astype("int"))*2

			b2 = ((img[i-1,j+1] >= img[i+1,j-1]).astype("int"))*4
					
			b3 = ((img[i,j+1] >= img[i,j-1]).astype("int"))*8

			new_image[i,j]= b0 + b1 + b2 + b3 
# on enleve les bords 
	cslbpImg=new_image[1:new_image.shape[0]-1,1:new_image.shape[1]-1]
	return cslbpImg
