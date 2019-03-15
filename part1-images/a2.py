import cv2
import numpy as np
import random
im1 = cv2.imread("eiffel_18.jpg",0)
im2 = cv2.imread("eiffel_19.jpg",0)
im3 = cv2.imread("colosseum_12.jpg",0)
im4 = cv2.imread("colosseum_8.jpg",0)
im5 = cv2.imread("colosseum_18.jpg",0)
im6 = cv2.imread("notredame_1.jpg")

img_list=[im1,im2,im3,im4,im5,im6]
jk=['a','b','c','d','e','f']
no_images=len(img_list)


 





def no_matches(img1,img2):
	# detect features 
	orb = cv2.ORB_create(nfeatures=500)
	(keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
	(keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

	distances={}
	for i in range(0, len(keypoints1)):
		temp=[]
		dist=0
		for j in range(0,len(keypoints2)):
		   distance=cv2.norm( descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
		   dist+=distance
		   print(i,j)
		   temp.append((j,distance))
		   if(j==len(keypoints2)-1):
		   	   t=sorted(temp,key=lambda x: x[1])
		   	   num=t[0]
		   	   den=t[1]
		   	   distances[i]=(num,den,dist)

	pairs=[]

	## Ratio test and thresholding according to D.Lowe's paper 
	first_count=0
	second_count=0
	for keys,values in distances.items():
		numerator=values[0][1]
		denominator=values[1][1]
		t=values[2]/500
		
		

		
		if(numerator<t*0.65):
			first_count+=1
			if(numerator/float(denominator))<0.85:
				second_count+=1
				
				end1 = tuple(np.round(keypoints1[keys].pt).astype(int))
			
				end3 = tuple(np.round(keypoints2[values[0][0]].pt).astype(int))
				pairs.append((end1,end3))
				
	
	return pairs,second_count,first_count

############################ distt is a matrix with all the distances between all images ####################################
############################ Will be used to look up inter image distances in K-means #######################################

distt=np.zeros((no_images,no_images),dtype=np.float32)
distt.fill(-1)
print("before",distt)

##ignoring distances of images with themselves
for i in range(distt.shape[0]):
	for j in range(distt.shape[1]):
		if(i!=j):
			if(distt[i][j]==-1):
				relation,s1,f1=no_matches(img_list[i],img_list[j])
				relation_symmetric,s2,f2=no_matches(img_list[j],img_list[i])
				ctr=0
				for match1 in relation:
					for match2 in relation_symmetric:
						if(match1[0]==match2[1] and match1[1]==match2[0]):
							ctr+=1
				total_matches=(s1+s2)
				distt[i][j]=(ctr/total_matches)*100
				distt[j][i]=(ctr/total_matches)*100
print(distt)
