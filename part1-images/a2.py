import cv2
import numpy as np
import random

im1 = cv2.imread("bigben_2.jpg",0)
im2 = cv2.imread("bigben_3.jpg",0)
im3 = cv2.imread("bigben_6.jpg",0)
im4 = cv2.imread("bigben_7.jpg",0)
im5 = cv2.imread("bigben_8.jpg",0)
im6 = cv2.imread("bigben_10.jpg",0)
im7 = cv2.imread("bigben_12.jpg",0)
im8 = cv2.imread("bigben_13.jpg",0)
im9 = cv2.imread("bigben_14.jpg",0)
im10 = cv2.imread("bigben_16.jpg",0)
im11 = cv2.imread("colosseum_3.jpg",0)
im12 = cv2.imread("colosseum_4.jpg",0)
im13 = cv2.imread("colosseum_5.jpg",0)
im14 = cv2.imread("colosseum_6.jpg",0)
im15 = cv2.imread("colosseum_8.jpg",0)
im16 = cv2.imread("colosseum_11.jpg",0)
im17 = cv2.imread("colosseum_12.jpg",0)
im18 = cv2.imread("colosseum_13.jpg",0)
im19 = cv2.imread("colosseum_15.jpg",0)
im20 = cv2.imread("colosseum_18.jpg",0)
im21 = cv2.imread("eiffel_1.jpg",0)
im22 = cv2.imread("eiffel_2.jpg",0)
im23 = cv2.imread("eiffel_3.jpg",0)
im24 = cv2.imread("eiffel_5.jpg",0)
im25 = cv2.imread("eiffel_6.jpg",0)
im26 = cv2.imread("eiffel_7.jpg",0)
im27 = cv2.imread("eiffel_15.jpg",0)
im28 = cv2.imread("eiffel_18.jpg",0)
im29 = cv2.imread("eiffel_19.jpg",0)
im30 = cv2.imread("eiffel_22.jpg",0)
im31 = cv2.imread("empiresate_9.jpg",0)
im32 = cv2.imread("empiresate_10.jpg",0)
im33 = cv2.imread("empiresate_12.jpg",0)
im34 = cv2.imread("empiresate_14.jpg",0)
im35 = cv2.imread("empiresate_15.jpg",0)
im36 = cv2.imread("empiresate_16.jpg",0)
im37 = cv2.imread("empiresate_22.jpg",0)
im38 = cv2.imread("empiresate_23.jpg",0)
im39 = cv2.imread("empiresate_25.jpg",0)
im40 = cv2.imread("empiresate_27.jpg",0)
im41 = cv2.imread("londoneye_2.jpg",0)
im42 = cv2.imread("londoneye_8.jpg",0)
im43 = cv2.imread("londoneye_9.jpg",0)
im44 = cv2.imread("londoneye_12.jpg",0)
im45 = cv2.imread("londoneye_13.jpg",0)
im46 = cv2.imread("londoneye_16.jpg",0)
im47 = cv2.imread("londoneye_17.jpg",0)
im48 = cv2.imread("londoneye_21.jpg",0)
im49 = cv2.imread("londoneye_22.jpg",0)
im50 = cv2.imread("londoneye_23.jpg",0)
im51 = cv2.imread("louvre_3.jpg",0)
im52 = cv2.imread("louvre_4.jpg",0)
im53 = cv2.imread("louvre_8.jpg",0)
im54 = cv2.imread("louvre_9.jpg",0)
im55 = cv2.imread("louvre_10.jpg",0)
im56 = cv2.imread("louvre_11.jpg",0)
im57 = cv2.imread("louvre_13.jpg",0)
im58 = cv2.imread("louvre_14.jpg",0)
im59 = cv2.imread("louvre_15.jpg",0)
im60 = cv2.imread("louvre_16.jpg",0)
im61 = cv2.imread("notredame_1.jpg",0)
im62 = cv2.imread("notredame_4.jpg",0)
im63 = cv2.imread("notredame_5.jpg",0)
im64 = cv2.imread("notredame_8.jpg",0)
im65 = cv2.imread("notredame_14.jpg",0)
im66 = cv2.imread("notredame_19.jpg",0)
im67 = cv2.imread("notredame_20.jpg",0)
im68 = cv2.imread("notredame_24.jpg",0)
im69 = cv2.imread("notredame_25.jpg",0)
im70 = cv2.imread("sanmarco_1.jpg",0)
im71 = cv2.imread("sanmarco_3.jpg",0)
im72 = cv2.imread("sanmarco_4.jpg",0)
im73 = cv2.imread("sanmarco_5.jpg",0)
im74 = cv2.imread("sanmarco_13.jpg",0)
im75 = cv2.imread("sanmarco_14.jpg",0)
im76 = cv2.imread("sanmarco_18.jpg",0)
im77 = cv2.imread("sanmarco_19.jpg",0)
im78 = cv2.imread("tatemodern_2.jpg",0)
im79 = cv2.imread("tatemodern_4.jpg",0)
im80 = cv2.imread("tatemodern_6.jpg",0)
im81 = cv2.imread("tatemodern_8.jpg",0)
im82 = cv2.imread("tatemodern_9.jpg",0)
im83 = cv2.imread("tatemodern_11.jpg",0)
im84 = cv2.imread("tatemodern_13.jpg",0)
im85 = cv2.imread("tatemodern_14.jpg",0)
im86 = cv2.imread("tatemodern_16.jpg",0)
im87 = cv2.imread("tatemodern_24.jpg",0)
im88 = cv2.imread("trafalgarsquare_15.jpg",0)
im89 = cv2.imread("trafalgarsquare_16.jpg",0)
im90 = cv2.imread("trafalgarsquare_20.jpg",0)
im91 = cv2.imread("trafalgarsquare_21.jpg",0)
im92 = cv2.imread("trafalgarsquare_22.jpg",0)
im93 = cv2.imread("trafalgarsquare_25.jpg",0)

img_list = [im1,im2,im3,im4,im5,im6,im7,im8,im9,im10,im11,im12,im13,im14,im15,im16,im17,im18,im19,im20,im21,im22,im23,im24,im25,im26,im27,im28,im29,im30,im31,im32,im33,im34,im35,im36,im37,im38,im39,im40,im41,im42,im43,im44,im45,im46,im47,im48,im49,im50,im51,im52,im53,im54,im55,im56,im57,im58,im59,im60,im61,im62,im63,im64,im65,im66,im67,im68,im69,im70,im71,im72,im73,im74,im75,im76,im77,im78,im79,im80,im81,im82,im83,im84,im85,im86,im87,im88,im89,im90,im91,im92,im93]
jk = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93]
no_images = len(img_list)

def no_matches(img1,img2):
'This function calculates the number of matches between images.'        
        
        temp = []
        dist = 0
        pairs = []
        distances = {}
	
        orb = cv2.ORB_create(nfeatures=500)
	(keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
	(keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
	
	for i in range(0, len(keypoints1)):
		for j in range(0, len(keypoints2)):
		   distance = cv2.norm( descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
		   dist += distance
		   temp.append((j, distance))
		   if j == len(keypoints2)-1:
		   	   t = sorted(temp, key=lambda x: x[1])
		   	   num = t[0]
		   	   den = t[1]
		   	   distances[i] = (num,den,dist)

	# Ratio testing and thresholding based on David Lowe's paper 
	first_count = 0
	second_count = 0
	for keys, values in distances.items():
		numerator = values[0][1]
		denominator = values[1][1]
		t = values[2]/500
		if numerator < t * 0.65:
			first_count += 1
			if numerator/float(denominator) < 0.85:
				second_count += 1
				end1 = tuple(np.round(keypoints1[keys].pt).astype(int))
				end3 = tuple(np.round(keypoints2[values[0][0]].pt).astype(int))
				pairs.append((end1,end3))
				
	return pairs, second_count, first_count

distt = np.zeros((no_images,no_images),dtype=np.float32) # Distance matrix
distt.fill(-1)
      
for i in range(distt.shape[0]):
    for j in range(distt.shape[1]):
      if i != j: # Ignoring distances with self
          if distt[i][j] == -1:
              relation, s1, f1 = no_matches(img_list[i],img_list[j])
	      relation_symmetric, s2, f2 = no_matches(img_list[j],img_list[i])
	      ctr = 0
	      for match1 in relation:
	          for match2 in relation_symmetric:
		      if match1[0] == match2[1] and match1[1] == match2[0]:
		          ctr += 1
	      total_matches = (s1 + s2) + 0.0001 
	      distt[i][j] = (ctr/total_matches) * 100
	      distt[j][i] = (ctr/total_matches) * 100
print(distt)
