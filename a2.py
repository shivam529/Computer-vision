
import cv2
import numpy as np
import random
import heapq	
import sys
from PIL import Image
import operator

######################################################## PART 1 DOCUMENTATION ###########################################################################
#########################################################################################################################################################

## no_matches() function##
## The Function no_matches returns match endpoints between two images, and also the count, i.e, total no. of end match endpints themselves,the function##
## sends the match pair endpoints which is later used in distance matrix creation. We detect 500 features/keypoints for each image and among those only##
## those keypoints are selected with a higher response value accessed from the keypoint object returned by the Orb.detectandcompute() function. We take##
## top 65% of the data in all those response values with ,hence garbage keypoints,if any are rejected on an average for lower corner response values   ##
## where higher response value denotes higher probability of a point being a corner. To find matches among a pair,we first use distance thresholding of ##
## 64(by trial and error) and then remove duplicate values with a ratio greater than 0.85(using the concept learnt in lecture videos)
def part1():
	li=[]
	im_li=[]
	for x in range(1,len(sys.argv)-3):
	    arg=sys.argv[x+2]
	    globals()['im_{}'.format(x)]  = cv2.imread(arg)
	    im_li.append(globals()['im_{}'.format(x)])
	    li.append(arg)
	k_user=int(sys.argv[2])
	file_to_write=sys.argv[-1]
	img_list=im_li
	jk=li
	no_images=len(img_list)
	jk_c=[item[:3] for item in jk]
	def no_matches(img1,img2):
		# detect features 
		orb = cv2.ORB_create(nfeatures=500)
		(keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
		(keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
		responses1=np.array([items.response for items in keypoints1])
		responses2=np.array([items.response for items in keypoints2])
		i1=np.array((np.where(responses1>np.percentile(responses1,35)))).tolist()[0]
		i2=np.array((np.where(responses2>np.percentile(responses2,35)))).tolist()[0]
		descriptors1=[descriptors1[i] for i in i1]
		descriptors2=[descriptors2[i] for i in i2]
		di=[heapq.nsmallest(2,enumerate([cv2.norm(desc1, desc2, cv2.NORM_HAMMING) for desc2 in descriptors2]),key=lambda x:x[1]) for desc1 in descriptors1]
		pairs=[]
		count=0
		for i in range(len(di)):
			## David Lowe's ratio test
			if(di[i][0][1])<64:
				if(di[i][0][1]/di[i][1][1]<0.85):
					count+=1
					end1 = tuple(np.round(keypoints1[i].pt).astype(int))
					end2 = tuple(np.round(keypoints2[di[i][0][0]].pt).astype(int))
					pairs.append((end1,end2))
		

		return pairs,count

## Distt is a matrix with inter distance(metric) between each possible pair of images, here for any given image pair i,j we first get match count and pairs##
## via the no_matches() function for both i,j and j,i and then find synmmetric matches, also the counts in both cases in recorded and further used in our  ##
## final distance metric		
	distt=np.zeros((no_images,no_images),dtype=np.float32)
	distt.fill(-1)

	#ignoring distances of images with themselves
	for i in range(distt.shape[0]):
		print("image # ",i+1)
		for j in range(distt.shape[1]):
			if(i!=j):
				if(distt[i][j]==-1):
					relation,s1=no_matches(img_list[i],img_list[j])
					relation_symmetric,s2=no_matches(img_list[j],img_list[i])
					ctr=sum([1  if (match1[0]==match2[1] and match1[1]==match2[0]) else 0 for match1 in relation for match2 in relation_symmetric])
					total_matches=(s1+s2)/10
					if(min(s1,s2)>0.8*max(s1,s2) and ctr<10):
						total_matches=2*total_matches
					if(min(s1,s2)<0.4*max(s1,s2)):
						total_matches=0.5*total_matches
					distt[i][j]=((ctr))*total_matches
					distt[j][i]=((ctr))*total_matches
	## initial centroid selection(to select farthest centroids)
## K means plus plus used to find intial centroids to subdue randomness involved in general K-means, also since the fist centroid in K-means++ itself is##
## random I find the centroid(cluster/mean/point) with largest value and then subsequently find centroids farthest from each other iteratively, this fun##
## is called by the K-means(next function) to find intial centroids and then go on with the general K-means.

	def kmeansplus(dist_matrix,images,k):
		# a=random.sample(set(list(range(0, len(images)))), 1)
		a=[int(np.where(dist_matrix==dist_matrix.max())[0][0])]
		means=[images[elem] for elem in a]
		while(len(means)<k):
			maxim=np.inf
			temp_images=[elem for elem in images if elem not in means]
			# for mean in means:
			farthest=None
			for imgs in temp_images:
				d=max([distt[images.index(imgs)][images.index(mean)] for mean in means])
				d=d**2
				
				if(d<maxim):
					farthest=imgs
					maxim=d
			means.append(farthest)
				
		return means
## Actual K-means uses the inter distance matrix distt to find similarity/distance between a pair of images and accordingly assigns cluster where any point##
## to any given centroid as close as the larger distance metric itself, then new centroids from clusters obtained in first iteration are calulcated based on##
## the image which is the most similar to other images on an average and the process continues. The K means comtuines to run unless the means don't change in##
## last five iterations##
	## Actual Kmeans
	def kmean(dist_matrix,images,k):
		means=kmeansplus(dist_matrix,images,k)
		temp_means=[]
		for i in range(100):
			groups={}
			dist={}
			temp_images=[elem for elem in images if elem not in means]
			for imgs in temp_images:
				for mean in means:
						temp=[]
						temp.append(distt[images.index(imgs)][images.index(mean)])
						dist[imgs]=dist.get(imgs,[])+temp
			for mean in means:
				for keys,values in dist.items():
					centroid=means[values.index(max(values))]
					if(mean==centroid):
						groups[mean]=groups.get(mean,[])+[keys]
					else:
						groups[mean]=groups.get(mean,[])+[]
			g_list=[]
			for keys,values in groups.items():
				g_list.append([keys]+values)
			##new means##
			means=[]
			means_single=[]
			for items in g_list:
				biggest_mean=[]
				avg_distance=0
				if(len(items)==1):
					means_single.append(items[0])
				else:
					for points in items:
						others=items[:items.index(points)]+items[items.index(points)+1:]
						add=sum([distt[images.index(elem)][images.index(points)] for elem in others])
						if(add>avg_distance):
							if(len(biggest_mean)>0):
								del biggest_mean[-1]
							biggest_mean.append(points)
							avg_distance=add
				means=means+biggest_mean
			means=means+means_single
			temp_means.append(means)
			if(i>6):
				lastfive=temp_means[-5:]
				if(set(lastfive[0])==set(lastfive[1])==set(lastfive[2])==set(lastfive[3])==set(lastfive[4])):
					break


					
		return g_list

	clusters=kmean(distt,jk,k_user)
	## accuracy ##
	# print(clusters)
	g=clusters
	f=open(file_to_write,'w')
	for cluster in clusters:
		f.write("\n")
		for item in cluster:
			f.write(item+" ")
	Tp=0
	Tn=0
	for f in g:
		class_cluster=jk_c[jk.index(f[0])]
		for assigned in f:
			if(class_cluster==jk_c[jk.index(assigned)]):
				rest=[it for it in f if it!=assigned]
				for assigned_rest in rest:
					if(class_cluster==jk_c[jk.index(assigned_rest)]):
						Tp+=1

	for f in g:
		a=0
		rest=[it for it in g if it!=f]
		class_cluster=jk_c[jk.index(f[0])]
		elemclass=[elem for elem in f if(jk_c[jk.index(f[0])])==class_cluster]
		lengthof=len(elemclass)
		# print(lengthof)
		for assigned in rest:
			for l in assigned:
				if(class_cluster==jk_c[jk.index(l)]):
					pass
				else:
					a+=1
		# print(a)
		Tn+=a*lengthof
	print("accuracy is ",(Tp+Tn)/(no_images*(no_images-1))*100)
######################################################## PART 2 ########################################################
########################################################################################################################
def part2():
    n=int(sys.argv[2])
    
    img_1=sys.argv[3]
    img_2=sys.argv[4]
    output_img=sys.argv[5]
    
    im=Image.open(img_1)
    
    lincoln_matrix=[[0.907,.0258,-182],[-0.153,1.44,58],[-0.000306,0.000731,1]]
    
    def transformation(orig_image,matrix,output_image):
        matrix=np.array(matrix)
        print("Transformation matrix is : \n", matrix)
        print("Applying transformation on image...")
        for x in range(orig_image.width):
            for y in range(orig_image.height):
                #form homogeneous coordinates [x,y,w]
                co_ord=[x,y,1]
                co_ord=np.array(co_ord)
                #find resultant coordinates
                ans=np.dot(matrix,co_ord)
                #divide by w
                final=(ans[:-1]/ans[-1]).astype(int).tolist()
                #get pixel value from original image
                pix_orig=orig_image.getpixel((x,y))
                #if point lies between image boundaries, paste the retrieved pixel values at that point
                if (final[0]>0 and final[0]<orig_image.width) and (final[1]<orig_image.height and final[1]>0):
                    output_image.putpixel((final[0],final[1]),pix_orig)
        #output_image.show()
        output_image.save(output_img)
        print("Done!")
        return output_image
    
    def translation(orig_image,source_coord1,dest_coord1):
        new_img = Image.new('RGB', orig_image.size, color = (0,0,0))
        transformation_matrix=[[1,0,dest_coord1[0]-source_coord1[0]],[0,1,dest_coord1[1]-source_coord1[1]],[0,0,1]]
        new_img=transformation(orig_image,transformation_matrix,new_img)

    
    def euclidean(orig_image,source_coord1,dest_coord1,source_coord2,dest_coord2):
        new_img = Image.new('RGB', orig_image.size, color = (0,0,0))
        #formulation of linear equations:
        #https://franklinta.com/2014/09/08/computing-css-matrix3d-transforms/
        #piazza post 181
        temp_trans_A=[[source_coord1[0],-source_coord1[1],1,0],
                  [source_coord1[1],source_coord1[0],0,1],
                  [source_coord2[0],-source_coord2[1],1,0],
                  [source_coord2[1],source_coord2[0],0,1]]
        temp_trans_A=np.asarray(temp_trans_A)
        #array of destination coords
        temp_trans_B=[[dest_coord1[0],dest_coord1[1],dest_coord2[0],dest_coord2[1]]]
        temp_trans_B=np.asarray(temp_trans_B).T
        #solve the equations
        final=np.linalg.solve(temp_trans_A,temp_trans_B)
        transformation_matrix=[[final[0,0],-final[1,0],final[2,0]],[final[1,0],final[0,0],final[3,0]],[0,0,1]]
        #apply transformation
        new_img=transformation(orig_image,transformation_matrix,new_img)

    
    def affine(orig_image,source_coord_1,dest_coord_1,source_coord_2,dest_coord_2,source_coord_3,dest_coord_3):
        new_img = Image.new('RGB', orig_image.size, color = (0,0,0))
        #https://www.ldv.ei.tum.de/fileadmin/w00bfa/www/content_uploads/Vorlesung_3.2_SpatialTransformations.pdf pg 10
        A=[[source_coord_1[0],source_coord_1[1],1],[source_coord_2[0],source_coord_2[1],1],[source_coord_3[0],source_coord_3[1],1]]
        B=[[dest_coord_1[0],dest_coord_1[1],1],[dest_coord_2[0],dest_coord_2[1],1],[dest_coord_3[0],dest_coord_3[1],1]]
        #solve the equations
        transformation_matrix=np.linalg.solve(A,B).T
        #apply transformation
        new_img=transformation(orig_image,transformation_matrix,new_img)

    
    def projective(orig_image,source_coord_1,dest_coord_1,source_coord_2,dest_coord_2,source_coord_3,dest_coord_3,source_coord_4,dest_coord_4):
        new_img = Image.new('RGB', orig_image.size, color = (0,0,0))
        #formulation of linear equations:
        #https://franklinta.com/2014/09/08/computing-css-matrix3d-transforms/
        #http://graphics.cs.cmu.edu/courses/15-463/2008_fall/Papers/proj.pdf page 3
        projective=[[source_coord_1[0],source_coord_1[1],1,0,0,0,-(source_coord_1[0]*dest_coord_1[0]),-(source_coord_1[1]*dest_coord_1[0])],
                [source_coord_2[0],source_coord_2[1],1,0,0,0,-(source_coord_2[0]*dest_coord_2[0]),-(source_coord_2[1]*dest_coord_2[0])],
                [source_coord_3[0],source_coord_3[1],1,0,0,0,-(source_coord_3[0]*dest_coord_3[0]),-(source_coord_3[1]*dest_coord_3[0])],
                [source_coord_4[0],source_coord_4[1],1,0,0,0,-(source_coord_4[0]*dest_coord_4[0]),-(source_coord_4[1]*dest_coord_4[0])],
                [0,0,0,source_coord_1[0],source_coord_1[1],1,-(source_coord_1[0]*dest_coord_1[1]),-(source_coord_1[1]*dest_coord_1[1])],
                [0,0,0,source_coord_2[0],source_coord_2[1],1,-(source_coord_2[0]*dest_coord_2[1]),-(source_coord_2[1]*dest_coord_2[1])],
                [0,0,0,source_coord_3[0],source_coord_3[1],1,-(source_coord_3[0]*dest_coord_3[1]),-(source_coord_3[1]*dest_coord_3[1])],
                [0,0,0,source_coord_4[0],source_coord_4[1],1,-(source_coord_4[0]*dest_coord_4[1]),-(source_coord_4[1]*dest_coord_4[1])]]
        #array of destination coords
        B=[[dest_coord_1[0],dest_coord_2[0],dest_coord_3[0],dest_coord_4[0],dest_coord_1[1],dest_coord_2[1],dest_coord_3[1],dest_coord_4[1]]]
        B=np.asarray(B).T
        #solve the equations
        ans=np.linalg.solve(projective,B)
        ans=np.append(ans,[[1]],axis=0)
        transformation_matrix=np.reshape(ans,(3,3))
        #apply transformation
        new_img=transformation(orig_image,transformation_matrix,new_img)

    
    #example coordinates for book2.jpg
    #euclidean(im2,(330,365),(408,455),(172,444),(312,598))
    #projective(im2,(141,131),(318,256),(480,159),(534,372),(493,630),(316,670),(64,601),(73,473))
    #translation(im,(0,0),(10,10))
    #affine(im,(50,50),(10,100),(200,50),(200,50),(50,200),(100,250))

    
    
    if n==1:
        if len(sys.argv)<8:
            print("Please check the number of arguments provided...")
        source1=eval(sys.argv[6])
        dest1=eval(sys.argv[7])
        translation(im,source1,dest1)
    elif n==2:
        if len(sys.argv)<10:
            print("Please check the number of arguments provided...")
        source1=eval(sys.argv[6])
        dest1=eval(sys.argv[7])
        source2=eval(sys.argv[8])
        dest2=eval(sys.argv[9])
        euclidean(im,source1,dest1,source2,dest2)
    elif n==3:
        if len(sys.argv)<12:
            print("Please check the number of arguments provided...")
        source1=eval(sys.argv[6])
        dest1=eval(sys.argv[7])
        source2=eval(sys.argv[8])
        dest2=eval(sys.argv[9])
        source3=eval(sys.argv[10])
        dest3=eval(sys.argv[11])
        affine(im,source1,dest1,source2,dest2,source3,dest3)
    elif n==4:
        if len(sys.argv)<14:
            print("Please check the number of arguments provided...")
        source1=eval(sys.argv[6])
        dest1=eval(sys.argv[7])
        source2=eval(sys.argv[8])
        dest2=eval(sys.argv[9])
        source3=eval(sys.argv[10])
        dest3=eval(sys.argv[11])
        source4=eval(sys.argv[12])
        dest4=eval(sys.argv[13])
        projective(im,source1,dest1,source2,dest2,source3,dest3,source4,dest4)
########################################################PART 3########################################################
######################################################################################################################


def part3():

    
     
    print("Reading Images...\n")
    
    image1 = sys.argv[2]
    image2 = sys.argv[3]
    output = sys.argv[4]
    
    # Read input images
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)
    
    # you can increase nfeatures to adjust how many features to detect 
    orb = cv2.ORB_create(nfeatures=1000)
    
    print("Detecting features...\n")
    
    # detect features 
    (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
    
    print("Matching features.....\n")
    matches = list()
    
    # Matching features based on the hamming distance and appending them to "matches" list with the four points as a sublists
    for i in range(len(descriptors1)):
        I = i
        J = -1
        min_Hamming_dist = 1000
        for j in range(len(descriptors2)):
            temp = cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
            if temp < min_Hamming_dist:
                min_Hamming_dist = temp
                J = j
        # Threshold of hamming distance set to 10
        if min_Hamming_dist < 10:
            matches.append([int(keypoints2[J].pt[0]),int(keypoints2[J].pt[1]),int(keypoints1[I].pt[0]),int(keypoints1[I].pt[1])])
    
    
    print("Finding hypothesis....\n")
    
    # Calculating a initial hypothesis based on the 1st 4 matches
    # We are taking set of 4 points which represent a quadrilateral in the images and calculating the hypothesis
    # The math governing this matrix calculation was understood from https://franklinta.com/2014/09/08/computing-css-matrix3d-transforms/
    a = [[matches[0][0],matches[0][1],1,0,0,0,-matches[0][2]*matches[0][0],-matches[0][2]*matches[0][1]],
         [0,0,0,matches[0][0],matches[0][1],1,-matches[0][3]*matches[0][0],-matches[0][3]*matches[0][1]],
         [matches[1][0],matches[1][1],1,0,0,0,-matches[1][2]*matches[1][0],-matches[1][2]*matches[1][1]],
         [0,0,0,matches[1][0],matches[1][1],1,-matches[1][3]*matches[1][0],-matches[1][3]*matches[1][1]],
         [matches[2][0],matches[2][1],1,0,0,0,-matches[2][2]*matches[2][0],-matches[2][2]*matches[2][1]],
         [0,0,0,matches[2][0],matches[2][1],1,-matches[2][3]*matches[2][0],-matches[2][3]*matches[2][1]],
         [matches[3][0],matches[3][1],1,0,0,0,-matches[3][2]*matches[3][0],-matches[3][2]*matches[3][1]],
         [0,0,0,matches[3][0],matches[3][1],1,-matches[3][3]*matches[3][0],-matches[3][3]*matches[3][1]]]
    b = [[matches[0][2]],
         [matches[0][3]],
         [matches[1][2]],
         [matches[1][3]],
         [matches[2][2]],
         [matches[2][3]],
         [matches[3][2]],
         [matches[3][3]]]
    
    # Solving the above two matrices to find the hypothesis matrix
    h = np.linalg.solve(a, b)
    
    Hypotheses = list()
    voting = dict()
    
    # Appending the initial hypothesis to a list
    Hypotheses.append(h)
    voting[0] = 0
    
    # Calculating 80 hypotheses
    for s in range(80):
        samples = random.sample(range(1,len(matches)-1), 4)
        a = [[matches[samples[0]][0],matches[samples[0]][1],1,0,0,0,-matches[samples[0]][2]*matches[samples[0]][0],-matches[samples[0]][2]*matches[samples[0]][1]],
         [0,0,0,matches[samples[0]][0],matches[samples[0]][1],1,-matches[samples[0]][3]*matches[samples[0]][0],-matches[samples[0]][3]*matches[samples[0]][1]],
         [matches[samples[1]][0],matches[samples[1]][1],1,0,0,0,-matches[samples[1]][2]*matches[samples[1]][0],-matches[samples[1]][2]*matches[samples[1]][1]],
         [0,0,0,matches[samples[1]][0],matches[samples[1]][1],1,-matches[samples[1]][3]*matches[samples[1]][0],-matches[samples[1]][3]*matches[samples[1]][1]],
         [matches[samples[2]][0],matches[samples[2]][1],1,0,0,0,-matches[samples[2]][2]*matches[samples[2]][0],-matches[samples[2]][2]*matches[samples[2]][1]],
         [0,0,0,matches[samples[2]][0],matches[samples[2]][1],1,-matches[samples[2]][3]*matches[samples[2]][0],-matches[samples[2]][3]*matches[samples[2]][1]],
         [matches[samples[3]][0],matches[samples[3]][1],1,0,0,0,-matches[samples[3]][2]*matches[samples[3]][0],-matches[samples[3]][2]*matches[samples[3]][1]],
         [0,0,0,matches[samples[3]][0],matches[samples[3]][1],1,-matches[samples[3]][3]*matches[samples[3]][0],-matches[samples[3]][3]*matches[samples[3]][1]]]
        b = [[matches[samples[0]][2]],
             [matches[samples[0]][3]],
             [matches[samples[1]][2]],
             [matches[samples[1]][3]],
             [matches[samples[2]][2]],
             [matches[samples[2]][3]],
             [matches[samples[3]][2]],
             [matches[samples[3]][3]]]
        h = np.linalg.solve(a, b)

        new_hypothesis = True
    
        # Checking each hypothesis with existing hypothesis
        # if its a new one adding it to the hypotheses list
        for i in range(len(Hypotheses)):
    
            # To check if the hypothesis exists we are finding the element wise differences between the two hypotheses
            # if they are similar then we vote for the already existing hypothesis
            if abs(np.sum(Hypotheses[i] - h)) < 1:
                voting[i] += 1
                new_hypothesis = False
    
        # Otherwise we add this as a new hypothesis
        if new_hypothesis == True:
            Hypotheses.append(h)
            voting[len(Hypotheses) - 1] = 0
    
    # Finally get the hypothesis with the maximum number of votes
    h = Hypotheses[max(voting.items(), key=operator.itemgetter(1))[0]]
    
    # Transform that hypothesis into a 3x3 matrix, the above calculation gives the h0, h1, h2, h3...h7 values as a column vector
    H = np.array([[h[0],h[1],h[2]],
         [h[3],h[4],h[5]],
         [h[6],h[7],1]])
    
    # Now we are stitching the two images together
    # We will transform the 2nd image so as to match the 1st Image using the hypothesis calculated above
    # Then we will place this transformed image onto a new image which has dimensions to accommodate the combination of the 
    # two images. To calculate the dimensions of the combined image, we find how the corner points of 2nd image transform
    # The corners of 2nd image are as follows
    a = [[0],[0],[1]]
    b = [[img2.shape[1]],[0],[1]]
    c = [[0],[img2.shape[0]],[1]]
    d = [[img2.shape[1]],[img2.shape[0]],[1]]
    
    # Transforming the corners
    A = np.dot(H,a)
    B = np.dot(H,b)
    C = np.dot(H,c)
    D = np.dot(H,d)
    
    # storing the x and y coordinates into a list
    x = [A[1][0],B[1][0],C[1][0],D[1][0]]
    y = [A[0][0],B[0][0],C[0][0],D[0][0]]
    
    # Some custom calculations to find the dimension using the x and y coordinates found above (developed after a lot of trial and error)
    max_x = max(x)
    min_x = min(x)
    max_y = max(y)
    min_y = min(y)
    x1 = list()
    y1 = list()
    
    for i in x:
        if i < 0:
            x1.append(abs(i))
        elif i > (max_x - min_x):
            x1.append(i - (max_x - min_x))
    for i in y:
        if i < 0:
            y1.append(abs(i))
        elif i > (max_y - min_y):
            y1.append(i - (max_y - min_y))
    
    # Getting x and y offsets that will be used while building the combined image
    X = int(max(x1))
    Y = int(max(y1))
    
    if (max_x - min_x + X + 2) - img1.shape[0] < 0:
        max_x += img1.shape[0] - (max_x - min_x + X + 2)
    
    # New Image which will be used as the combined image
    img4 = np.zeros((int(max_x - min_x + X + 2), int(max_y - min_y + Y + 2), 3))
    
    shift1 = False
    if A[1][0] < 0 or B[1][0] < 0 or C[1][0] < 0 or D[1][0] < 0:
        shift1 = True 
    else:
        X = 0
    
    print("Creating panorama...\n")
    
    # Read the input images again but as color this time
    img1 = cv2.imread(image1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image2, cv2.IMREAD_COLOR)
    
    # Transforming 2nd Image using the hypothesis detected above and placing it onto the combined image
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
    
            # actual x,y coordinates
            z = [[j],[i],[1]]
    
            # Transformed x,y coordinates
            Z = np.dot(H,z)
            img4[int(Z[1]) + X][int(Z[0])] = img2[i][j]
    
    
    # Placing the 1st Image now onto the combined image, by averaging the pixel values where both images intersect
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if np.array_equal(img4[i+X][j],[0,0,0]) == True:
                img4[i+X][j] = img1[i][j]
            else:
                img4[i+X][j] = np.floor_divide((img4[i+X][j] + img1[i][j]),2)
    
    cv2.imwrite(output, img4)

if sys.argv[1]=='part2':
    part2()
elif sys.argv[1]=='part1':
	part1()
elif sys.argv[1]=='part3':
	part3()






