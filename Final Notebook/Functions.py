#!/usr/bin/env python
# coding: utf-8

# In[1]:


def data_analysis(data):
    
    # The shape of the data
    print("The shape of the data set:")
    print(f"The data set consists of {data.shape[0]} rows and {data.shape[1]} columns.")
    
    print('\n***********************************************')
    # Missing Values Check
    print("The existence of missing values in each column:")
    print(data.isnull().any())
    
    print('\n***********************************************')
    # Info of the data
    print("General information about the data:")
    print(data.info())
    
    print('\n***********************************************')
    # Number of unique values in each column
    print("The number of unique values in each column:")
    print(data.nunique())


# In[ ]:


def create_paths_df(csv_name, dirlist, classes):
    filepaths=[]
    labels=[]
    for i,j in zip(dirlist, classes):
        filelist=os.listdir(i)
        for f in filelist:
            filepath=os.path.join (i,f)
            filepaths.append(filepath)
            labels.append(j)
    print ('filepaths:', len(filepaths), '   labels:', len(labels))

    # save the DataFrame as a CSV file
    filepath_df = pd.DataFrame({'filepath':filepaths, 'Label':labels})
    filepath_df.to_csv(csv_name, index=False)


# In[ ]:


"""
Finds the extreme points on the image and crops the rectangular out of them
"""
def crop_img(path):
    img = cv2.imread(path)
#     plt.imshow(img)
#     plt.show()
#     print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     print(gray.shape)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
#     plt.imshow(new_img)
#     plt.show()
    return new_img


# In[2]:


"""
Finds important features extracted from the main image
"""
def image_features(path):
    # image_feature_data = [mean, std, entropy, rms, kurtios, hmi]
    image_feature_data = []
    ########################### First order statical features #############################################
    img = cv2.imread(path)
    # Mean of image
    mean_of_img = np.mean(img)
    image_feature_data.append(mean_of_img)
    
    # Standard deviation of image
    std_of_img = np.std(img)
    image_feature_data.append(std_of_img)
    
    # Vaiance of image
    # Convert the image to a NumPy array
    image_array = np.array(img)
    # Compute the variance of the pixel intensities
    variance = np.var(image_array)
    image_feature_data.append(variance)
    
    # Skewness of image
    # Flatten the array to a 1D array of pixel intensities
    pixel_values = image_array.flatten()
    # Compute the skewness of the pixel intensities
    skewness = skew(pixel_values)
    image_feature_data.append(skewness)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ########################### Enotropy #############################################
    # Compute the histogram of the image
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    # Normalize the histogram
    hist = hist / np.sum(hist)
    # Compute the entropy of the image
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    image_feature_data.append(entropy)
    
    ########################### RMS #############################################
    # Compute the RMS of the image
    rms = np.sqrt(np.mean(np.square(img)))
    image_feature_data.append(rms)
    
    ########################### kurtosis #############################################
    # Calculate the kurtosis of the image
    k = kurtosis(img.flatten())
    image_feature_data.append(k)
    ########################### HMI #############################################
    # Calculate the image moments of the grayscale image
    moments = cv2.moments(gray)
    # Calculate the centroid of the grayscale image
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    # Define the radius of the circular mask
    radius = int(min(gray.shape) * 0.4)
    # Create the circular mask
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    # Apply the mask to the grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    # Calculate the Hu moments of the masked grayscale image
    hu_moments = cv2.HuMoments(cv2.moments(masked_gray)).flatten()
    # Normalize the Hu moments
    hu_moments_normalized = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
    # Extract the first four Hu moment invariants
    # (HU1) brightness or intensity of the image
    # (HU2) is related to the symmetry of the image around its vertical axis
    # (HU3) is related to the symmetry of the image around its horizontal axis.
    # (HU4) is related to the skewness of the image, or the degree to which it is asymmetrical.
    hu_invariants = hu_moments_normalized[:4]
    image_feature_data.append(hu_moments_normalized[0])
    image_feature_data.append(hu_moments_normalized[1])
    image_feature_data.append(hu_moments_normalized[2])
    image_feature_data.append(hu_moments_normalized[3])
    return image_feature_data

"""
Finds GLCM features extracted from the main image
"""
def get_feature(img):
    fet = []
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    distances = [1]  # Specify the pixel distances for the GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Specify the angles for the GLCM
    glcm = graycomatrix(image_gray, distances, angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    for i in range(len(contrast[0])):
        fet.append(contrast[0][i])
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    for x in range(len(dissimilarity[0])):
        fet.append(dissimilarity[0][x])
    homogeneity = graycoprops(glcm, 'homogeneity')
    for y in range(len(homogeneity[0])):
        fet.append(homogeneity [0][y])
    energy = graycoprops(glcm, 'energy')
    for z in range(len(energy[0])):
        fet.append(energy[0][z])
    correlation = graycoprops(glcm, 'correlation')
    for v in range(len(correlation[0])):
        fet.append(correlation[0][v])
    return fet


# In[ ]:


"""
1. Finds the contours of the image
2. then extract the tumor contours
3. then extract the mean of area, perimeter, minor ans major axis of contour
"""
# the mean of all tumor contours   
def image_process_tumor_extraction(img):     
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(gray_img.shape)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    ret, thres_img = cv2.threshold(blur_img,100,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    full_contoured_img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    
    # Select the contour(s) that correspond to the tumor
    tumor_contours = []
    tumor_contour_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000 and area < 50000:
            tumor_contour_areas.append(area)
            tumor_contours.append(contour)
       
    area_of_tumor_contours = []
    perimeter_of_tumor_contours = []
    circulatory_of_tumor_contours = []
    convexity_of_tumor_contours = []
#     fractal_dimension_tumor_contours = []
    solidity_tumor_contours = []
    diameter_tumor_contours = []
    major_axis = []
    minor_axis = []
    eccentricity_tumor_contours = []
    # for extraction of the mean of features of the tumor_contours
    features_tumor_array = [area_of_tumor_contours, perimeter_of_tumor_contours, circulatory_of_tumor_contours, convexity_of_tumor_contours, solidity_tumor_contours, diameter_tumor_contours, major_axis, minor_axis]
#     print(len(features_tumor_array))
    if len(tumor_contours)>0:
        for cnt in tumor_contours:
            ################################ Area Feature  #################################
            area = cv2.contourArea(cnt)
            # print('Area - {}'.format(area))
            area_of_tumor_contours.append(area)
    
            ############################### perimeter Feature ##############################
            perimeter = cv2.arcLength(cnt, True)
            # print('Perimeter - {}'.format(perimeter))
            perimeter_of_tumor_contours.append(perimeter)
    
            ############################### Circle Feature ##############################
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            circulatory_of_tumor_contours.append(circularity)

            ############################### Convexity Feature ##############################
            convex_hull = cv2.convexHull(cnt)
            convexity = cv2.contourArea(convex_hull)
            convexity_of_tumor_contours.append(convexity)

            ############################### solidity Feature ##############################
            if convexity == 0 :
                solidity = 0
            else:
                solidity = float(area)/convexity
            solidity_tumor_contours.append(solidity)

            ############################### Diameter Feature ##############################
            equi_diameter = np.sqrt(4*area/np.pi)
            diameter_tumor_contours.append(equi_diameter)

            ############################### Axis Feature ##############################
            try:
                (x2,y2), (MA,ma), angle = cv2.fitEllipse(cnt)
            except:
                MA = 0
                ma = 0
            major_axis.append(MA)
            minor_axis.append(ma)
    
    features_data = [] 
    if(len(contours) > 0):
        for i in features_tumor_array:
            mean_feature = np.mean(i)
            features_data.append(mean_feature)
        return features_data

    else:
        print('No Tumor Found')
        for i in range(len(features_tumor_array)):
            features_data.append(0)
        return features_data


# In[ ]:


def create_dataframe(frames, csv_name):
    data = pd.DataFrame(columns=frames)
# features_tumor_array = [area_of_tumor_contours, perimeter_of_tumor_contours, circulatory_of_tumor_contours,
#                            convexity_of_tumor_contours, fractal_dimension_tumor_contours, solidity_tumor_contours
#                            major_axis, minor_axis, eccentricity_tumor_contours]

for i in range(len(paths['filepath'])):
    if(paths['Label'][i] == 'No'):
        img_path = paths['filepath'][i]
        d = image_features(img_path)
        img = crop_img(img_path)
        tumor_features = image_process_tumor_extraction(img)
        for i in tumor_features:
            d.append(i)
        glcm_features = get_feature(img)
        for x in glcm_features:
            d.append(x)
        d.append(0)
        new_data = pd.DataFrame([d], columns=frames)
        data = data.append(new_data, ignore_index=True)
        
    elif(paths['Label'][i] == 'pituitary'):
        img_path = paths['filepath'][i]
        d = image_features(img_path)
        img = crop_img(img_path)
        tumor_features = image_process_tumor_extraction(img)
        for i in tumor_features:
            d.append(i)
        glcm_features = get_feature(img)
        for x in glcm_features:
            d.append(x)
        d.append(1)
        new_data = pd.DataFrame([d], columns=frames)
        data = data.append(new_data, ignore_index=True)
        
    elif(paths['Label'][i] == 'meningioma'):
        img_path = paths['filepath'][i]
        d = image_features(img_path)
        img = crop_img(img_path)
        tumor_features = image_process_tumor_extraction(img)
        for i in tumor_features:
            d.append(i)
        glcm_features = get_feature(img)
        for x in glcm_features:
            d.append(x)
        d.append(2)
        new_data = pd.DataFrame([d], columns=frames)
        data = data.append(new_data, ignore_index=True)
        
    else:
        img_path = paths['filepath'][i]
        d = image_features(img_path)
        img = crop_img(img_path)
        tumor_features = image_process_tumor_extraction(img)
        for i in tumor_features:
            d.append(i)
        glcm_features = get_feature(img)
        for x in glcm_features:
            d.append(x)
        d.append(3)
        new_data = pd.DataFrame([d], columns=frames)
        data = data.append(new_data, ignore_index=True)
    data.to_csv(csv_name, index=False) 


# In[ ]:





# In[ ]:




