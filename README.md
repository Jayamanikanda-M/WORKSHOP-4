# WORKSHOP-4
## Coin-Detection-using-OpenCV-in-Python
# AIM
To detect and count the total number of coins present in an image using OpenCV morphological operations, thresholding, and SimpleBlobDetector.

# ALGORITHM
Step 1 — Read the Image

Step 2 — Convert to Grayscale

Step 3 — Split into B, G and R Channels

Step 4 — Perform Thresholding

Step 5 — Perform Morphological Operations

# PROGRAM
Developed by JAYAMANIKANDA M REG NO:- 212225230113
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read image
image = cv2.imread("seashell.jpg")

# Display original image
imageCopy = image.copy()
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.show()


# Step 2: Convert image to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(imageGray, cmap="gray")
plt.title("Grayscale Image")
plt.show()


# Step 3: Split image into B, G and R channels
imageB, imageG, imageR = cv2.split(image)

plt.figure(figsize=(20, 12))
plt.subplot(141)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(142)
plt.imshow(imageB, cmap="gray")
plt.title("Blue Channel")

plt.subplot(143)
plt.imshow(imageG, cmap="gray")
plt.title("Green Channel")

plt.subplot(144)
plt.imshow(imageR, cmap="gray")
plt.title("Red Channel")

plt.show()


# Step 4: Thresholding
_, imageThreshold = cv2.threshold(
    imageG,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

plt.imshow(imageThreshold, cmap="gray")
plt.title("Thresholded Image")
plt.show()


# Step 5: Morphological operations
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

imageDilated = cv2.dilate(
    imageThreshold,
    kernel,
    iterations=1
)

imageDilated2 = cv2.dilate(
    imageThreshold,
    kernel,
    iterations=2
)

plt.imshow(imageDilated2, cmap="gray")
plt.title("Dilated Image Iteration 2")
plt.show()


imageEroded = cv2.erode(
    imageDilated2,
    kernel,
    iterations=1
)

plt.imshow(imageEroded, cmap="gray")
plt.title("Eroded Image")
plt.show()


# Step 6: Create SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0
params.minDistBetweenBlobs = 2

# Filter by Area
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)


# Step 7: Detect blobs
keypoints = detector.detect(imageEroded)

# Draw detected coins
output = cv2.drawKeypoints(
    image,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 10))
plt.imshow(output[:, :, ::-1])
plt.title("Final Shell Detection")
plt.axis("off")
plt.show()


# Print number of detected coins
print(f"Number of coins detected: {len(keypoints)}")
```
# Output
<img width="595" height="546" alt="image" src="https://github.com/user-attachments/assets/047c8fa3-9e83-4f82-aa7d-795595049a10" />
<img width="1301" height="626" alt="image" src="https://github.com/user-attachments/assets/3b5d4d90-0ebd-4195-97d9-d0b00339d926" />
<img width="1401" height="367" alt="image" src="https://github.com/user-attachments/assets/fcacfe93-cfca-4f78-91b9-a4e6e03e4239" />
<img width="650" height="547" alt="image" src="https://github.com/user-attachments/assets/a28e7467-e8c7-4711-ac1d-a7dce8bf5626" />
<img width="555" height="547" alt="image" src="https://github.com/user-attachments/assets/9d080532-c266-433f-a0a4-6549c7f83a60" />
<img width="737" height="546" alt="image" src="https://github.com/user-attachments/assets/aa4322ab-3061-4209-a802-ab683a99408b" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9ba90b20-92fc-4c48-af81-92f5532db1f2" />



# RESULT
Thus, Coin Detection using OpenCV in Python is executed successfully.
..







..


