import cv2
import matplotlib.pyplot as plt

# Load the face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Read the image
img = cv2.imread('images.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

# Draw rectangles around detected faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

for (x, y, w, h) in faces:
    # Calculate the center of the face
    center = (x + w // 2, y + h // 2)
    radius = int((w + h) / 4 * 1.3)  # Average radius based on width and height
    cv2.circle(img, center, radius, (0, 255, 0), 2) 

# Convert the image to RGB for displaying
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes
plt.show()

