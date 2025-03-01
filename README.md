# Compute-Similarity-App

1. Data Parsing and Preprocessing
* Data Parsing:
We developed helper functions (parse_people_csv and get_people_images) to parse metadata from the LFW dataset. These functions map person names to their corresponding image paths, which is useful for evaluation and dataset preparation.

* Preprocessing Pipeline:
We implemented a PreprocessingPipeline class that:

Loads Images: Reads images from a file path or NumPy array.
Face Detection & Alignment: Uses MTCNN to detect faces and extract the face region.
Resizing and Normalization: Resizes the detected face to a consistent target size (e.g., 224×224) and normalizes pixel values to the [0,1] range.
2. Feature Extraction Using FaceNet
Why FaceNet?
FaceNet is a specialized model for face recognition that maps images to a 512-dimensional embedding space. It is designed so that embeddings of the same person are close together while embeddings of different people are far apart.

Implementation:
We built a FeatureExtractionPipeline class that uses the FaceNet model (InceptionResnetV1 from the facenet-pytorch library) to generate feature embeddings.

The pipeline includes a transformation that converts the preprocessed image to the expected format (resizing to 160×160, converting to tensor, and normalizing to the range [-1, 1]).
The model outputs a 512-dimensional feature vector for each image.
3. Similarity Calculation Using Cosine Similarity
Why Cosine Similarity?
Cosine similarity is a measure of the angle between two vectors. When embeddings are L2-normalized (as is standard with FaceNet), cosine similarity is particularly effective:
A value close to 1 indicates very similar (or nearly identical) embeddings.
A lower value indicates dissimilar faces.
Implementation:
We implemented helper functions to:
L2-normalize the feature vectors.
Calculate cosine similarity between two embeddings.
Convert the cosine similarity to a percentage (by clamping the value between 0 and 1 and multiplying by 100).
4. Deployment with Flask
Flask Web App:
We integrated our pipelines into a Flask application. The app:

Accepts two image uploads from the user.
Processes each image through the preprocessing and feature extraction pipelines.
Calculates the cosine similarity between the extracted features.
Displays the similarity percentage along with the uploaded images for a user-friendly result.
Aesthetic Touches:
The final web page not only shows the similarity percentage but also plots the two images side by side for better visualization.

Error Resolution Journey
During this project, we encountered and solved several challenges:

Dependency Conflicts:
We resolved issues related to missing modules (like cv2 and tensorflow) and handled version conflicts by monkey patching (e.g., adding is_directory to Pillow and patching missing ONNX attributes in PyTorch).
Environment Setup:
We addressed Windows-specific issues (such as enabling long path support and adjusting PATH) to ensure our environment worked seamlessly.
Data Format Compatibility:
We ensured that our preprocessing pipeline output was correctly formatted (converted from float32 to uint8) for the FaceNet transformation.
Technologies and Libraries Used
FaceNet (InceptionResnetV1):
A deep learning model tailored for face recognition that provides highly discriminative embeddings.

MTCNN:
Used for face detection and alignment, ensuring that our feature extraction focuses on the face region.

Cosine Similarity:
A mathematical measure to determine similarity between two vectors. Ideal for L2-normalized embeddings from FaceNet.

Flask:
A lightweight web framework in Python used for deploying our model as a web application.

Other Libraries:
OpenCV (for image handling), NumPy (for numerical operations), and torchvision (for image transformations) played supporting roles.
