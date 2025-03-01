# Compute-Similarity-App

### 1. Data Parsing and Preprocessing
* Data Parsing:
helper functions to parse metadata from the lfw dataset. These functions map person names to their corresponding image paths, those functions were used in the notebook for the lfw dataset, i included them in the app if needed.

* Preprocessing Pipeline:
an implementation of PreprocessingPipeline class that:

  * Loads Images: Reads images from a file path or NumPy array.
  * Face Detection & Alignment: Uses MTCNN to detect faces and extract the face region.
  * Resizing and Normalization: Resizes the detected face to a consistent target size (e.g., 224×224) and normalizes pixel values to the [0,1] range.

### 2. Feature Extraction Using FaceNet
* Why FaceNet?
 FaceNet is a specialized model for face recognition that maps images to a 512-dimensional embedding space. It is designed so that embeddings of the same person are close together while embeddings of different people are far apart.

* Implementation:
built a FeatureExtractionPipeline class that uses the FaceNet model (InceptionResnetV1 from the facenet-pytorch library) to generate feature embeddings.

 * The pipeline includes a transformation that converts the preprocessed image to the expected format (resizing to 160×160, converting to tensor, and normalizing to the range [-1, 1]).
 * The model outputs a 512-dimensional feature vector for each image.
 * ##### Note: i resized twice you can guess because the preprocessing code was made for ResNet then i moved to FaceNet, i didn't change the code cuz it performed well and there wasn't a need, i will change it later!

### 3. Similarity Calculation Using Cosine Similarity
* Why Cosine Similarity?
Cosine similarity is a measure of the angle between two vectors. When embeddings are L2-normalized (as is standard with FaceNet), cosine similarity is particularly effective:
A value close to 1 indicates very similar (or nearly identical) embeddings.
A lower value indicates dissimilar faces.
* Implementation:
implemented helper functions to:
 * L2-normalize the feature vectors.
 * Calculate cosine similarity between two embeddings.
 * Convert the cosine similarity to a percentage (by clamping the value between 0 and 1 and multiplying by 100).

### 4. Deployment with Flask
* Flask Web App:
integrated the pipelines into a Flask application. The app:

 * Accepts two image uploads from the user.
 * Processes each image through the preprocessing and feature extraction pipelines.
 * Calculates the cosine similarity between the extracted features.
 * Displays the similarity percentage along with the uploaded images for a user-friendly result.

### Error Resolution
During this project, i encountered some errors that might happen with somebody:

handled version conflicts by monkey patching (adding is_directory to Pillow and patching missing ONNX attributes in PyTorch).
addressed Windows-specific issues (such as enabling long path support and adjusting PATH) to ensure the environment worked seamlessly.
ensured that the preprocessing pipeline output was correctly formatted (converted from float32 to uint8) for the FaceNet transformation.

#### Technologies and Libraries Used
* FaceNet (InceptionResnetV1):
A deep learning model tailored for face recognition that provides highly discriminative embeddings.

* MTCNN:
Used for face detection and alignment, ensuring that the feature extraction focuses on the face region.

* Cosine Similarity:
A mathematical measure to determine similarity between two vectors. Ideal for L2-normalized embeddings from FaceNet.

* Flask:
A lightweight web framework in Python used for deploying the model.

* Other Libraries:
OpenCV (for image handling), NumPy (for numerical operations), and torchvision (for image transformations).
