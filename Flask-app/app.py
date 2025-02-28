# app.py

# ======= Imports =======
import os
import csv
import glob
import io
import base64
import cv2
import numpy as np
import torch
from flask import Flask, request
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from mtcnn import MTCNN
import PIL._util
from PIL import Image

# ======= Monkey Patches =======
if not hasattr(PIL._util, 'is_directory'):
    PIL._util.is_directory = lambda path: os.path.isdir(path)

if not hasattr(torch._C._onnx, "PYTORCH_ONNX_CAFFE2_BUNDLE"):
    torch._C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE = None

# ======= PART 1: Dataset Parsing (if needed) =======
def parse_people_csv(people_csv_path):
    people = {}
    with open(people_csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            try:
                count = int(row[1])
            except ValueError:
                count = 0
            people[name] = count
    return people

def get_people_images(lfw_root, people_csv_path):
    people = parse_people_csv(people_csv_path)
    mapping = {}
    for person in people.keys():
        person_dir = os.path.join(lfw_root, person)
        if os.path.isdir(person_dir):
            image_paths = glob.glob(os.path.join(person_dir, "*.jpg"))
            mapping[person] = sorted(image_paths)
        else:
            print(f"Directory not found for person: {person}")
    return mapping

# ======= PART 2: Preprocessing Pipeline =======
class PreprocessingPipeline:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.detector = MTCNN()  # For face detection

    def load_image(self, image_input):
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to load image from path: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Input should be a file path or a numpy array.")
        return image

    def detect_and_align(self, image):
        results = self.detector.detect_faces(image)
        if results:
            face = results[0]
            x, y, width, height = face['box']
            x, y = abs(x), abs(y)
            face_image = image[y:y+height, x:x+width]
            return face_image
        else:
            raise ValueError("No face detected in the image.")

    def resize_image(self, image):
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def normalize_image(self, image):
        image = image.astype("float32") / 255.0
        return image

    def preprocess(self, image_input):
        image = self.load_image(image_input)
        face_image = self.detect_and_align(image)
        resized_image = self.resize_image(face_image)
        normalized_image = self.normalize_image(resized_image)
        return normalized_image

# ======= PART 3: Feature Extraction Pipeline (FaceNet) =======
class FeatureExtractionPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8) if x.dtype == np.float32 else x),
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    
    def extract_features(self, preprocessed_image):
        img_tensor = self.transform(preprocessed_image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        features = features.squeeze().cpu().numpy()
        return features

# ======= PART 4: Similarity Calculation =======
def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

def calculate_cosine_similarity(feature1, feature2):
    f1 = l2_normalize(feature1)
    f2 = l2_normalize(feature2)
    cosine_sim = np.dot(f1, f2)
    return cosine_sim

def convert_cosine_similarity_to_percentage(cosine_sim):
    cosine_sim = np.clip(cosine_sim, 0, 1)
    return cosine_sim * 100

# ======= Flask App Setup =======
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Initialize pipelines globally
preprocessing_pipeline = PreprocessingPipeline(target_size=(224, 224))
feature_extraction_pipeline = FeatureExtractionPipeline(device='cpu')

# ======= Flask Routes =======
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Please upload two images."
        file1 = request.files['image1']
        file2 = request.files['image2']
        if file1.filename == '' or file2.filename == '':
            return "No file selected."
        try:
            # Open uploaded files with PIL and convert to RGB
            image1 = Image.open(file1).convert('RGB')
            image2 = Image.open(file2).convert('RGB')
            # Convert PIL images to numpy arrays
            image1_np = np.array(image1)
            image2_np = np.array(image2)
            # Preprocess images
            preprocessed_img1 = preprocessing_pipeline.preprocess(image1_np)
            preprocessed_img2 = preprocessing_pipeline.preprocess(image2_np)
            # Extract features
            features1 = feature_extraction_pipeline.extract_features(preprocessed_img1)
            features2 = feature_extraction_pipeline.extract_features(preprocessed_img2)
            # Calculate cosine similarity and convert to percentage
            cosine_sim = calculate_cosine_similarity(features1, features2)
            similarity_percentage = convert_cosine_similarity_to_percentage(cosine_sim)
            
            # Convert images to base64 for display in HTML
            buffered1 = io.BytesIO()
            buffered2 = io.BytesIO()
            image1.save(buffered1, format="JPEG")
            image2.save(buffered2, format="JPEG")
            img_str1 = base64.b64encode(buffered1.getvalue()).decode("utf-8")
            img_str2 = base64.b64encode(buffered2.getvalue()).decode("utf-8")
            
            return f'''
            <!doctype html>
            <html>
            <head>
                <title>Face Similarity Result</title>
            </head>
            <body>
                <h1>Similarity Percentage: {similarity_percentage:.2f}%</h1>
                <div style="display: flex; flex-direction: row;">
                    <div style="margin-right: 20px;">
                        <img src="data:image/jpeg;base64,{img_str1}" alt="Image 1" width="300">
                    </div>
                    <div>
                        <img src="data:image/jpeg;base64,{img_str2}" alt="Image 2" width="300">
                    </div>
                </div>
                <br>
                <a href="/">Try again</a>
            </body>
            </html>
            '''
        except Exception as e:
            return f"Error processing images: {str(e)}"
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Face Similarity Checker</title>
    </head>
    <body>
        <h1>Upload Two Images</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="image1">Image 1:</label>
            <input type="file" name="image1"><br><br>
            <label for="image2">Image 2:</label>
            <input type="file" name="image2"><br><br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
