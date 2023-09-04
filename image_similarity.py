from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_image_similarity(image1_path, image2_path):
    base_model = VGG16(weights='imagenet', include_top=False)
    
    def preprocess_image(image_path):
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    
    features1 = base_model.predict(image1)
    features2 = base_model.predict(image2)
    
    return cosine_similarity(features1.flatten().reshape(1, -1), features2.flatten().reshape(1, -1))[0][0]
