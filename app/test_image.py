import os
import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



### Tests the pre-trained models pero only for images

# load pre-trained models
tesseract_model_path = '../../../../../usr/share/tesseract-ocr/4.00/tessdata/bybyn.traineddata'
rnn_model_path = '../models/baybayin_to_english_translation_model.keras'

# tesseract config options for improved character recognition
tesseract_config = r'--oem 3 --psm 6'

# load Tesseract OCR model
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# load RNN model
rnn_model = load_model(rnn_model_path)

# dictionaries to map characters to indices
char_to_index = {'ᜊ': 0, 'ᜑ': 1, 'ᜏ': 2, 'ᜌ': 3}
index_to_latscr = {0: 'ba', 1: 'ha', 2: 'wa', 3: 'ya'}


# function to preprocess the image and ensure input shape matches model expectations
def preprocess_image(image_path, target_shape):
    # load the image
    image = cv2.imread(image_path)
    
    # resize the image to the target shape
    image = cv2.resize(image, target_shape)
    
    # convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def recognize_and_translate_character(image_path):
    # read the PNG image using OpenCV
    image = cv2.imread(image_path)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # perform OCR to recognize the characters
    recognized_character = pytesseract.image_to_string(gray_image, lang='bybyn', config=tesseract_config)
    recognized_character = recognized_character.strip()

    # convert recognized character to its corresponding index
    input_seq = np.array([[char_to_index[recognized_character]]])

    # predict probabilities for each class using the RNN model
    predicted_probabilities = rnn_model.predict(input_seq)

    # get the index of the class with the highest probability
    predicted_class_index = np.argmax(predicted_probabilities)

    # perform translation using the RNN model
    translation = index_to_latscr[predicted_class_index]

    return recognized_character, translation


def commence(img_path):
    recognized_characters, translated_characters = recognize_and_translate_character(img_path)
    
    print(img_path)
    print("Recognized characters:", recognized_characters)
    print("Translated characters:", translated_characters)
    print()


# Example usage
dir = "../validation-data"

for i in os.listdir(dir):
    if i.endswith(('.png')):
        commence(os.path.join(dir, i))