# Baybayin OCR with RNN translation

## Project Overview
This project involves drawing four single-stroke Baybayin characters in the air (ba, ha, wa, ya), which are captured by a Raspberry Pi's camera and uses the Kanade-Lucas-Tomasi algorithm for tracking the optical flow. The tracked character data is then fed into a Tesseract model trained to recognize these characters from the optical flow. For translation, a Recurrent Neural Network (RNN) model is employed to convert the recognized Baybayin characters into Latin script, which the application then displays as an output.

## Demo

[![Watch the video](https://github.com/pinedalaraaa/baybayin-ocr-rnn/blob/main/demo/thumbnail.png?raw=true)](https://github.com/pinedalaraaa/baybayin-ocr-rnn/blob/main/demo/sample2.mp4?raw=true)

## Technologies Used

-   **Engine:** Tesseract-OCR 4.1.1 [Documentation](https://tesseract-ocr.github.io/tessdoc/)
-   **Libraries:** Keras 3.1, Tensorflow 2.16, OpenCV, Numpy, PyQt5
-   **Annotation Tool:** LabelImg

## Installing needed files for training Tesseract-OCR

1. Clone the following repositories:

    - [Tesstrain](https://github.com/tesseract-ocr/tesstrain)
    - [Tessdata_best](https://github.com/tesseract-ocr/tessdata_best)

2. Install [tesseract-ocr](https://tesseract-ocr.github.io/tessdoc/Compiling-%E2%80%93-GitInstallation.html)
3. Download additional libraries needed for training tesseract:
    - libicu-dev
    - libpango1.0-dev
    - libcairo2-dev
4. Add tesseract binary/executable to environment variables as TESSERACT_PATH
   e.g.
    ```
    /usr/bin/tesseract
    ```

## Training Tesseract-OCR

1. Get frame from optical flow videos that show the completed characters. Save the frames as images.
2. Run `generate_ground_truth.py` to rename the files in the training dataset and create the ground truth files for each image.
3. Navigate to the cloned tesstrain repository.
4. Add training samples and ground truth files for all characters to:

```
tesstrain/data/my-custom-model-ground-truth
```
note: the folder name should be [my-custom-model]-ground-truth

5. In the tesstrain repo, run the following command:

```
make training MODEL_NAME=my-custom-model START_MODEL=eng TESSDATA=path/to/tessdata_best
```

6. Wait for the training to finish. Expect that my_custom_model.traineddata will be generated.
7. Move `my_custom_model.traineddata` to tessdata directory.
   e.g.
    ```
    /usr/share/tesseract-ocr/4.00/tessdata
    ```
8. Run the following command to check if tesseract detects your new model:
    ```
    tesseract --list-langs
    ```
9. Run the following command to use your new model on validation dataset:
    ```
    tesseract --tessdata-dir [/path/to/tessdata-directory] [image_to_be_tested.png] stdout -l [my_custom_model] --psm [6]
    ```

## How to add a newly trained model to Tesseract

1. Clone this repository.

2. Navigate to the cloned repository: e.g. ~/Desktop/baybayin-ocr-rnn

2. Copy new model to tesstrain directory
sudo cp [source location of trained model] [destination location]
sudo cp ./baybayin.traineddata /usr/share/tesseract-ocr/4.00/tessdata

3. Check language models detected by Tesseract
tesseract --list-langs


## Training RNN model on translating Baybayin unicode blocks

1. Install Tensorflow version >= 2.16

2. Install Keras version >= 3.1

3. Run `rnn_training.py` program to train a neural network model using GRU RNN. Expect to receive a compressed file as output with .keras extension.

4. Integrate the model with the program in `final.py` as demonstrated in `test_image.py` program.

## How to run the application

1. Navigate to `original/` directory.
   
2. Run `final.py` using Python3.

3. The Region of Interest as seen by the camera must at least be a plain background.
