import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk


class OpticalFlowOCR:
    def __init__(self):
        # load pre-trained models
        self.rnn_model_path = '../models/baybayin_to_english_translation_model.keras'

        # trained data
        self.tesseract_model_path = '/usr/share/tesseract-ocr/4.00/tessdata/bybyn.traineddata'

        # load Tesseract OCR model
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        self.tesseract_config = r'--oem 3 --psm 6'

        # load RNN model
        self.rnn_model = load_model(self.rnn_model_path)

        # define a dictionary to map characters to indices and indices to Latin script
        self.char_to_index = {'ᜊ': 0, 'ᜑ': 1, 'ᜏ': 2, 'ᜌ': 3}
        self.index_to_latscr = {0: 'ba', 1: 'ha', 2: 'wa', 3: 'ya'}

        # initialize video capture
        self.cap = cv2.VideoCapture(0)
        _, self.frame = self.cap.read()
        self.old_gray = None

        # define ROI coordinates
        self.roi_coords = (100, 100, 200, 200)

        # parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(50, 50), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # variables for KLT optical flow
        self.point_selected = False
        self.point = ()
        self.old_points = np.array([[]])

        # setup a mask frame for drawing optical flow
        self.mask = np.zeros_like(self.frame)

        # variables to track optical flow density stabilization
        self.prev_density = 0
        self.density_change_threshold = 10
        self.stable_frames = 0
        self.required_stable_frame_count = 50
        self.continue_tracking = True
    
    # method to check if camera stream is working properly
    def check_camera(self):
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")
    
    # method to determine starting point for optical flow
    def select_point(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)
            self.point_selected = True
            self.old_points = np.array([[x, y]], dtype=np.float32)

    # method to use OCR and RNN models on optical flow
    def recognize_and_translate_character(self, frame):
        # convert the frame containing the optical flow to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # perform OCR to recognize the characters
        recognized_character = pytesseract.image_to_string(gray_image, lang='bybyn', config=self.tesseract_config).strip()
        recognized_character = recognized_character.strip()
        
        # check if the recognized character is in the mapping dictionary
        if recognized_character in self.char_to_index:
            # prepare input for RNN
            input_seq = np.array([[self.char_to_index[recognized_character]]])
            
            # predict with RNN
            predicted_probabilities = self.rnn_model.predict(input_seq)
            predicted_class_index = np.argmax(predicted_probabilities)
            translation = self.index_to_latscr[predicted_class_index]
            return recognized_character, translation
        
        # if character is not recognized, return None
        return None, None

    # method to flip the frame horizontally to fix the mirror view issue
    def flip_frame(self, frame):
        return cv2.flip(frame, 1)
    
    # method to process frame by frame for Optical Flow
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to read a frame from the camera.")
            return False
        
        # check if camera should be mirrored or not
        if self.is_flip.get():
            frame = self.flip_frame(frame)
        
        # convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.old_gray is None:
            self.old_gray = gray_frame

        # draw ROI rectangle
        roi_x, roi_y, roi_width, roi_height = self.roi_coords
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
        roi = gray_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # calculate optical flow
        if self.point_selected:
            cv2.circle(frame, self.point, 5, (0, 0, 255), 3)

            new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, gray_frame, self.old_points, None, **self.lk_params)
            self.old_gray = gray_frame.copy()
            self.old_points = new_points.copy()

            x, y = new_points.astype(int).ravel()
            
            cv2.circle(self.mask, (int(new_points.ravel()[0]), int(new_points.ravel()[1])), 2, (0, 255, 0), 2)
            cv2.imshow("Masked ROI", self.mask)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # check if optical flow has stopped
            self.update_density()
        
        # show the two frames
        cv2.imshow("Frame", frame)
        cv2.imshow("ROI Frame", roi)

        # continue unless ESC key is pressed
        key = cv2.waitKey(1)
        return key != 27

    # method to measure the density of optical flow in the ROI
    def update_density(self):
        current_density = np.count_nonzero(self.mask)
        density_change = abs(current_density - self.prev_density)

        # calculate counter for stable frames
        if density_change < self.density_change_threshold:
            self.stable_frames += 1
        else:
            self.stable_frames = 0

        # recognize and translate character when optical flow stabilizes
        if self.stable_frames > self.required_stable_frame_count:
            print("Optical flow stabilized, initiating OCR...")
            recognized_character, translation = self.recognize_and_translate_character(self.mask)
            
            if recognized_character:
                print(f"Recognized: {recognized_character} - Translated: {translation}")
                self.trans_text.configure(state="normal")
                self.trans_text.delete(0, tk.END)  # Clear existing text
                self.trans_text.insert(0, translation)
                self.trans_text.configure(state="readonly")
            else:
                print("Character not recognized.")
                self.trans_text.configure(state="normal")
                self.trans_text.delete(0, tk.END)  # Clear existing text
                self.trans_text.insert(0, "N/A")
                self.trans_text.configure(state="readonly")
            
            # clear the mask after processing
            self.mask = np.zeros_like(self.frame) 
            self.point_selected = False

            # destroy masked roi window
            cv2.destroyWindow("Masked ROI")

            # reset counter after OCR
            self.stable_frames = 0

        # update density for next frame comparison
        self.prev_density = current_density

    # method to close the windows
    def close_win(self):
        self.isClose = False
        self.cap.release()
        cv2.destroyWindow("Frame")
        cv2.destroyWindow("ROI Frame")

    # method to run the optical flow system
    def run(self, root, is_flip, trans_text):
        self.check_camera()
        self.is_flip = is_flip
        self.trans_text = trans_text

        # setup mouse callback
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.select_point)

        while self.continue_tracking:
            if not self.process_frame():
                break
            root.update()

        self.cap.release()
        cv2.destroyAllWindows()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Baybayin Translation")
        self.disply_width = 640
        self.display_height = 480
        self.ocr_system = None
        self.is_flip = tk.IntVar()

        self.init_ui()

    # initialize UI layout
    def init_ui(self):
        output_label = tk.Label(self.root, text="Output:")
        output_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.translated_text = tk.Entry(self.root, width=20, state='readonly')
        self.translated_text.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # radio group for mirror view
        radio_group_label = tk.Label(self.root, text="Mirror View:")
        radio_group_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.radio_yes = tk.Radiobutton(self.root, text="Yes", variable=self.is_flip, value=1)
        self.radio_yes.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.radio_no = tk.Radiobutton(self.root, text="No", variable=self.is_flip, value=0)
        self.radio_no.select()
        self.radio_no.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # buttons
        self.btn_open_cam = tk.Button(self.root, text='Open Camera', command=self.open_cam)
        self.btn_open_cam.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.btn_close_cam = tk.Button(self.root, text='Close Camera', command=self.close_cam)
        self.btn_close_cam.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.btn_exit = tk.Button(self.root, text='Exit', command=self.close_application)
        self.btn_exit.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

    # method to close the app
    def close_application(self):
        # check if the camera is running
        if self.ocr_system:
            self.close_cam()

        self.root.quit()

    # method to open the camera
    def open_cam(self):
        # if OpticalFlowOCR system is not yet running, instantiate it
        if not self.ocr_system:
            self.ocr_system = OpticalFlowOCR()

        print(self.is_flip)

        # run OpticalFlowOCR system
        self.ocr_system.run(self.root, self.is_flip, self.translated_text)
        # delete object instance
        del self.ocr_system
        # set ocr_system attribute to None after deleting
        self.ocr_system = None

    # method to close the camera
    def close_cam(self):
        # close the windows and delete the instantiated OpticalFlowOCR object
        self.ocr_system.close_win()
        # celete object instance
        del self.ocr_system
        # set ocr_system attribute to None after deleting
        self.ocr_system = None

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()