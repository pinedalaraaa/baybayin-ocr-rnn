import xml.etree.ElementTree as ET
import cv2
import os


# Function to parse XML annotation file
def parse_xml_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

# Function to extract bounding boxes from XML annotation
def extract_bounding_boxes_and_labels(root):
    bounding_boxes = []
    labels = []
    for obj in root.findall('.//object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bounding_boxes.append((xmin, ymin, xmax, ymax))
        labels.append(label)
    return bounding_boxes, labels


# Function to crop characters from images
def crop_characters_from_images(image_path, bounding_boxes):
    image = cv2.imread(image_path)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
        cropped_image = image[ymin:ymax, xmin:xmax]
        cv2.imwrite(f'cropped_image_{i}.jpg', cropped_image)


# Function to start the parsing for a given XML file and cropping of original image
def commence_parsing(filepathXML, filepathImage, counter):

    # Parse XML annotation file
    xml_file = filepathXML
    image_path = filepathImage
    image = cv2.imread(image_path)

    root = parse_xml_annotation(xml_file)
    bounding_boxes, labels = extract_bounding_boxes_and_labels(root)

    if image is None:
        print(f"Error: Unable to load image from '{image_path}'")
    else:
        # Crop Characters Using Bounding Box Coordinates
        for xmin, ymin, xmax, ymax in bounding_boxes:
            try:
                cropped_image = image[ymin:ymax, xmin:xmax]
                # Save cropped image with label
                cv2.imwrite(f'C:/Users/Ray/Downloads/mindo_exer5/ocr-with-rnn-translation/Training Data/YA/cropped/Ya_{counter}.jpg', cropped_image)
            except Exception as e:
                print(f"Error processing bounding box {counter}: {e}")



# Path to the directory containing the annotated data in XML format
annotations = r"C:\Users\Ray\Downloads\mindo_exer5\ocr-with-rnn-translation\Training Data\YA\annotated"
original_images = r"C:\Users\Ray\Downloads\mindo_exer5\ocr-with-rnn-translation\Training Data\YA\original_images"


# Get a list of XML files and image files
xml_files = sorted(os.listdir(annotations))
image_files = sorted(os.listdir(original_images))


counter=0
# os.listdir(directory) gets all the folders listed in the given directory
for fileXML, fileImage in zip(xml_files, image_files):
    # os.path.join(a, b*) concatenates at least two strings into a pathname
    commence_parsing(os.path.join(annotations, fileXML), os.path.join(original_images, fileImage), counter)
    counter += 1


