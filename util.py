from difflib import get_close_matches
import re, csv
from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
import string
import easyocr
import sqlite3


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)
trOCR=False

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def search_license_plate(csv_file_path, license_plate_number):
    """
    Search for a license plate number in the CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.
        license_plate_number (str): The license plate number to search for.

    Returns:
        dict or None: The result containing the row details if found, or None if not found.
    """
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)

        # Iterate through each row in the CSV
        for row in csv_reader:
            # Check if the license plate number matches the search
            if row['license_number'] == license_plate_number:
                return {
                    'frame_nmr': row['frame_nmr'],
                    'car_id': row['car_id'],
                    'car_bbox': row['car_bbox'],
                    'license_plate_bbox': row['license_plate_bbox'],
                    'license_plate_bbox_score': row['license_plate_bbox_score'],
                    'license_number_score': row['license_number_score']
                }

    # Return None if the license plate number is not found
    return None


def create_database(db_path):
    """
    Create a SQLite database with a table for storing license plate detection data.
    Frame number and car ID are stored as integers.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create a table for storing license plate data with frame_nmr and car_id as integers
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS license_plate_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_nmr INTEGER,                  -- Store frame number as INTEGER
            car_id INTEGER,                     -- Store car ID as INTEGER
            car_bbox TEXT,
            license_plate_bbox TEXT,
            license_plate_bbox_score REAL,
            license_number TEXT,
            license_number_score REAL
        )
    ''')

    # Create an index on the license_number column to optimize searches
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_license_number ON license_plate_data (license_number)')

    conn.commit()
    conn.close()


def insert_data(db_path, results):
    """
    Insert the results into the database with frame_nmr and car_id stored as integers.

    Args:
        db_path (str): Path to the SQLite database file.
        results (dict): Dictionary containing the results to insert.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for frame_nmr in results.keys():
        for car_id in results[frame_nmr].keys():
            if 'car' in results[frame_nmr][car_id].keys() and \
               'license_plate' in results[frame_nmr][car_id].keys() and \
               'text' in results[frame_nmr][car_id]['license_plate'].keys():

                car_bbox = '[{} {} {} {}]'.format(
                    results[frame_nmr][car_id]['car']['bbox'][0],
                    results[frame_nmr][car_id]['car']['bbox'][1],
                    results[frame_nmr][car_id]['car']['bbox'][2],
                    results[frame_nmr][car_id]['car']['bbox'][3]
                )

                license_plate_bbox = '[{} {} {} {}]'.format(
                    results[frame_nmr][car_id]['license_plate']['bbox'][0],
                    results[frame_nmr][car_id]['license_plate']['bbox'][1],
                    results[frame_nmr][car_id]['license_plate']['bbox'][2],
                    results[frame_nmr][car_id]['license_plate']['bbox'][3]
                )

                # Insert data into the database, ensuring frame_nmr and car_id are integers
                cursor.execute('''
                    INSERT INTO license_plate_data (frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(frame_nmr),                              # Convert frame_nmr to integer
                    # Convert car_id to integer
                    int(car_id),
                    car_bbox,
                    license_plate_bbox,
                    results[frame_nmr][car_id]['license_plate']['bbox_score'],
                    results[frame_nmr][car_id]['license_plate']['text'],
                    results[frame_nmr][car_id]['license_plate']['text_score']
                ))

    conn.commit()
    conn.close()


def search_license_plate_db(db_path, license_plate_number):
    """
    Search for a license plate number in the SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        license_plate_number (str): The license plate number to search for.

    Returns:
        dict or None: The result containing the row details if found, or None if not found.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score
        FROM license_plate_data
        WHERE license_number = ?
    ''', (license_plate_number,))

    row = cursor.fetchone()

    conn.close()

    if row:
        return {
            # frame_nmr is returned as integer
            'frame_nmr': row[0],
            # car_id is returned as integer
            'car_id': row[1],
            'car_bbox': row[2],
            'license_plate_bbox': row[3],
            'license_plate_bbox_score': row[4],
            'license_number': row[5],
            'license_number_score': row[6]
        }

    return None


# Dictionary of valid state codes
valid_states = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "MH": "Maharashtra",
    "MP": "Madhya Pradesh",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
}

#visual correction map for non-numeric characters and common OCR errors
visual_correction_map = {
    'O': '0', 'D': '0', 'Q': '0', 'o': '0', 'U': '0',
    'I': '1', 'L': '1', 'T': '1', '|': '1', '/': '1', '\\': '1', '!': '1', 'i': '1',
    'Z': '2', '?': '2', 'z': '2',
    'S': '5', '$': '5', 's': '5',
    'B': '8', 'b': '8', 'g': '9', 'G': '6',
    '@': '0', '%': '0', '&': '8'
}

remove_chars = str.maketrans ({' ': '',  # Removes any spaces mistakenly inserted
    '-': '', '_': '', '.': '', ',': ''  # Remove common separators or noise
})


# Regex pattern for Indian license plate
plate_pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"

# Function to correct visually similar characters in unique ID


def correct_unique_id(unique_id):
    corrected_id = ""
    for char in unique_id:
        if char.isdigit():
            corrected_id += char
        elif char in visual_correction_map:
            corrected_id += visual_correction_map[char]
        else:
            return "Invalid unique ID"  # If a completely unrecognizable character
    return corrected_id



# Function to validate and correct license plate
def validate_and_correct_plate(plate):
    # First, check the overall structure using regex
    if not re.match(plate_pattern, plate):
        return "Invalid plate structure"

    # Extract the components
    state = plate[:2]
    city_code = plate[2:4]
    series_identifier = plate[4:6]
    unique_id = plate[6:]

    # Check and correct the state
    if state not in valid_states:
        # Get closest match for the state code using fuzzy matching
        closest_state = get_close_matches(
            state, valid_states.keys(), n=1, cutoff=0.7)
        if closest_state:
            state = closest_state[0]
        else:
            return "Invalid state code"

    # City code must be numeric
    if not city_code.isdigit():
        return "Invalid city code"

    # Series identifier must be alphabetic
    if not series_identifier.isalpha():
        return "Invalid series identifier"

    # Correct and validate unique ID
    corrected_unique_id = correct_unique_id(unique_id)
    if corrected_unique_id == "Invalid unique ID":
        return corrected_unique_id

    # Reconstruct the corrected license plate
    corrected_plate = f"{state}{city_code}{series_identifier}{corrected_unique_id}"

    return corrected_plate


# Example usage
plate = "RJ02AB12S4"
corrected_plate = validate_and_correct_plate(plate)
print(corrected_plate)  # Should correct "S" to "5" in unique ID



def read_license_plate(license_plate_crop, trOCR=False, ocr_model=None):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    if trOCR:
        detections = ocr_model.readtext(license_plate_crop)
        if len(detections)>7:
            return detections, 1
    else:
        detections = reader.readtext(license_plate_crop)

        for detection in detections:
            bbox, text, score = detection
            plate_number = text.translate(remove_chars)
            
            plate_number = validate_and_correct_plate(plate_number)
            if "Invalid" not in plate_number:
                return plate_number, score
                # return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for key,value in vehicle_track_ids.items():
        xcar1, ycar1, xcar2, ycar2 = value

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = key
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx], car_indx

    return [-1, -1, -1, -1], -1


class VehicleTracker:
    def __init__(self, maxDisappeared=50):
        # Initialize the next unique object ID, the tracked vehicles, and disappeared count.
        self.nextObjectID = 0
        self.vehicles = OrderedDict()
        self.vehiclesbox = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, detection):
        # Register a new vehicle with a new ID.
        self.vehicles[self.nextObjectID] = centroid
        self.vehiclesbox[self.nextObjectID] = detection
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove the vehicle from tracking.
        del self.vehicles[objectID]
        del self.vehiclesbox[objectID]
        del self.disappeared[objectID]

    def update(self, detections):
        # If no detections, mark vehicles as disappeared.
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.vehiclesbox

        # Convert detection boxes to centroids.
        inputCentroids = np.zeros((len(detections), 2), dtype="int")
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection  # Bounding box (x1, y1, x2, y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no existing vehicles, register all new centroids.
        if len(self.vehicles) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i],detections[i])
        else:
            # Existing vehicle IDs and their centroids.
            objectIDs = list(self.vehicles.keys())
            objectCentroids = list(self.vehicles.values())

            # Compute distances between existing centroids and new detections.
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Find the best matches between current and new centroids.
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Track which rows/cols have been examined.
            usedRows = set()
            usedCols = set()

            # Assign centroids to vehicle IDs based on the smallest distance.
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.vehicles[objectID] = inputCentroids[col]
                self.vehiclesbox[objectID] = detections[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Handle unmatched rows (vehicles that disappeared).
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Mark vehicles that disappeared.
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # Register new centroids as new vehicles.
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col],detections[col])

        return self.vehiclesbox


class trocr:
    def __init__(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        self.model_name = "microsoft/trocr-base-printed"
        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.PIL = Image

    def readtext(self, image):
        image = self.PIL.fromarray(image)
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").pixel_values
        outputs = self.model.generate(inputs)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded


