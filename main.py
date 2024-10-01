from ultralytics import YOLO
import cv2
import numpy as np
from util import *
from config import Config
from util import get_car, read_license_plate, search_license_plate, write_csv, search_license_plate_db, insert_data, create_database, VehicleTracker, trocr


results = {}
# Initialize
config = Config()
mot_tracker = VehicleTracker(maxDisappeared=50)
# load models
coco_model = YOLO(config.Yolo_model).to(config.Device)
license_plate_detector = YOLO(config.liscenplate_model).to(config.Device)


# load video
cap = cv2.VideoCapture(config.Capture_Device)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
if config.Video_Output:
    output = cv2.VideoWriter(config.Video_Output, cv2.VideoWriter_fourcc(*config.Video_format), fps, (frame_width, frame_height))

#choose OCR model EasyOCR or TrOCR here
config.trOCR=False
if trOCR == True:
    ocr_model = trocr()
else:
    ocr_model = None

# choose mode of running search or store 
mode = config.MODE
# choose storing method sqlite or csv
store_in = config.STORE_IN
search_csv = config.CSV_PATH

if store_in == "sqlite":
    db_path = config.DATABASE_PATH
    create_database(db_path)


vehicles = [2, 3, 5, 7, 67] #Yolo class ids for vehicles 67 for cellphones

# read frames

def detector():
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame,verbose=False,stream=False, device="cuda")[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if config.Video_Output:
                    framex= cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2])

            # track vehicles
            track_ids = mot_tracker.update(detections_)
            
            # detect license plates
            license_plates = license_plate_detector(frame,verbose=False,stream=False,device="cuda")[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                if config.Video_Output:
                    framex = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    output.write(framex)

                # assign license plate to car
                bbox, car_id = get_car(license_plate, track_ids)
                xcar1, ycar1, xcar2, ycar2 = bbox

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    
                    #OTHER PREPROCESSING STEPS CAN BE ADDED HERE threshold not used as it doenst work well
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, trOCR, ocr_model)

                    if license_plate_text is not None:
                        print(license_plate_text, license_plate_text_score)
                        
                        if store_in == "csv" and mode == "store":
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        if store_in == "csv" and mode == "search":
                            result = search_license_plate(search_csv, license_plate_text)
                            if result is None:
                                print("License plate not found")
                            else:
                                print(result)
                                
                        if store_in == "sqlite" and mode == "store":
                            results = {
                                frame_nmr: {                             # Frame number (as integer, stored as string here for dict key)
                                    car_id: {                       # Car ID (as integer, stored as string here for dict key)
                                        'car': {
                                            'bbox': [xcar1, ycar1, xcar2, ycar2]
                                        },
                                        'license_plate': {
                                            'bbox': [x1, y1, x2, y2],
                                            'bbox_score': score,
                                            'text': license_plate_text,
                                            'text_score': license_plate_text_score
                                        }   }   }   }
                            insert_data(db_path,results)
                            
                        if store_in == "sqlite" and mode == "search":
                            result = search_license_plate_db(db_path, license_plate_text)
                            if result is None:
                                print("License plate not found")
                            else:
                                print(result)


if __name__ == '__main__':
    detector()
    if config.Video_Output:
        output.release()

    # write results
    if store_in == "csv" and mode == "store":
        write_csv(results, './test.csv')
