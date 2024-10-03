
class Config:
    # Mode for capture
    CAPTURE_MODE = 'video'  # 'video' or 'image' or 'ipcam' or 'file'
    # 0 for default camera or use external camera /dev/cam1 or use IP-link for IPcamera
    #Example
    # Capture_Device = "rtmp://10.2.33.223/live/stream1"
    # Capture_Device = "/home/utka/DRONE/Drone_testing/number_plate/num3.mp4"
    Capture_Device = 0
    Display = True  # Display the input frame Not for output and it will be laggy
    
    # Database and path configuration
    DATABASE_PATH = 'license_plate.db'
    MODE = 'search'  # 'store' or 'search' 
    STORE_IN = 'sqlite'  # 'csv' or 'sqlite'  
    Device = 'cpu'  # 'cpu' or 'cuda'
    CSV_PATH = 'license_plate.csv'
    
    # Video Output Configuration 
    Video_Output = False
    Video_Output_path = 'output.mp4'
    Video_format = 'mp4v'
    
    #path to models
    liscenplate_model = 'license_plate_detector.pt'
    Yolo_model = 'yolov8n.pt'
    
    # Auto configured via easyOCR
    trOCR = False # If True, use trocr() else Using EasyOCR
    #Not efficient for real-time processing
    OCR_MODEL = 'ocr_model.pt'

    # RPI GPIO Configuration
    RPI = False # Make it True if you are using Raspberry Pi and Need to control LED
    GPIO_PIN = 18  # GPIO pin for Orange processing
    GPIO_PIN2 = 17  # GPIO pin for Green found 
    GPIO_PIN3 = 15  # GPIO pin for Red not found
    

    # To be Added
    DEBUG_MODE = False
    LOGGING_ENABLED = True
