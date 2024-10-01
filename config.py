
class Config:
    # Mode for capture
    CAPTURE_MODE = 'video'  # 'video' or 'image'
    Capture_Device = 0  # 0 for default camera or use external camera /dev/cam1 or use IP-link for IPcamera
    
    # Database and path configuration
    DATABASE_PATH = 'license_plate_data.db'
    MODE = 'store'  # 'store' or 'search' 
    STORE_IN = 'sqlite'  # 'csv' or 'sqlite'
    Device = 'cpu'  # 'cpu' or 'cuda'
    CSV_PATH = 'license_plate.csv'
    
    # Video Output Configuration 
    Video_Output = False
    Video_Output_path = 'output.mp4'
    Video_format = 'mp4v'
    
    #path to models
    liscenplate_model = 'license_plate_model.pth'
    Yolo_model = 'yolov8s.pt'
    
    # Auto configured via easyOCR
    trOCR = False # If True, use trocr() else Using EasyOCR
    
    OCR_MODEL = 'ocr_model.pth'

    # RPI GPIO Configuration
    USE_RPI = False # Make it True if you are using Raspberry Pi and Need to control LED
    GPIO_PIN = 18  # GPIO pin for Red
    GPIO_PIN2 = 17  # GPIO pin for Green
    GPIO_PIN3 = 15  # GPIO pin for Yellow
    
    
    # To be Added
    DEBUG_MODE = False
    LOGGING_ENABLED = False
