# Auto Liscence detector and Finder

```
Nplate/
├── visualize.py         # Script to visualize bounding boxes
├── add_missing_data.py  # Script to interpolate missing bounding boxes
├── util.py              # Utility functions for license plate processing
├── requirements.txt      # Project dependencies
├── config.py             # Configuration settings
└── main.py               # Main script for ALPR system
```

## Features

- **Vehicle and License Plate Detection:** Detects vehicles and their license plates using YOLOv8.
- **Optical Character Recognition (OCR):** Recognizes license plate characters using EasyOCR or TrOCR.
- **Vehicle Tracking:** Tracks multiple vehicles across frames using a centroid-based tracker.
- **Data Storage:** Stores detected license plates and associated data in a CSV file or SQLite database.
- **License Plate Search:** Searches for a specific license plate within the stored data.
- **Raspberry Pi Support:** Can be deployed on a Raspberry Pi with GPIO control for LEDs.
- **Visual Similarity Correction:** Corrects common OCR errors based on visually similar characters.
- **License Plate Validation:** Validates and corrects Indian license plate formats.
  
## Installation

### Prerequisites

- **Raspberry Pi (Optional):** If deploying on a Raspberry Pi, ensure you have a Raspberry Pi with Raspbian OS installed.

### Dependencies
#### System Dependencies:

```bash
sudo apt-get update
sudo apt-get install python3-pip python3-dev python-rpi.gpio
```
#### Install Pip Dependencies
```
pip3 install -r requirements.txt
```



## Configuration

The system's behavior is controlled by the `config.py` file. You can modify the following settings:

- **`CAPTURE_MODE`:** Sets the capture mode ('video', 'image', 'ipcam', or 'file').
- **`Capture_Device`:** Specifies the video source (camera index, video file path, or IP camera URL).
- **`Display`:** Enables or disables displaying the output frame.
- **`DATABASE_PATH`:** Sets the path to the SQLite database file.
- **`MODE`:** Sets the operating mode ('store' or 'search').
- **`STORE_IN`:** Sets the storage method ('csv' or 'sqlite').
- **`Device`:** Specifies the device to run the models on ('cpu' or 'cuda').
- **`CSV_PATH`:** Sets the path to the CSV file.
- **`Video_Output`:** Enables or disables video output.
- **`Video_Output_path`:** Sets the path to the output video file.
- **`Video_format`:** Specifies the video format ('mp4v', 'XVID', etc.).
- **`liscenplate_model`:** Sets the path to the license plate detection model.
- **`Yolo_model`:** Sets the path to the YOLOv8 vehicle detection model.
- **`trOCR`:** Enables or disables the use of TrOCR for OCR.
- **`OCR_MODEL`:** Sets the path to the TrOCR model (if enabled).
- **`RPI`:** Enables or disables Raspberry Pi GPIO control for LEDs.
- **`GPIO_PIN`, `GPIO_PIN2`, `GPIO_PIN3`:** Set the GPIO pins for the LEDs (if RPI is enabled). 
- **`DEBUG_MODE`:** Non Functiona;
- **`LOGGING_ENABLED`:** Enable to print Detections and Other Details.

## Usage

### Storing License Plate Data

1.  **Configure `config.py`:**
    -   Set `MODE` to 'store'.
    -   Set `STORE_IN` to either 'csv' or 'sqlite'.
    -   If using 'sqlite', specify the `DATABASE_PATH`.
    -   If using 'csv', specify the `CSV_PATH`.
    -   Configure other settings as needed (capture mode, video source, etc.).

2.  **Run the script:**

```bash
python3 main.py
```



### Searching for a License Plate

1.  **Configure `config.py`:**
    -   Set `MODE` to 'search'.
    -   Set `STORE_IN` to either 'csv' or 'sqlite'.
    -   If using 'sqlite', specify the `DATABASE_PATH`.
    -   If using 'csv', specify the `CSV_PATH`.
    -   Configure other settings as needed (capture mode, video source, etc.).

2.  **Run the script:**

```bash
python3 main.py
```

The script will detect vehicles and their license plates, extract the license plate text using OCR, and search or store it in the chosen storage method. If the license plate is found, it will output the associated information (frame number, car ID, bounding boxes). 
Currently both search and store are not running simultaneous but can be added.







## Missing Features

-   **Improved License Plate Detection:** The current system relies on a generic object detection model and OCR model for license plate detection. A dedicated license plate detection model could improve accuracy.
-   **Enhanced OCR Accuracy:** The OCR accuracy can be further improved by exploring different OCR engines or fine-tuning existing models.
-   **Performance Optimization:** The system's performance can be optimized further
-   **Real-Time Processing:** Optimize the system for real-time performance, especially on resource-constrained devices like the Raspberry Pi. Consider using techniques like model quantization and asynchronous processing.
-   **Multiple License Plate Formats:** Add support for different license plate formats from various countries or regions. Limited to only 10 digit Indian format


