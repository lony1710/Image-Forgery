import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os
import time
import numpy as np
from PIL.ExifTags import TAGS

# Load the trained model
model = load_model('image_forgery_model.h5')

# Function to convert an image to ELA (Error Level Analysis)
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_file_name.jpg'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# Function to preprocess image
def prepare_image(image_path, image_size=(128, 128)):
    ela_image = convert_to_ela_image(image_path, 90)
    ela_image = ela_image.resize(image_size)
    ela_array = np.array(ela_image).astype('float32') / 255.0  # Normalize
    ela_array = np.expand_dims(ela_array, axis=0)  # Reshape for model
    return ela_image, ela_array

# Function to predict and highlight fake regions
def detect_forgery(image_path):
    # Preprocess image
    ela_image, image_array = prepare_image(image_path)

    # Predict using the model
    prediction = model.predict(image_array)[0]
    class_label = "Real" if np.argmax(prediction) == 1 else "Fake"

    # Print result
    # Get the confidence for the predicted class
    confidence = np.max(prediction)
    print(f"Prediction: {class_label} (Confidence: {confidence*100:.2f}%)")

    
    # If fake, generate a heatmap to highlight fake regions
    if class_label == "Fake":
        highlight_fake_regions(image_path)

    # Show images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(ela_image, cmap="gray")
    plt.title("ELA Image")
    
    plt.show()

# Function to highlight fake areas using heatmaps
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_file_name.jpg'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# Function to preprocess image
def prepare_image(image_path, image_size=(128, 128)):
    ela_image = convert_to_ela_image(image_path, 90)
    ela_image = ela_image.resize(image_size)
    ela_array = np.array(ela_image).astype('float32') / 255.0  # Normalize
    ela_array = np.expand_dims(ela_array, axis=0)  # Reshape for model
    return ela_image, ela_array

# Function to highlight fake areas with a proper heatmap
def highlight_fake_regions(image_path):
    # Load and preprocess the image
    ela_image, _ = prepare_image(image_path)

    # Convert ELA image to grayscale
    ela_gray = np.array(ela_image.convert('L'))

    # Resize grayscale image to match the original image size
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ela_gray = cv2.resize(ela_gray, (original.shape[1], original.shape[0]))

    # Apply Gaussian blur to smooth noise
    ela_blurred = cv2.GaussianBlur(ela_gray, (5, 5), 0)

    # Use Sobel edge detection to find manipulated regions
    sobelx = cv2.Sobel(ela_blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(ela_blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the edges and convert to 8-bit grayscale
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap to get a heatmap effect (reddish-yellow)
    heatmap = cv2.applyColorMap(sobel_edges, cv2.COLORMAP_JET)

    # Blend heatmap with original image
    overlaid = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    # Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlaid)
    plt.title("Fake Regions Highlighted")
    plt.axis("off")

    plt.show()



# Function to get image metadata (size, format, mode, file size)
def get_image_metadata(image_path):
    img = Image.open(image_path)
    size = img.size  # width, height
    format = img.format
    mode = img.mode  # Color space (RGB, L, etc.)
    file_size = os.path.getsize(image_path)
    print(f"Image Size: {size} (width, height)")
    print(f"Format: {format}")
    print(f"Mode: {mode}")
    print(f"File Size: {file_size / 1024:.2f} KB")
    return size, format, mode, file_size


# Function to get image creation date from EXIF metadata
def get_image_creation_date(image_path):
    """
    Extract the image creation date from EXIF metadata.
    """
    img = Image.open(image_path)
    exif_data = img._getexif()
    
    # If EXIF data exists and contains DateTime, return the DateTime
    if exif_data:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'DateTime':
                return value
    return "No creation date available"



# Function to get camera information (make, model) from EXIF data
def get_camera_info(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    
    camera_info = {}
    if exif_data:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'Make' or tag_name == 'Model':
                camera_info[tag_name] = value
                
    return camera_info


# Function to get location info (GPS)
def get_location_info(image_path):
    """
    Extract GPS data from the EXIF metadata and return formatted DMS coordinates.
    If no GPS data is found, return 'N/A' for all fields.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        gps_info = {}
        if exif_data and isinstance(exif_data, dict):  # Ensure exif_data is a dictionary
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                # Look for GPS info in EXIF metadata
                if tag_name == 'GPSInfo' and isinstance(value, dict):  # Ensure GPSInfo is a dictionary
                    gps_info = value
                    break  # Once GPS data is found, exit the loop

        if gps_info:
            # GPSInfo is a dictionary, extract Latitude, Longitude, and Timestamp
            latitude = gps_info.get(2, 'N/A')  # Latitude is at tag 2
            longitude = gps_info.get(4, 'N/A')  # Longitude is at tag 4
            timestamp = gps_info.get(29, 'N/A')  # Timestamp is at tag 29

            # If latitude and longitude are tuples, convert them to decimal degrees
            if latitude != 'N/A' and isinstance(latitude, tuple):
                latitude_decimal = latitude[0] + (latitude[1] / 60.0) + (latitude[2] / 3600.0)
                latitude_dms = decimal_to_dms(latitude_decimal, is_latitude=True)
            else:
                latitude_dms = 'N/A'
            
            if longitude != 'N/A' and isinstance(longitude, tuple):
                longitude_decimal = longitude[0] + (longitude[1] / 60.0) + (longitude[2] / 3600.0)
                longitude_dms = decimal_to_dms(longitude_decimal, is_latitude=False)
            else:
                longitude_dms = 'N/A'
            
            return {
                'Latitude': latitude_dms,
                'Longitude': longitude_dms,
                'Timestamp': timestamp
            }
        else:
            return {
                'Latitude': 'N/A',
                'Longitude': 'N/A',
                'Timestamp': 'N/A'
            }
    except Exception as e:
        print(f"Error extracting location info: {e}")
        return {
            'Latitude': 'N/A',
            'Longitude': 'N/A',
            'Timestamp': 'N/A'
        }


def decimal_to_dms(degrees, is_latitude=True):
    """
    Convert decimal degrees to degrees, minutes, and seconds (DMS) format.
    - is_latitude indicates if the degrees are for latitude (North/South)
    """
    # Get absolute value of degrees
    degrees_abs = abs(degrees)
    d = int(degrees_abs)  # Degrees
    m = int((degrees_abs - d) * 60)  # Minutes
    s = (degrees_abs - d - m / 60) * 3600  # Seconds
    
    # Determine the direction (N/S for latitude, E/W for longitude)
    if is_latitude:
        direction = 'N' if degrees >= 0 else 'S'
    else:
        direction = 'E' if degrees >= 0 else 'W'
    
    # Return formatted string
    return f"{d}° {m}' {s:.2f}\" {direction}"
# Function to get file details (including creation and last modified times)
def get_file_details(image_path):
    file_stats = os.stat(image_path)
    
    # Get file creation and last modified timestamps
    creation_time = time.ctime(file_stats.st_ctime)  # Creation time
    last_modified_time = time.ctime(file_stats.st_mtime)  # Last modified time
    
    return creation_time, last_modified_time


# Helper function to clean strings and remove null characters
def clean_string(s):
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    return s.strip('\x00')


def extract_image_details(image_path):
    # Preprocess image (convert to ELA and resize)
    ela_image, image_array = prepare_image(image_path)  # image_array is now ready for prediction
    
    # Get image metadata
    size, format, mode, file_size = get_image_metadata(image_path)
    
    # Get prediction score (forged score)
    prediction = model.predict(image_array)[0]  # Using the preprocessed image_array
    confidence = np.max(prediction)
    class_label = "Real" if np.argmax(prediction) == 1 else "Fake"
    prediction_result = f"Prediction: {class_label} : {confidence * 100:.2f}%"
    
    # Get camera info
    camera_info = get_camera_info(image_path)
     # Decode only if the value is in bytes
    camera_info_cleaned = {
        'Make': clean_string(camera_info.get('Make', 'N/A')),
        'Model': clean_string(camera_info.get('Model', 'N/A'))
    }
    

    # Get location info (GPS)
    location_info = get_location_info(image_path)
    location_info_cleaned = {}
    if location_info:
        location_info_cleaned['Latitude'] = location_info.get(2, ('N/A', 'N/A'))[0]
        location_info_cleaned['Longitude'] = location_info.get(4, ('N/A', 'N/A'))[0]
        location_info_cleaned['Timestamp'] = location_info.get(29, 'N/A')
    
    # Get image creation date from EXIF
    image_creation_date = get_image_creation_date(image_path)
    
    # Get file details (creation/edit history)
    creation_time, last_modified_time = get_file_details(image_path)
    
    # Print all the details line by line
    print(f"Prediction: {prediction_result}")
    print(f"Image Size: {size[0]}x{size[1]} (width x height)")
    print(f"Format: {format}")
    print(f"Mode: {mode}")
    print(f"File Size: {file_size / 1024:.2f} KB")
    print(f"Camera Info: {camera_info_cleaned}")
    print(f"Location Info (GPS): {location_info_cleaned}")
    print(f"Image Creation Date: {image_creation_date}")
    print(f"File Creation Time: {creation_time}")
    print(f"File Last Modified Time: {last_modified_time}")
    return{
        'Prediction': prediction_result,
        'Image Size': (size[0], size[1]),
        'Format': format,
        'Mode': mode,
        'File Size': file_size / 1024,  # In KB
        'Camera Info': camera_info_cleaned,
        'Location Info (GPS)': location_info_cleaned,
        'Image Creation Date': image_creation_date,
        'File Creation Time': creation_time,
        'File Last Modified Time': last_modified_time
    }

# Test the function with an image
test_image_path = 'Au_ani_0002.jpg'  # Change to your image path
detect_forgery(test_image_path)
details = extract_image_details(test_image_path)
print(details)