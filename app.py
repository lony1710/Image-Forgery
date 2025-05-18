from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
from PIL.ExifTags import TAGS
from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MongoDB Atlas connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client['image_authentix']  # Use your database name
user_collection = db['users']  # Users collection

# Email settings
email_user = os.getenv("EMAIL_USER")
email_pass = os.getenv("EMAIL_PASS")

# Store OTP for registration and login
generated_otp = ""
login_otp = ""


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load forgery detection model
model = load_model('image_forgery_model.h5')

# Convert image to ELA (Error Level Analysis)
def convert_to_ela_image(path, quality=90):
    temp_filename = 'temp_file_name.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_filename)
    return ela_image

# Preprocess image for model
def prepare_image(image_path, image_size=(128, 128)):
    ela_image = convert_to_ela_image(image_path)
    ela_image = ela_image.resize(image_size)
    ela_array = np.array(ela_image).astype('float32') / 255.0  # Normalize
    ela_array = np.expand_dims(ela_array, axis=0)  # Reshape for model
    return ela_image, ela_array

# Detect forgery
def detect_forgery(image_path):
    ela_image, image_array = prepare_image(image_path)
    prediction = model.predict(image_array)[0]
    class_label = "Authentic" if np.argmax(prediction) == 1 else "Forged"
    confidence = np.max(prediction)
    return class_label, confidence, ela_image

# Highlight fake regions with heatmap
def highlight_fake_regions(image_path):
    ela_image, _ = prepare_image(image_path)
    ela_gray = np.array(ela_image.convert('L'))
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ela_gray = cv2.resize(ela_gray, (original.shape[1], original.shape[0]))

    # Edge detection with Sobel filter
    sobelx = cv2.Sobel(ela_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(ela_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create heatmap and overlay
    heatmap = cv2.applyColorMap(sobel_edges, cv2.COLORMAP_JET)
    overlaid = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

     # Generate unique filename for overlaid image
    timestamp = str(int(time.time()))
    output_path = f'static/overlaid_image_{timestamp}.jpg'
    
    # Save overlaid image with the unique name
    cv2.imwrite(output_path, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
    return output_path

# Calculate ELA Variance
def calculate_ela_variance(ela_image):
    variance = np.var(np.array(ela_image.convert('L')))
    return ("Low" if variance < 50 else "Medium" if variance < 150 else "High"), variance

# Calculate Noise Consistency
def calculate_noise_consistency(ela_image):
    noise_std = np.std(np.array(ela_image.convert('L')))
    return ("Low" if noise_std < 30 else "Medium" if noise_std < 70 else "High"), noise_std

# Extract image metadata
def get_image_metadata(image_path):
    img = Image.open(image_path)
    size, format, mode = img.size, img.format, img.mode
    file_size = os.path.getsize(image_path) / 1024  # KB
    return size, format, mode, f"{file_size:.2f} KB"



def get_image_creation_date(image_path):
    """
    Extract the image creation date from EXIF metadata.
    If no EXIF data or DateTime tag is found, return 'N/A'.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data and isinstance(exif_data, dict):  # Ensure exif_data is a dictionary
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'DateTime':
                    return value
        return "N/A"  # Fallback if no creation date is found
    except Exception as e:
        print(f"Error extracting creation date: {e}")
        return "N/A"

def get_camera_info(image_path):
    """
    Extract camera make and model from EXIF metadata.
    If no EXIF data or camera info is found, return 'N/A' for both fields.
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        camera_info = {'Make': 'N/A', 'Model': 'N/A'}  # Default values
        if exif_data and isinstance(exif_data, dict):  # Ensure exif_data is a dictionary
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == 'Make':
                    camera_info['Make'] = clean_string(value)
                elif tag_name == 'Model':
                    camera_info['Model'] = clean_string(value)
                
        return camera_info
    except Exception as e:
        print(f"Error extracting camera info: {e}")
        return {'Make': 'N/A', 'Model': 'N/A'}

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


def clean_string(s):
    """
    Clean strings and remove null characters.
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    return s.strip('\x00')

 
@app.route('/')   
def index():
    return render_template('index.html')

@app.route('/login')   
def login_page():
    return render_template('login.html')

@app.route('/register')   
def register_page():
    return render_template('signup.html')

@app.route('/dashboard')   
def dashboard():
    return render_template('dashboard.html')

@app.route('/analysis')   
def analysis():
    return render_template('analysis.html')

@app.route('/history')   
def history():
    return render_template('reports-history.html')
@app.route('/logout')   
def logout():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400




    file = request.files['file']
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    try:
        # Run analysis
        class_label, confidence, ela_image = detect_forgery(image_path)
        highlighted_image_path = highlight_fake_regions(image_path) if class_label == "Forged" else None
        ela_variance_label, ela_variance_value = calculate_ela_variance(ela_image)
        noise_consistency_label, noise_consistency_value = calculate_noise_consistency(ela_image)
        size, format, mode, file_size = get_image_metadata(image_path)

        # Get additional metadata
        image_creation_date = get_image_creation_date(image_path)
        camera_info = get_camera_info(image_path)
        location_info = get_location_info(image_path)

        # Format metadata
        metadata = f"""
        Size: {size[0]}x{size[1]}
        Format: {format}
        Mode: {mode}
        File Size: {file_size}
        Creation Date: {image_creation_date}
        Camera Make: {camera_info['Make']}
        Camera Model: {camera_info['Model']}
        Location: {location_info['Latitude']}, {location_info['Longitude']} (Timestamp: {location_info['Timestamp']})
        """

        response_data = {
            'class_label': class_label,
            'confidence': f"{confidence * 100:.2f}%",
            'ela_variance': f"{ela_variance_label} (Variance: {ela_variance_value:.2f})",
            'noise_consistency': f"{noise_consistency_label} (Std Dev: {noise_consistency_value:.2f})",
            'metadata': metadata.strip(),
            'image': highlighted_image_path if class_label == "Forged" else None
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500


def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])


def send_otp_email(to_email, otp, subject):
    """Send OTP email using SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = to_email
        msg['Subject'] = subject

        body = f"Your OTP is: {otp}"
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_user, email_pass)
            text = msg.as_string()
            server.sendmail(email_user, to_email, text)

        print(f"OTP sent to {to_email}")
    except Exception as e:
        print(f"Error sending OTP email: {e}")


@app.route('/register', methods=['POST'])
def register():
    """Register new user and send OTP"""
    global generated_otp
    data = request.json
    email = data.get('email')
    full_name = data.get('fullName')
    password = data.get('password')

    if not email or not full_name or not password:
        return jsonify({'error': 'Please provide full name, email, and password'}), 400

    # Check if email already exists
    if user_collection.find_one({'email': email}):
        return jsonify({'error': 'Email already exists. Please login instead'}), 400

    # Generate OTP and send via email
    generated_otp = generate_otp()
    send_otp_email(email, generated_otp, 'Your OTP for Image Authentix Registration')

    return jsonify({'message': 'OTP sent successfully!'}), 200


@app.route('/verify-registration-otp', methods=['POST'])
def verify_registration_otp():
    """Verify registration OTP"""
    global generated_otp
    data = request.json
    otp = data.get('otp')

    if otp == generated_otp:
        return jsonify({'message': 'OTP verified successfully'}), 200
    else:
        return jsonify({'error': 'Invalid OTP. Please try again.'}), 400


@app.route('/register-final', methods=['POST'])
def register_final():
    """Complete registration after OTP verification"""
    global generated_otp
    data = request.json
    email = data.get('email')
    full_name = data.get('fullName')
    password = data.get('password')
    otp = data.get('otp')

    if otp != generated_otp:
        return jsonify({'error': 'Invalid OTP. Please try again.'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    user = {
        'fullName': full_name,
        'email': email,
        'password': hashed_password
    }

    user_collection.insert_one(user)
    return jsonify({'message': 'User registered successfully!'}), 201


@app.route('/login', methods=['POST'])
def login():
    """Login and send OTP for login"""
    global login_otp
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Please provide both email and password'}), 400

    user = user_collection.find_one({'email': email})
    if not user:
        return jsonify({'error': 'User not found. Please register'}), 400

    # Check password
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'error': 'Invalid password'}), 400

    # Generate OTP and send via email
    login_otp = generate_otp()
    send_otp_email(email, login_otp, 'Your OTP for Image Authentix Login')

    return jsonify({'message': 'OTP sent successfully!'}), 200


@app.route('/verify-login-otp', methods=['POST'])
def verify_login_otp():
    """Verify login OTP"""
    global login_otp
    data = request.json
    otp = data.get('otp')

    if otp == login_otp:
        return jsonify({'message': 'OTP verified successfully. You are now logged in.'}), 200
    else:
        return jsonify({'error': 'Invalid OTP. Please try again.'}), 400



if __name__ == '__main__':
    app.run(debug=True)
