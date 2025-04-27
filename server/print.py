from flask import Flask, request, jsonify, render_template
import base64
import os
import tempfile
from PIL import Image
import io
import datetime
import subprocess

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create images directory if it doesn't exist
os.makedirs('saved_images', exist_ok=True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/print-image', methods=['POST'])
def print_image():
    try:
        # Get base64 image from request
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_base64 = data['image_base64']
        
        # Strip the base64 prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create a white background for images with transparency
        if image.mode in ('RGBA', 'LA'):
            background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image, image.split()[-1])
            image = background
        
        # Convert to RGB mode if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_images/image_{timestamp}.jpg"
        
        # Get absolute path for the file
        abs_filepath = os.path.abspath(filename)
        
        # Save the image with high quality
        image.save(filename, format='JPEG', quality=95)
        
        # Log to console
        print(f"Image saved to: {filename}")
        
        # Open the image in the default viewer (non-blocking)
        try:
            # Use os.startfile which is available on Windows
            os.startfile(abs_filepath)
            print(f"Opened image in default viewer")
        except Exception as viewer_error:
            print(f"Warning: Could not open image in viewer: {str(viewer_error)}")
        
        return jsonify({
            'success': True,
            'message': 'Image saved to file and opened in viewer',
            'filepath': filename
        })
    
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting Flask app on http://127.0.0.1:5000")
    print(f"Images will be saved to the 'saved_images' directory")
    app.run(debug=True)
