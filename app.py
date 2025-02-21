from flask import Flask, jsonify, send_from_directory, render_template, request
import faiss_code
import os

app = Flask(__name__)

IMAGE_FOLDER = os.path.join("static", "images") 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.get_json()
    user_instruction = data.get('instruction')
    
    # Retrieve the folder name(s) from the function
    folder_names = faiss_code.retrieve_top_images_from_text(user_instruction)

    print("start")
    print(folder_names)  # Print the folder name(s) returned
    print("end")

    # Construct the response to send back, including folder names
    images_info = [
        {"filename": "img_retrieved.png", "folder_name": folder_names[0]}  # Example for the first folder name
    ]

    return jsonify({"images": images_info})


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
