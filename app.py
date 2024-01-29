import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow import keras



app = Flask(__name__)

# Load your pre-trained model
model = load_model('machine_detect_model.h5')
class_names = ['chest-fly', 'chest-press', 'lat-pull-down', 'leg-press']  # Replace with your actual class names
# Function to make predictions
def predict_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(180, 180))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)

    # Get class names
    class_names = ['chest-fly', 'chest-press', 'lat-pull-down', 'leg-press']  # Replace with your actual class names

    result = {}
    for class_name, score in zip(class_names, predictions[0]):
        result[class_name] = float(score)

    return result


@app.route('/classify', methods=['POST'])
def classify():
    try:
        image = request.files['image']
        if image:
            # Load the image using PIL
            temp_image_path = 'temp_image.jpg'
            image.save(temp_image_path)

            # Get predictions for the image
            predictions = predict_image(temp_image_path)
           # os.remove(image)

            class_name = max(predictions, key=predictions.get)



            # Return the classification result
            return jsonify({'class_name': class_name})
        else:
            return jsonify({'error': 'No image provided in the request.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
