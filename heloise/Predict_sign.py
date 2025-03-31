import tkinter as tk
from tkinter import filedialog
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('traffic_sign_model.h5')

# Mapping from model output to sign names
signs = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "Entry prohibited",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signal",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def predict_sign(image_path):
    img = image.load_img(image_path, target_size=(30, 30))  # Match the input size of the model
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    probability = np.max(predictions)
    return (predicted_class, signs[predicted_class], probability)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_sign(file_path)
        result_label.config(text=f'Prediction: {prediction[1]} (Class: {prediction[0]}, Probability: {prediction[2]:.3f})')

app = tk.Tk()
app.title("Traffic Sign Predictor")

upload_button = tk.Button(app, text="Upload Image", command=upload_image)
upload_button.pack()

result_label = tk.Label(app, text="")
result_label.pack()

app.mainloop()