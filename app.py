#import libraries
from multiprocessing.managers import Value

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model_data = pickle.load(open('model.pkl', 'rb'))
loaded_model = model_data["model"]
label_encoder = model_data["label_encoder"]


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
    except ValueError:
        return render_template('index.html', prediction_text="Vui lòng nhập đúng định dạng cho tất cả các thông tin.")


    input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

    predicted_variety_numberic = loaded_model.prdict(input_data)
    predicted_variety = label_encoder.inverse_transform([int(round(predicted_variety_numberic[0]))])


    return render_template('index.html', prediction_text=f"Dự đoán loài hoa: {predicted_variety[0]}")

if __name__ == "__main__":
    app.run(debug=True)