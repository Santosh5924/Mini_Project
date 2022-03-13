import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
sc = pickle.load(open('scaler1.pkl', 'rb'))
model = pickle.load(open('classifier1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    pred = model.predict( sc.transform(final_features) )
    return render_template('result.html', prediction = pred)

if __name__ == "__main__":
    app.run(debug=True)