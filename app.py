from flask import Flask, render_template,request
from sklearn.linear_model import LinearRegression
import numpy as np
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/index', methods=['POST'])
def index():
    input_x = request.form['a']
    input_y = request.form['b']
    x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([2, 4, 5, 8, 10, 12, 14, 16, 18, 20])
    model = LinearRegression()
    model.fit(x, y)
    x_new = np.array([[float(input_x)]])
    y_new = model.predict(x_new)
    return f'Predicted Y for X = {x_new} is {y_new[0]}'
if __name__ == '__main__':
    app.run(debug=True)
