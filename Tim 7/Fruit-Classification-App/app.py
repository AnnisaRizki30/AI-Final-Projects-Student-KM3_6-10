import os
import sys
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/Prediction', methods=['GET','POST'])
def pred():
    global fruit_dict, rotten, plot_url
    if request.method=='POST':
         file = request.files['file']
         org_img, img= prediction.preprocess(file)

         print(img.shape)
         fruit_dict=prediction.classify_fruit(img)
         rotten=prediction.check_rotten(img)

         img_x=BytesIO()
         plt.imshow(org_img/255.0)
         plt.savefig(img_x,format='png')
         plt.close()
         img_x.seek(0)
         plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')


    return render_template('Pred.html', fruit_dict=fruit_dict, rotten=rotten, plot_url=plot_url)

if __name__=='__main__':
    app.run(debug=True)

