import operator
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from PIL import Image, ImageFile
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


app = Flask(__name__)


MODEL_PATH = 'model/my_model.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()   
print('Model loaded. Check http://127.0.0.1:5000/')

global color_dict

def preprocess(file):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    org_img=Image.open(BytesIO(file.read()))
    org_img.load()
    img=org_img.resize((200,200), Image.ANTIALIAS)
    img=image.img_to_array(img)
    org_img=image.img_to_array(org_img)
    return org_img, np.expand_dims(img,axis=0)

def classify_color(img):
    color_dict={}
    color_dict['Biru']=round(model.predict(img)[0][0]*100,8)
    color_dict['Coklat']=round(model.predict(img)[0][1]*100,8)
    color_dict['Hijau']=round(model.predict(img)[0][2]*100,8)
    color_dict['Hitam']=round(model.predict(img)[0][3]*100,8)
    color_dict['Kuning']=round(model.predict(img)[0][4]*100,8)
    color_dict['Merah']=round(model.predict(img)[0][5]*100,8)
    color_dict['Orange']=round(model.predict(img)[0][6]*100,8)
    color_dict['ungu']=round(model.predict(img)[0][7]*100,8)
    for value in color_dict:
     if color_dict[value]<=0.001:
        color_dict[value]=0.00
    return color_dict

@app.route('/', methods=['GET'])
def splash():
    return render_template('splash.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        org_img, img = preprocess(file)
        color_dict=classify_color(img)
        
        prediction = max(color_dict.items(), key=operator.itemgetter(1))[0]

        img_x=BytesIO()
        plt.imshow(org_img/255.0)
        plt.savefig(img_x,format='png')
        plt.close()
        img_x.seek(0)
        plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')
        
    return render_template('index.html', prediction=prediction, plot_url=plot_url)

    
if __name__ == "__main__":
    app.run(debug=True)