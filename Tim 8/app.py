# import train
import numpy as np
from PIL import Image, ImageFile
from flask import Flask, render_template,request, flash, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'inisecretkey'


ALLOWED_EXTENSTIONS = set(["jpg","png","jpeg","gif"])

def allowed_files(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSTIONS


# model inceptionV3 yang sudah di train dan test
model_new = load_model("model/IncDogBreed.h5")

def preprocess_foto(filepath):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    org_img=Image.open(BytesIO(filepath.read()))
    org_img.load()
    gambar=org_img.resize((300,300), Image.ANTIALIAS)
    gambar = image.img_to_array(gambar)
    org_img = image.img_to_array(org_img)
    return org_img, np.expand_dims(gambar,axis=0)

def detection_object(gambar):
    # label 
    labels = ['beagle', 'bull_mastiff', 'chihuahua', 'german_shepherd', 'golden_retriever', 'maltese', 'pomeranian', 'pug', 'shih_tzu', 'siberian_husky']
    # memprediksi foto 
    proba  = model_new.predict(gambar)[0]

    # for loop untuk menemukan hasil deteksi paling besar
    prob  = 0
    hasil = 0
    for (label, p) in zip(labels,proba):
        if p > prob:
            prob  = p
            hasil = label
    return hasil



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def prediksi():
    if "file" not in request.files:
        flash("no file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == "" :
        flash("tidak ada foto yang diupload")
        return redirect(request.url)
    if file and allowed_files(file.filename):
        # mengambil foto dari input file
        file = request.files["file"]
        # preprocess pada gambar sebelum di deteksi
        org_img, foto = preprocess_foto(file)
        # deteksi gambar
        model = detection_object(foto)

        img_x=BytesIO()
        plt.imshow(org_img/255.0)
        plt.savefig(img_x,format='png')
        plt.close()
        img_x.seek(0)
        plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')

        respon = f"ras anjing nya adalah {model}"
        flash(respon)
        return render_template('index.html', prediction=respon, plot_url=plot_url)
    else :
        flash("masukkan file berformat png, jpg, jpeg atau gif")
        return redirect(request.url)

if __name__ == "__main__":    
    app.run(debug=True)