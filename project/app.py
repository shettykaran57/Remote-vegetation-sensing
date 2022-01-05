from flask import Flask, request, url_for,render_template
import requests
import urllib.parse
import requests
import urllib.parse
import os
from werkzeug.utils import redirect
from glob import glob
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import cv2

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.seterr(divide='ignore', invalid='ignore')




app = Flask(__name__)


@app.route('/')
def home():
    return render_template('base.html',   
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])

@app.route('/base.html')
def index():
    return render_template('base.html',   
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])


@app.route('/co_ordinate', methods=["GET","POST"])
def co_ordinate():
    # Get Data from input
    lat = request.form.get("latitude")
    lon = request.form.get("longitude")

    #Function for getting map
    def maptype(map):
        ln=600
        bd=300
        ratio=f"{bd}x{ln}"
        zoom="17"
        access_token="pk.eyJ1Ijoic2hldHR5a2FyYW41NyIsImEiOiJja3VwNHNuaGwyM242MzNvNnNlMm1obTdyIn0.1zYRiUsXVCn4YMJ94-Ng-Q"
        api=f"https://api.mapbox.com/styles/v1/mapbox/{map}/static/{lon},{lat},{zoom}/{ratio}?access_token={access_token}"
        return api[0:]

    #Function for Saving file
    def file_name(name,file_t):
        file_t = f"/home/karan/Remote-vegetation-sensing/project/static/data/{file_t}.tiff"
        file = open(file_t[0:], "wb")
        file.write(name.content)
        file.close()

    #function called
    sample = requests.get(maptype("satellite-v9"))
    dark = requests.get(maptype("dark-v10"))
    light = requests.get(maptype("light-v10"))
    street = requests.get(maptype("streets-v11"))
    satellite = requests.get(maptype("satellite-v9"))
    street_old = requests.get(maptype("streets-v9"))
    satellite_new = requests.get(maptype("satellite-streets-v9"))

    file_name(sample,"sample")
    file_name(dark,"dark")
    file_name(light,"light")
    file_name(street,"street")
    file_name(street_old,"street_old")
    file_name(satellite,"satellite")
    file_name(satellite_new,"satellite_new")

    file_name(sample,"sample")
    full_filename="static/data/sample.tiff" #File path to show
    #Variable to view Data
    class_cord_temp="btn btn-success btn-lg float-right"
    button_tex_temp="Preprocess"

    return render_template('base.html',cord=full_filename ,file_name="Map",
    class_cord=class_cord_temp,button_text=button_tex_temp,
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])
    



@app.route('/address', methods=["GET","POST"])
def address():
    # Get Data from input
    city = request.form.get("city")
    address=urllib.parse.quote(city)
    country = request.form.get("country")

    url = "https://nominatim.openstreetmap.org/?addressdetails=1&q=" + address + "+" + country +"&format=json&limit=1"

    response = requests.get(url).json()


    try:
        
        lat=response[0]["lat"]
        lon=response[0]["lon"]

        
        result=f"Map cordinate:-({lat},{lon})"
  

    except IndexError:
        result="Address is invalid"
        lat=""
        lon=""  




        


    print(result)
    #Function for getting map
    def maptype(map):
        ln=600
        bd=300
        ratio=f"{bd}x{ln}"
        zoom="17"
        access_token="pk.eyJ1Ijoic2hldHR5a2FyYW41NyIsImEiOiJja3VwNHNuaGwyM242MzNvNnNlMm1obTdyIn0.1zYRiUsXVCn4YMJ94-Ng-Q"
        api=f"https://api.mapbox.com/styles/v1/mapbox/{map}/static/{lon},{lat},{zoom}/{ratio}?access_token={access_token}"
        return api[0:]

    #Function for Saving file
    def file_name(name,file_t):
        file_t = f"/home/karan/Remote-vegetation-sensing/project/static/data/{file_t}.tiff"
        file = open(file_t[0:], "wb")
        file.write(name.content)
        file.close()

    #function called
    sample = requests.get(maptype("satellite-v9"))
    dark = requests.get(maptype("dark-v10"))
    light = requests.get(maptype("light-v10"))
    street = requests.get(maptype("streets-v11"))
    satellite = requests.get(maptype("satellite-v9"))
    street_old = requests.get(maptype("streets-v9"))
    satellite_new = requests.get(maptype("satellite-streets-v9"))

    file_name(sample,"sample")
    file_name(dark,"dark")
    file_name(light,"light")
    file_name(street,"street")
    file_name(street_old,"street_old")
    file_name(satellite,"satellite")
    file_name(satellite_new,"satellite_new")

    full_filename="static/data/sample.tiff" #File path to show
    #Variable to view Data
    class_cord_temp="btn btn-success btn-lg float-right"
    button_tex_temp="Preprocess"

    return render_template('base.html',cord=full_filename ,file_name=result,
    class_cord=class_cord_temp,button_text=button_tex_temp,
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])



@app.route('/preprocess' , methods=["GET","POST"])
def preprocess():

    S_sentinel_bands = glob(r"/home/karan/Remote-vegetation-sensing/project/static/data/*.tiff")
    S_sentinel_bands.sort()
    S_sentinel_bands
    l = []
    for i in S_sentinel_bands:
        with rio.open(i, 'r') as f:
            l.append(f.read(1))
    j=l
    arr_st = np.stack(j)

    # Preprocessing 
    x = np.moveaxis(arr_st, 0, -1)
    x.shape
    ln=600
    bd=300
    x.reshape(-1, 7).shape, ln*bd

    X_data = x.reshape(-1, 7)
    scaler = StandardScaler().fit(X_data)   
    X_scaled = scaler.transform(X_data)






    pca = PCA(n_components = 7)
    pca.fit(X_scaled)
    data = pca.transform(X_scaled)

    kmeans = KMeans(n_clusters = 5, random_state = 99)
    kmeans.fit(data)
    labels = kmeans.predict(data)

    fig = px.imshow(labels.reshape(600, 300), 
          color_continuous_scale = ['darkgreen', 'green', 'black', '#CA6F1E', 'navy', 'forestgreen'])

    fig.update_xaxes(showticklabels=False)

    fig.update_yaxes(showticklabels=False)

    fig=fig.update_layout(
        autosize=False,
        width=500,
        height=1000,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        # paper_bgcolor="LightSteelBlue",
    )

    fig.write_image('/home/karan/Remote-vegetation-sensing/project/static/module/preprocessing.png')
    prepos="static/module/preprocessing.png"
    fig='/home/karan/Remote-vegetation-sensing/project/static/module/preprocessing.png'
    img = cv2.imread(fig)
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []
    # loop over the image channels
    his=""
    vec=""
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
        # plot the histogram
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
        plt.savefig('/home/karan/Remote-vegetation-sensing/project/static/module/preprocessing_hist.png')
        his='static/module/preprocessing_hist.png'
        vec= np.array(features).flatten().shape
    return render_template('map.html',pre=prepos,histo=his,vector=vec)


if __name__=='__main__':
    app.run(debug=True)
