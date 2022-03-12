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
from flask import send_file,Flask,send_from_directory



import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import cv2
from fpdf import FPDF
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.seterr(divide='ignore', invalid='ignore')

local_path='/home/karan/Remote-vegetation-sensing/project/'


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('base.html',   
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])

"""
@app.route('/base.html')
def index():
    return render_template('base.html',   
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])"""


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
        file_t = f'{local_path}static/data/{file_t}.tiff'
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

    return render_template('base.html',cord=full_filename ,file_name="Map and View",
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
        file_t = f'{local_path}static/data/{file_t}.tiff'
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


#Preprocess----
@app.route('/preprocess' , methods=["GET","POST"])
def preprocess():
    S_sentinel_bands = glob(r'/home/karan/Remote-vegetation-sensing/project/static/data/*.tiff')
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

    fig.write_image(f'{local_path}static/module/preprocessing.png')
    prepos="static/module/preprocessing.png"
    fig=f'{local_path}static/module/preprocessing.png'
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
        plt.savefig(f'{local_path}static/module/preprocessing_hist.png')
        his='static/module/preprocessing_hist.png'
        vec= np.array(features).flatten().shape
    return render_template('preprocess.html',pre=prepos,histo=his,vector=vec)


@app.route('/ndvi' , methods=["GET","POST"])
def ndvi():
    S_sentinel_bands = glob(r'/home/karan/Remote-vegetation-sensing/project/static/data/*.tiff')
    S_sentinel_bands.sort()
    S_sentinel_bands
    l = []
    for i in S_sentinel_bands:
        with rio.open(i, 'r') as f:
            l.append(f.read(1))
    j=l
    arr_st = np.stack(j)
        

    ndvi = es.normalized_diff(arr_st[6], arr_st[3])

    ndvi_image=ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

    ndvi_image.figure.savefig(f'{local_path}static/module/ndvi_image.png')
    nd='static/module/ndvi_image.png'
    fig=f'{local_path}static/module/ndvi_image.png'
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
        plt.savefig(f'{local_path}static/module/ndvi_hist.png')
        his='static/module/ndvi_hist.png'
        vec= np.array(features).flatten().shape

    return render_template('ndvi.html',ndvi=nd,histo=his)

@app.route('/savi' , methods=["GET","POST"])
def savi():
    S_sentinel_bands = glob(r'/home/karan/Remote-vegetation-sensing/project/static/data/*.tiff')
    S_sentinel_bands.sort()
    S_sentinel_bands
    l = []
    for i in S_sentinel_bands:
        with rio.open(i, 'r') as f:
            l.append(f.read(1))
    j=l
    arr_st = np.stack(j)

    L = 0.5

    savi = ((arr_st[6] - arr_st[3]) / (arr_st[6] + arr_st[3] + L)) * (1 + L)
    savi_image=ep.plot_bands(savi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14))

    
    savi_image.figure.savefig(f'{local_path}static/module/savi_image.png')
    sa='static/module/savi_image.png'
    fig=f'{local_path}static/module/savi_image.png'
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
        plt.savefig(f'{local_path}static/module/savi_hist.png')
        his='static/module/savi_hist.png'

    return render_template('savi.html',savi=sa,histo=his)

@app.route('/hsv' , methods=["GET","POST"])
def hsv():
    loc=f'{local_path}static/data/sample.tiff'
    imagergb = cv2.imread(loc)
    hsvImage = cv2.cvtColor(imagergb, cv2.COLOR_BGR2HSV)
    #converting the image to HSV color space using cvtColor function
    imagehsv = cv2.cvtColor(imagergb, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f'{local_path}static/model/hsv.png', imagehsv)
    #function for creating model
    def imagemask(lower,upper,name):
        lower=np.array(lower)
        upper=np.array(upper)
        imagemask=cv2.inRange(imagehsv, lower, upper)
        cv2.imwrite(f'{local_path}static/model/{name}.png', imagemask)
    # imagemask fucntion Syntax imagemask([lowerthreshold],[upperthreshold],nameoffile)
    imagemask([60, 42, 43],[135, 255, 255],"crop")
    imagemask([29, 72, 39],[42, 100, 250],"barenland")
    imagemask([0, 0, 0],[359, 84, 80],"cultivated_land")
    imagemask([0, 87, 69],[130, 100, 80],"tree")
    ori='static/data/sample.tiff'
    hs='static/model/hsv.png'

    return render_template('hsv.html',original=ori,hsv=hs)

@app.route('/mask' , methods=["GET","POST"])
def mask():
    crp='static/model/crop.png'
    barr='static/model/barenland.png'
    cul='static/model/cultivated_land.png'
    tre='static/model/tree.png'
    return render_template('mask.html',bare=barr,crop=crp,cult=cul,tree=tre)


@app.route('/data' , methods=["GET","POST"])
def data():

    crop=f'{local_path}static/model/crop.png'
    barenland=f'{local_path}static/model/barenland.png'
    cultivated_land=f'{local_path}static/model/cultivated_land.png'
    tree=f'{local_path}static/model/tree.png'
    def area(path,name):
        img = cv2.imread(path)
        # counting the number of pixels
        number_of_white_pix = np.sum(img == 255)
        number_of_black_pix = np.sum(img == 0)
        total=number_of_white_pix + number_of_black_pix
        percent=(number_of_white_pix/total)*100
        return {name:percent}


    datas={}
    datas.update({'Data' : 'Percentage'})
    datas.update(area(tree,'Tree'))
    datas.update(area(crop,'Crop'))
    datas.update(area(barenland,'Baren Land'))
    datas.update(area(cultivated_land,'Cultivated Land'))

    
    total={}
    total.update(area(tree,'tree'))
    total.update(area(crop,'crop'))
    total.update(area(barenland,'barenland'))
    total.update(area(cultivated_land,'cultivated_land'))
    sizes = total.values()
    add=0
    for i in sizes:
        add=add+i
    adds=add
    labels = total.keys()


    data_dict = total
    data_items = data_dict.items()
    data_list = list(data_items)

    df = pd.DataFrame(data_list)
    df.columns =['Data', 'Percentage']

    fig, ax = plt.subplots()
    fig.set_size_inches(3, 3)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    #ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    ax.set_title('Remote Vegetation')
    plt.legend(loc='lower right')
    plt.savefig('df.png')

    pdf=FPDF(format='letter')
    pdf.add_page() #always needed
    pdf.set_font('arial', 'B', 11)
    pdf.cell(60)
    pdf.cell(75, 10, 'Remote Vegetation Index', 0, 2, 'C')
    pdf.cell(90, 10, '', 0, 2, 'C')


    columnNameList = list(df.columns)
    for header in columnNameList[:-1]:
        pdf.cell(35, 10, header, 1, 0, 'C')
    pdf.cell(35, 10, columnNameList[-1], 1, 1, 'C')
    pdf.set_font('arial', '', 11)

    for i in range(0, len(df)):
        pdf.cell(60)
        pdf.cell(35, 10, df['Data'][i], 1, 0, 'C')
        pdf.cell(35, 10, str(round(100*(df['Percentage'][i]/adds), 2)), 1, 1, 'C')
    pdf.cell(90, 10, '', 0, 2, 'C')
    pdf.cell(55, 10, '', 0, 0, 'C')

    #insert chart
    pdf.image('df.png', x = None, y = None, w=0, h=0, type='', link='')
    pdf.output('df.pdf', 'F')

    """
    total=total.values()
    add=0
    for i in total:
        add=add+i
    add=100-add
    datas["Unknown"] = add

    
    """

    return render_template ('data.html',data=datas )

@app.route('/view', methods=["GET","POST"])
def view():
    workingdir = os.path.abspath(os.getcwd())
    filepath = '/home/karan/Remote-vegetation-sensing/df.pdf'
    return send_file(filepath)
    
    

    
if __name__=='__main__':
    app.run(debug=True)
