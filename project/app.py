from flask import Flask, request, url_for,render_template
import requests
import urllib.parse
import requests
import urllib.parse
import os
from werkzeug.utils import redirect



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
    lat = request.form.get("latitude")
    lon = request.form.get("longitude")
    # Some internal stuff with processed text

    def maptype(map):
        ln=600
        bd=300
        ratio=f"{bd}x{ln}"
        zoom="17"
        access_token="pk.eyJ1Ijoic2hldHR5a2FyYW41NyIsImEiOiJja3VwNHNuaGwyM242MzNvNnNlMm1obTdyIn0.1zYRiUsXVCn4YMJ94-Ng-Q"
        api=f"https://api.mapbox.com/styles/v1/mapbox/{map}/static/{lon},{lat},{zoom}/{ratio}?access_token={access_token}"
        return api[0:]
    def file_name(name,file_t):
        file_t = f"/home/karan/Remote-vegetation-sensing/project/static/data/{file_t}.tiff"
        file = open(file_t[0:], "wb")
        file.write(name.content)
        file.close()
    sample = requests.get(maptype("satellite-v9"))
    file_name(sample,"sample")
    full_filename="static/data/sample.tiff" #File path to show
    class_cord_temp="btn btn-success btn-lg float-right"
    button_tex_temp="Preprocess"

    return render_template('base.html',cord=full_filename ,file_name="Map",
    class_cord=class_cord_temp,button_text=button_tex_temp,
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])
    
   
    
"""

@app.route('/test')
def index():

    return render_template('base.html')



@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    return render_template('base.html',j=select) # just to see what select is
"""
if __name__=='__main__':
    app.run(debug=True)
