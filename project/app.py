from flask import Flask, request, url_for,render_template
import requests
import urllib.parse

app = Flask(__name__)

@app.route('/')
def home():
 
    return render_template('base.html',   
    data=[{'choose': 'Choose option'},{'choose': 'Co-ordinate'},{'choose': 'Address'}])


@app.route('/co_ordinate')
def index():
    lat = request.form.get("latitude") 
    lon = request.form.get("logitude") 

    return render_template('base.html')


"""
@app.route("/test" , methods=['GET', 'POST'])
def test():
    select = request.form.get('comp_select')
    return render_template('base.html',j=select) # just to see what select is
"""
if __name__=='__main__':
    app.run(debug=True)
