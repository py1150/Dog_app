
from flask import Flask, render_template, request

# Flask instance
app = Flask(__name__)

#from classify_dogs import classify_dog_breed, VGG19_predict_breed

# import all python functions
#from import

# load model - see import
#VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')

# define variables here
testvar='teststr'

# home
@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        img_path=request.form['path']
        #return redirect(url_for('result'))
        return redirect('result.html')
    else:
        return render_template('home.html')

# result page
@app.route('/result', methods=['GET','POST'])
def result():

    #classify_dogs.classify_dog_breed(img_path)
    return render_template('result.html', model=testvar)

if __name__ == '__main__':
    app.run(debug=True)