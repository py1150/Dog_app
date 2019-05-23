import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug.utils import secure_filename



# file upload
#UPLOAD_FOLDER = 'static/images'
#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



# Flask instance
app = Flask(__name__)

print('start import classify_dogs')
from classify_dogs import *
graph = tf.get_default_graph()



# define upload set
photos=UploadSet('photos',IMAGES)
# set destination folder for upload
app.config['UPLOADED_PHOTOS_DEST']='static/images'
# load configuration for upload sets
configure_uploads(app,photos)

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define variables here
#testvar='teststr'

# test prediction
#test_image='/home/bernd/Pictures/Dachshund.jpg'
#test_image='/home/bernd/Documents/Python/Dog_app/static/images/Dachshund.jpg'
#test_pred=classify_dog_breed(test_image)
#print(test_pred)

#def allowed_file(filename):
#	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# home
@app.route('/', methods=['GET','POST'])
def index():
    #if request.method=='POST' and 'photo' in request.files:
     #   filename=photos.save(request.files['photo'])
      #  return filename
        #flash('File(s) successfully uploaded')
    return render_template('home2.html')


        # check if the post request has the file part
        #if 'file' not in request.files:
        #    flash('No file part')
        #    return redirect(request.url)
        #file = request.files['file']
        #if file.filename == '':
        #    flash('No file selected for uploading')
        #    return redirect(request.url)
        #if file and allowed_file(file.filename):
        #    filename = secure_filename(file.filename)
        #    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #    flash('File(s) successfully uploaded')
            #return redirect('/result')
        #    return render_template('result.html')
    #else:
        #return render_template('home.html')


# result page
@app.route('/result', methods=['GET','POST'])
def result():
    if request.method=='POST' and 'photo' in request.files:
       filename=photos.save(request.files['photo'])
       global graph
       with graph.as_default():
        i_type, pred=classify_dog_breed('/home/bernd/Documents/Python/Dog_app/static/images/'+filename)
       fullname='/home/bernd/Documents/Python/Dog_app/static/images/'+filename
       return render_template('result2.html', userval=filename, fullname=fullname, pred=pred, i_type=i_type)

    #if request.method=='POST' and 'photo' in request.files:
    #    filename = photos.save(request.files['photo'])
    #    return render_template('result.html', userval=filename)
        #    #classify_dog_breed(img_path)

    #    img_path=request.form['path']
    #else:
    return render_template('home2.html')

if __name__ == '__main__':
    app.secret_key = '123'
    app.run(debug=True)