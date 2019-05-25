# Dog_app

Dog_app is a web application which uses a deep learning model for image classifaction in order to classify images according to dog breed. If an image contains a human being, the most resembling dog breed is displayed. The web application is started by executing the file run.py.
The repository also contains the juypter notebook file which was used to estimate the model (dog_app.pdf and dog_app.ipynb)
The app can be run by executing run.py (details see below)


General Info
-------------
Technically, the web app is depicted by the following procedure:
- the user is prompted to upload an image to be classified according to dog breed
- after successful upload, the result as provided


Contained files and folders
----------------------------

- /saved_models
	contains all models the web app is using
- /static
	contains style.css to add style to the html files of the web app
	
- /static/images
	folder is used to store the uploaed images
		
- /templates
	contains all html files needed to run the application
	
	The app consists of two different sites which are displayed: 
	- home2.html: home page to prompt the user to upload an image
	- result2.html: site to display results of classification 
	
- run.py

	is used to execute the web application

- classify_dogs.py

	contains all functions and libraries necessary to carry out the classification and to provide the result
	is imported by run.py

- load_model.py

	loads the deep learning model to perform the classification
	is imported by classify_dogs.py	
	
		
Jupyter Notebook files:

These files contain the Jupyter notebook which was used to derive all models. They are not necessary for running the web app.
- dog_app.ipynb
- dog_app.pdf 


Source Information
------------------
All images are provided by unsplash.com


Project Information
-------------------
- Project overview and motivation

The goal of the project is to classify a dog breed from an image. In case, the image of a human is provided, the dog breed which most closely resembles the person, should be classified.
In order to do so, a convolutional neural network is trained. Specifically, transfer learning is used to employing a pre-trained model as a basis (VGG19) which is enhanced by adding additional convolutional network layers.

In order to test the model, accuracy on training and test sets are measured. Also, example images of dogs and humans are provided which are then classified by the model.

The functionality of the model is provided to a user by integrating it into a web app.

The model is trained in a Jupyter notebook. The flask framework is then used to create a web app based on the derived model.


- Conclusion

The model is successfully trained and can be employed to classify dogs/humans as well as their (resembled) breed.
A satisfactory accuracy is reached given the training time / GPU capability constraints. This is reflected in the results of the example image where it can be seen
that the model clearly differs successfully between dogs and humans. Also, it is capable to identify the dog breed or a closely related dog breed. However, as can be 
expected from the obtained accuracy, the classfication occasionally fails.

Further room for improvements could be derived from the following areas:
- additional investment in training time with GPU capabilities
- addtional data preprocessing such as data augmentation
- use of other pre-trained models such as "Resnet50"

