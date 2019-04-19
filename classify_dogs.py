# import necessary libraries
from keras.callbacks import ModelCheckpoint


# import predict method
from extract_bottleneck_features import *

def VGG19_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# load model
# reference to file path
VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')


# algorithm as function
def classify_dog_breed(img_path):

    # show image function
    def show_image(img_path):
        import cv2
        import matplotlib.pyplot as plt
        #%matplotlib inline only valid for Jupyter notebook

        # load color (BGR) image
        img = cv2.imread(img_path)

        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # display the image, along with bounding box
        plt.imshow(cv_rgb)
        plt.show()

    # classify dog / human
    def classify_type(img_path):
        """
        classifies input image in type: (human, dog, neither)
        output:
        type_i_dict...dictionary
        """
        # initialize dictionary
        type_i_dict={'human':False,'dog':False,'neither':False}
        #print(type_i_dict)

        # check for human, dog, neither
        type_i_dict['human']=face_detector(img_path)
        type_i_dict['dog']=dog_detector(img_path)

        # special cases

        # neither dog nor human
        if (type_i_dict['human']==False & type_i_dict['dog']==False):
            type_i_dict['neither']==True
        else:
            type_i_dict['neither']==False

        # both dog and human --> in case of doubt: dog (classifier no classification error above)
        if (type_i_dict['human']==True & type_i_dict['dog']==True):
            type_i_dict['human']==False
        #print(type_i_dict)

        return(type_i_dict)

    # Show image
    show_image(img_path)

    # classify the image type: dog / human
    image_type=classify_type(img_path)

    # Predict and print output
    if image_type['dog']==True:
        print('The image shows a dog.\n')
        prediction=VGG19_predict_breed(img_path)
        print('Predicted breed: %s' %prediction[15:len(prediction)])

    elif image_type['human']==True:
        print('The image shows a human.')
        prediction=VGG19_predict_breed(img_path)
        print('Resembled breed: %s' %prediction[15:len(prediction)])
        # show example image
        search=prediction[15:len(prediction)]
        test_list=[name for name in train_files if search in name]
        if len(test_list)>0:
            example_image=test_list[0]
            print('Comparison:')
            show_image(example_image)

    elif image_type['neither']==True:
        print('The image shows neither a dog, nor a human.\n')
        print('A prediction cannot be made.')

    else:
        print('Error.')