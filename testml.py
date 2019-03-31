from keras.applications.resnet50 import ResNet50 # changed the module name
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

"""
Creates a CNN with the ResNet50 architecture and initializes the
weights to those trained using ImageNet. If the ImageNet ResNet50
weights aren't yet on the server, this will download them from
the keras github page.
"""

model = ResNet50(weights='imagenet')



class Classifier:
    def __init__(self):
        self._architecture = 'ResNet50'
        self._target_size = (224,224)

    def process_image(self, img_path):

        self._img_path = img_path
        img = image.load_img(img_path, target_size=self._target_size)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        self._orig_img = img
        self._x = x

    def classify_image(self):

        self._preds = model.predict(self._x)

    def get_classification(self):

        self.category = decode_predictions(self._preds, top=1)[0][0][1] # added the missing [0] because tuple with data was inside the list that was inside the list


    def get_border(self):
        if len(self.category)%2 > 0: # added self.category
            self.border = 'blue'
        else:
            self.border = 'red' # changed to red

    def response(self):
        return { 'border': self.border, 'category': self.category}

    def pipeline(self, img_path):
        self.process_image(img_path)
        self.classify_image()
        self.get_classification()
        self.get_border() # added fucntion that was missing

        return self.response()
