from flask import Flask
from testml import Classifier

app = Flask(__name__)

classifier = Classifier()
out = classifier.pipeline('elephant.jpg')

@app.route("/elephant")
def ml():

    return str(out)

if __name__ == "__main__":
    app.run()

#the result should be on localhost:5000/elephant