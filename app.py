import logging
from flask import Flask, request, jsonify

from trainer_deployment_files.model import LRModel

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

model = LRModel()
model.initialize_model()

@app.route('/')
def test():
    '''
    Used for easy testing (avoid command lines to check if API works).
    When running the api, you can copy the address on your browser first to see if you see the message below.
    Alternatively: You can just CURL the url from command line to see if it returns the message below.
    If you can see it, it means the API works, and you can move on to query some features for predictions.
    '''
    return jsonify({
      'Testing': 'It works!'
    })


@app.route('/predict', methods=['POST'])
def predict():
    '''
    Return A Prediction.
    '''
    data = request.get_json()
    app.logger.info("Record To predict: {}".format(data))
    app.logger.info(type(data))
    input_data = [data["data"]]
    app.logger.info(input_data)

    input_data = model.scale_input(input_data)
    prediction = model.predict(input_data)

    app.logger.info(prediction)
    response_data = prediction[0]
    return {"prediction": str(response_data)}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
