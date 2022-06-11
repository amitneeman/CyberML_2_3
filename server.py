from flask import Flask, abort, request, current_app
from functools import wraps
from flask_limiter import Limiter

from modelSDK import ModelSKD

app = Flask(__name__)


def get_ip():
    return request.environ['REMOTE_ADDR']


limiter = Limiter(app, get_ip)


def log_request_params(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("The client IP is: {}".format(request.environ['REMOTE_ADDR']))
        print("The client port is: {}".format(request.environ['REMOTE_PORT']))
        print("The client request json is: {}".format(request.get_json()))
        return f(*args, **kwargs)

    return decorated_function


@app.post('/predator/predict')
@log_request_params
@limiter.limit("10/minute", key_func=lambda: get_ip())
def predict_predator():
    request_data = request.get_json()
    prediction_result = current_app.model_sdk.predict_label_for_chat(request_data)
    return {"prediction": prediction_result}


if __name__ == '__main__':
    # run app in debug mode on port 5000
    with app.app_context():
        current_app.model_sdk = ModelSKD.load_model()
    app.run(debug=True, port=5000)
