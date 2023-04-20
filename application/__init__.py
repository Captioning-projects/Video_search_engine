from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['SECRET_KEY'] = '75f99765ab52e13194e112135e67e46cbac2d56f'
cors = CORS(app, resources={r"/receive_json/": {"origins": "*",
                                        "methods": ["GET", "POST"],
                                        }},
            headers=["Content-Type", "Authorization"])

# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

from application import routes
