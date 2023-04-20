from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = '75f99765ab52e13194e112135e67e46cbac2d56f'
cors = CORS(app, resources={r"/api/*": {"origins": "*",
                                        "methods": ["GET", "POST"],
                                        }},
            headers=["Content-Type", "Authorization"])

from application import routes
