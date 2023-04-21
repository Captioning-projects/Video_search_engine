import os
from flask import render_template, redirect, flash, request
from application import app
import pandas as pd
from application.output import run_query
from flask_cors import CORS, cross_origin

CORS(app,supports_credentials=True)

@app.route('/', methods=['GET', 'POST'])
# @cross_origin()
def home():
    if request.method == 'POST':
        text = request.form['query'].lower()
        # "White dogs running"+
        values = run_query(text)
        return render_template('home.html',values=values)

    return render_template('home.html')

@app.route('/receive_json/', methods=['POST'])
# @cross_origin()
def receive_json():
    json_data = request.get_json()
    video_id = json_data['video_id']
    video_caption = json_data['video_caption']
    csv_loc = '/home/projects/Video_search_engine/output/vid_results.csv'
    df = pd.read_csv(csv_loc)
    index_ = df.tail(1).index[0]+1
    df.loc[index_] = [index_,video_caption, video_id]
    df.to_csv(csv_loc,index=False)
    return 'JSON sent and processed successfully'


@app.route('/uploadvideo', methods=['POST'])
def upload_video():
    print('hi')
    if 'video_file' not in request.files:
        print('No file uploaded')
        return 'No file uploaded', 400
    video_file = request.files['video_file']
    video_file.save('uploaded_video.mp4')
    return 'Video file uploaded successfully'

