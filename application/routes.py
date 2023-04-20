import os
from flask import render_template, redirect, flash, request
from application import app
import pandas 
from application.output import run_query


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['query'].lower()
        # "White dogs running"
        values = run_query(text)
        return render_template('home.html',values=values)

    return render_template('home.html')

@app.route('/receive_json', methods=['POST'])
def receive_json():
    json_data = request.get_json()
    video_id = json_data['video_id']
    video_caption = json_data['video_caption']
    csv_loc = '/home/projects/Video_search_engine/output/vid_results.csv'
    df = pd.read_csv(csv_loc)
    index_ = df.tail(1).index[0]+1
    df.loc[index_] = [video_caption, video_id]
    df.to_csv(csv_loc)
    return 'JSON sent and processed successfully'


    
