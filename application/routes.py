import os
from flask import render_template, redirect, flash, request
from application import app

from application.output import run_query


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['query'].lower()
        # "White dogs running"
        values = run_query(text)
        return render_template('home.html',values=values)

    return render_template('home.html')

@app.route('/receive_json, methods=['POST'])
def receive_json():
    json_data = request.get_json()
    video_id = json_data['video_id']
    video_caption = json_data['video_caption']
    with open('specific_text_file.txt', 'a') as f:
        f.write(f'{video_id}, {video_caption}\n')
    return 'JSON sent and processed successfully'


    
