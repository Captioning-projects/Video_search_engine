import os
from flask import render_template, redirect, flash, request
from application import app

from application.output import run_query


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['query'].lower()
        # "White dogs running"+
        values = run_query(text)
        return render_template('home.html',values=values)

    return render_template('home.html')


@app.route('/uploadvideo', methods=['POST'])
def upload_video():
    print('hi')
    if 'video_file' not in request.files:
        return 'No file uploaded', 400
    video_file = request.files['video_file']
    video_file.save('uploaded_video.mp4')
    return 'Video file uploaded successfully'

