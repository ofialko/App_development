from flask import Flask
import os
from pathlib import Path
import subprocess

model_path = 'ImageCaptioning.pytorch'

app = Flask(__name__)

@app.route("/")
def predict():
    cmd = 'python eval.py --model data/FC/fc-model.pth --infos_path data/FC/fc-infos.pkl --image_folder ../imgs'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,cwd=model_path)
    out, _ = process.communicate()

    return out

    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
    
  
  
