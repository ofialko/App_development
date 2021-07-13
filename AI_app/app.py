from flask import Flask, jsonify
import subprocess

model_path = 'ImageCaptioning.pytorch'

app = Flask(__name__)

@app.route("/")
def predict():
    cmd_list = ['python', f'{model_path}/eval.py', 
        '--model', f'{model_path}/data/FC/fc-model.pth',
        '--infos_path', f'{model_path}/data/FC/fc-infos.pkl', 
        '--image_folder', 'imgs']  

    output = subprocess.run(cmd_list, capture_output=True, text=True).stdout
    
    return output

    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
    
  
  
