import os.path as osp
import glob
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request
import os
import shutil

app = Flask(__name__)


import RRDBNet_arch as arch
model_path = 'models/RRDB_ESRGAN_x4.pth'  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

def empty_folder(folder_path):
    """Empty the contents of a folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route('/', methods=['GET', 'POST'])
def enhance():
    if request.method == 'POST':
        imageFile = request.files['Imagefile']
        imagePath = os.path.join('static', 'uploads', imageFile.filename)
        imageFile.save(imagePath)

        
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        
        base = os.path.splitext(os.path.basename(imagePath))[0]
        enhancedImagePath = os.path.join('static', 'results', '{:s}_rlt.png'.format(base))
        cv2.imwrite(enhancedImagePath, output)

        original_image = os.path.join('/', imagePath)
        enhanced_image = os.path.join('/', enhancedImagePath)
        return render_template('index.html', original_image=original_image, enhanced_image=enhanced_image)
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)

