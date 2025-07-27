from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from glcm import GLCMAnalyzer
import os
import numpy as np

app = Flask(__name__)
app.config.update({
    'SECRET_KEY': 'your-secret-key',
    'UPLOAD_FOLDER': 'static/uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB
})

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
OFFSETS = {
    'Horizontal (1,0)': (1, 0),
    'Vertical (0,1)': (0, 1),
    'Diagonal (1,1)': (1, 1),
    'Anti-diagonal (-1,1)': (-1, 1)
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            analyzer = GLCMAnalyzer(filepath)
            results = {}
            
            for direction, (dx, dy) in OFFSETS.items():
                glcm = analyzer.compute_glcm(dx, dy)
                features = analyzer.extract_features(glcm)
                save_prefix = os.path.join(app.config['UPLOAD_FOLDER'], 
                                         f"{direction.replace(' ', '_')}_{os.path.splitext(filename)[0]}")
                analyzer.visualize_glcm(glcm, save_prefix)
                
                results[direction] = {
                    'features': features,
                    'heatmap': f"{save_prefix}_2d.png",
                    'surface': f"{save_prefix}_3d.png"
                }
            
            return render_template('index.html', 
                                 original=filename,
                                 results=results)
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
