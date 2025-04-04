import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
import PyPDF2
import nltk
import re
import math
from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import werkzeug.utils
from datetime import datetime

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "pdf_summarizer_secret_key"

# Create upload and output directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'summaries')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

class PDFSummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization")
        self.stop_words = set(stopwords.words('english'))
        
        # Define words per minute based on average reading speeds by grade level
        self.wpm_by_grade = {
            1: 80, 2: 100, 3: 120, 4: 140, 5: 150, 
            6: 160, 7: 175, 8: 185, 9: 200, 
            10: 210, 11: 220, 12: 230
        }
        
        # Vocabulary complexity by grade level
        self.grade_vocab = {
            # Grade 1-3: Simple vocabulary
            (1, 3): ["simple", "basic", "common", "easy", "small", "big", "good", "bad"],
            # Grade 4-6: Growing vocabulary
            (4, 6): ["several", "various", "different", "important", "significant", "example"],
            # Grade 7-9: Intermediate vocabulary
            (7, 9): ["analyze", "compare", "contrast", "evaluate", "examine", "illustrate"],
            # Grade 10-12: Advanced vocabulary
            (10, 12): ["comprehensive", "substantial", "fundamental", "consequently", "nevertheless"]
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    
    def preprocess_text(self, text):
        """Clean and preprocess the extracted text."""
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove references, figure captions, etc.
        text = re.sub(r'\[.*?\]', '', text)
        # Remove page numbers
        text = re.sub(r'\b\d+\b', '', text)
        return text.strip()
    
    def calculate_readability_score(self, text):
        """Calculate the Flesch-Kincaid Grade Level score."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        word_count = len([word for word in words if word.isalpha()])
        sentence_count = len(sentences)
        
        if sentence_count == 0 or word_count == 0:
            return 0
            
        syllable_count = 0
        for word in words:
            if word.isalpha():
                syllable_count += self.count_syllables(word)
        
        # Flesch-Kincaid Grade Level formula
        score = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        return max(1, min(12, round(score)))
    
    def count_syllables(self, word):
        """Count syllables in a word using a simple heuristic."""
        word = word.lower()
        # Exception cases
        if len(word) <= 3:
            return 1
            
        # Remove 'es', 'ed' at the end
        word = re.sub(r'e$', '', word)
        word = re.sub(r'es$', '', word)
        word = re.sub(r'ed$', '', word)
        
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        return max(1, count)
    
    def adapt_to_grade_level(self, text, target_grade):
        """Adapt text complexity to the target grade level."""
        words = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        # Find appropriate vocabulary range
        vocab_range = None
        for grade_range in self.grade_vocab.keys():
            if grade_range[0] <= target_grade <= grade_range[1]:
                vocab_range = grade_range
                break
        
        if not vocab_range:
            return text
            
        # Adjust sentence structure and vocabulary based on grade level
        adapted_words = []
        for word, tag in pos_tags:
            # Replace complex words with simpler alternatives for lower grades
            if target_grade <= 6 and len(word) > 8 and word.lower() not in self.stop_words:
                # For lower grades, replace with simpler words
                adapted_words.append("important")
            else:
                adapted_words.append(word)
        
        # Adjust sentence length
        sentences = sent_tokenize(' '.join(adapted_words))
        adapted_sentences = []
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            # Target sentence length based on grade
            target_length = 5 + target_grade  # Simple formula: younger grades get shorter sentences
            
            if len(words) > target_length * 2 and target_grade < 7:
                # Split long sentences for lower grades
                mid = len(words) // 2
                adapted_sentences.append(' '.join(words[:mid]) + '.')
                adapted_sentences.append(' '.join(words[mid:]))
            else:
                adapted_sentences.append(sentence)
                
        return ' '.join(adapted_sentences)
    
    def generate_summary(self, text, grade_level, minutes):
        """Generate a summary based on grade level and time length."""
        # Calculate target word count based on reading speed
        reading_speed = self.wpm_by_grade.get(grade_level, 150)  # Default to 150 wpm if grade not found
        target_word_count = reading_speed * minutes
        
        # Determine chunk size based on the summarizer's max input length
        max_chunk_size = 1024  # Typical limit for transformer models
        
        # Process text in chunks if necessary
        if len(text.split()) > max_chunk_size:
            chunks = self.split_into_chunks(text, max_chunk_size)
            summaries = []
            
            # Calculate how many words we want from each chunk
            words_per_chunk = target_word_count // len(chunks)
            
            for chunk in chunks:
                # Calculate what percentage of the chunk we want to keep
                chunk_word_count = len(chunk.split())
                percentage = min(100, (words_per_chunk / chunk_word_count) * 100)
                
                # Summarize the chunk
                chunk_summary = self.summarizer(chunk, max_length=words_per_chunk, 
                                               min_length=max(30, words_per_chunk // 2), 
                                               do_sample=False)[0]['summary_text']
                summaries.append(chunk_summary)
            
            summary = " ".join(summaries)
        else:
            # If text is small enough, summarize it directly
            max_length = min(target_word_count, len(text.split()))
            min_length = max(30, max_length // 2)
            
            summary = self.summarizer(text, max_length=max_length, 
                                     min_length=min_length, 
                                     do_sample=False)[0]['summary_text']
        
        # Adapt the summary to the target grade level
        adapted_summary = self.adapt_to_grade_level(summary, grade_level)
        
        # Convert to bulleted summary
        return self.convert_to_bullet_points(adapted_summary)
    
    def split_into_chunks(self, text, chunk_size):
        """Split text into chunks of approximately equal size."""
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    def convert_to_bullet_points(self, text):
        """Convert a text summary into bullet points."""
        sentences = sent_tokenize(text)
        bullet_points = []
        
        # Group related sentences together
        current_point = []
        for sentence in sentences:
            # Simplified logic: Start a new bullet point every 1-2 sentences
            if len(current_point) >= 2 or not current_point:
                if current_point:
                    bullet_points.append(' '.join(current_point))
                current_point = [sentence]
            else:
                current_point.append(sentence)
        
        # Add the last point
        if current_point:
            bullet_points.append(' '.join(current_point))
            
        # Format as bullet points
        return '\n'.join([f"• {point}" for point in bullet_points])
    
    def summarize_pdf(self, pdf_path, grade_level, minutes):
        """Main method to summarize a PDF file."""
        if grade_level < 1 or grade_level > 12:
            return "Error: Grade level must be between 1 and 12."
        
        if minutes not in [10, 20, 30]:
            return "Error: Summary length must be 10, 20, or 30 minutes."
        
        try:
            text = self.extract_text_from_pdf(pdf_path)
            preprocessed_text = self.preprocess_text(text)
            
            # Determine the original grade level of the text
            original_grade = self.calculate_readability_score(preprocessed_text)
            
            # Generate summary adapted to the target grade level
            summary = self.generate_summary(preprocessed_text, grade_level, minutes)
            
            # Prepare metadata for the summary
            wpm = self.wpm_by_grade[grade_level]
            word_count = len(summary.split())
            estimated_time = round(word_count / wpm, 1)
            
            result = {
                "summary": summary,
                "original_grade": original_grade,
                "target_grade": grade_level,
                "target_minutes": minutes,
                "target_words": minutes * wpm,
                "actual_words": word_count,
                "estimated_minutes": estimated_time
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

# Initialize the summarizer
summarizer = PDFSummarizer()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and summarization."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and file.filename.endswith('.pdf'):
        # Get form data
        grade_level = int(request.form.get('grade_level', 8))
        minutes = int(request.form.get('minutes', 20))
        
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = werkzeug.utils.secure_filename(f"{timestamp}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Generate summary
        result = summarizer.summarize_pdf(file_path, grade_level, minutes)
        
        if "error" in result:
            flash(f"Error processing file: {result['error']}")
            return redirect(url_for('index'))
        
        # Save summary to a file
        output_filename = f"{os.path.splitext(filename)[0]}_g{grade_level}_{minutes}min.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        with open(output_path, 'w') as f:
            f.write(f"SUMMARY FOR GRADE {grade_level}\n")
            f.write(f"TARGET LENGTH: {minutes} minutes ({result['target_words']} words)\n")
            f.write(f"ACTUAL LENGTH: {result['estimated_minutes']} minutes ({result['actual_words']} words)\n")
            f.write(f"ORIGINAL TEXT GRADE LEVEL: {result['original_grade']}\n")
            f.write("-------------------------------------------------------------------------------\n")
            f.write(result['summary'])
        
        # Pass results to template
        return render_template('result.html', 
                              result=result,
                              filename=os.path.basename(file_path),
                              output_filename=output_filename)
    
    flash('Please upload a PDF file')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Allow downloading of summary files."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/summaries')
def view_summaries():
    """View list of all generated summaries."""
    summaries = []
    for filename in os.listdir(app.config['OUTPUT_FOLDER']):
        if filename.endswith('.txt'):
            # Extract metadata from filename
            parts = filename.replace('.txt', '').split('_')
            if len(parts) >= 2:
                grade_info = parts[-2]
                minutes_info = parts[-1]
                
                grade = grade_info.replace('g', '')
                minutes = minutes_info.replace('min', '')
                
                original_name = '_'.join(parts[:-2])
                
                summaries.append({
                    'filename': filename,
                    'original_name': original_name,
                    'grade': grade,
                    'minutes': minutes,
                    'created': os.path.getctime(os.path.join(app.config['OUTPUT_FOLDER'], filename))
                })
    
    # Sort by creation time (newest first)
    summaries.sort(key=lambda x: x['created'], reverse=True)
    
    return render_template('summaries.html', summaries=summaries)

# Create HTML templates
def create_templates():
    # Create templates directory
    templates_dir = os.path.join(os.getcwd(), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Textbook Chapter Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 2rem;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .upload-area {
            border: 2px dashed #6c757d;
            border-radius: 5px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            margin-bottom: 1rem;
            background-color: #f1f3f5;
        }
        .upload-area:hover {
            background-color: #e9ecef;
        }
        .hidden {
            display: none;
        }
        .loading {
            display: none;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h2 class="mb-0">PDF Textbook Chapter Summarizer</h2>
                    </div>
                    <div class="card-body">
                        {% with messages = get_flashed_messages() %}
                            {% if messages %}
                                {% for message in messages %}
                                    <div class="alert alert-warning">{{ message }}</div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}
                        
                        <form action="/upload" method="post" enctype="multipart/form-data" id="upload-form">
                            <div class="upload-area" id="upload-area">
                                <img src="https://cdn-icons-png.flaticon.com/512/4208/4208479.png" width="64" height="64" class="mb-3">
                                <h5>Drag & Drop PDF file here</h5>
                                <p>or click to select file</p>
                            </div>
                            <input type="file" name="file" id="file-input" class="hidden" accept=".pdf">
                            
                            <div id="file-info" class="alert alert-info hidden">
                                <p class="mb-0">Selected file: <span id="file-name"></span></p>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="grade_level" class="form-label">Grade Level:</label>
                                    <select name="grade_level" id="grade_level" class="form-select">
                                        {% for i in range(1, 13) %}
                                            <option value="{{ i }}">Grade {{ i }}</option>
                                        {% endfor %}
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label for="minutes" class="form-label">Summary Length:</label>
                                    <select name="minutes" id="minutes" class="form-select">
                                        <option value="10">10 minutes</option>
                                        <option value="20" selected>20 minutes</option>
                                        <option value="30">30 minutes</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary" id="submit-btn">Summarize</button>
                                <a href="/summaries" class="btn btn-outline-secondary">View Previous Summaries</a>
                            </div>
                        </form>
                        
                        <div class="loading text-center" id="loading">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-2">Generating summary, please wait...</p>
                            <p class="text-muted small">This may take several minutes for large documents</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const fileInfo = document.getElementById('file-info');
        const fileName = document.getElementById('file-name');
        const form = document.getElementById('upload-form');
        const loading = document.getElementById('loading');
        const submitBtn = document.getElementById('submit-btn');
        
        // Trigger file input when clicking on upload area
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Handle drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('bg-light');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('bg-light');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('bg-light');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileInfo();
            }
        });
        
        // Update file info when file is selected
        fileInput.addEventListener('change', updateFileInfo);
        
        function updateFileInfo() {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                fileInfo.classList.remove('hidden');
                fileName.textContent = file.name;
                
                // Check if file is PDF
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    fileInfo.classList.remove('alert-info');
                    fileInfo.classList.add('alert-danger');
                    fileName.textContent = file.name + ' (Please select a PDF file)';
                    submitBtn.disabled = true;
                } else {
                    fileInfo.classList.remove('alert-danger');
                    fileInfo.classList.add('alert-info');
                    submitBtn.disabled = false;
                }
            } else {
                fileInfo.classList.add('hidden');
            }
        }
        
        // Show loading spinner when form is submitted
        form.addEventListener('submit', (e) => {
            if (fileInput.files.length > 0 && fileInput.files[0].name.toLowerCase().endsWith('.pdf')) {
                form.style.display = 'none';
                loading.style.display = 'block';
            } else {
                e.preventDefault();
                alert('Please select a PDF file.');
            }
        });
    </script>
</body>
</html>
        ''')
    
    # Create result.html
    with open(os.path.join(templates_dir, 'result.html'), 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary Result - PDF Textbook Chapter Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 2rem;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .summary-content {
            white-space: pre-line;
            line-height: 1.6;
        }
        .metadata {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-10">
                <div class="card mb-4">
                    <div class="card-header bg-success text-white d-flex justify-content-between align-items-center">
                        <h2 class="mb-0">Summary Generated</h2>
                        <a href="/" class="btn btn-outline-light btn-sm">New Summary</a>
                    </div>
                    <div class="card-body">
                        <div class="metadata">
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Original File:</strong> {{ filename }}</p>
                                    <p><strong>Original Text Grade Level:</strong> {{ result.original_grade }}</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Target Grade:</strong> {{ result.target_grade }}</p>
                                    <p><strong>Target Length:</strong> {{ result.target_minutes }} minutes ({{ result.target_words }} words)</p>
                                    <p><strong>Actual Length:</strong> {{ result.estimated_minutes }} minutes ({{ result.actual_words }} words)</p>
                                </div>
                            </div>
                        </div>
                        
                        <h3>Summary</h3>
                        <div class="summary-content p-3">
                            {{ result.summary|safe }}
                        </div>
                        
                        <div class="mt-4 d-flex justify-content-between">
                            <a href="/download/{{ output_filename }}" class="btn btn-primary">Download Summary</a>
                            <a href="/summaries" class="btn btn-outline-secondary">View All Summaries</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        ''')
    
    # Create summaries.html
    with open(os.path.join(templates_dir, 'summaries.html'), 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Summaries - PDF Textbook Chapter Summarizer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 2rem;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-10">
                <div class="card mb-4">
                    <div class="card-header bg-info text-white d-flex justify-content-between align-items-center">
                        <h2 class="mb-0">All Generated Summaries</h2>
                        <a href="/" class="btn btn-outline-light btn-sm">New Summary</a>
                    </div>
                    <div class="card-body">
                        {% if summaries %}
                            <div class="table-responsive">
                                <table class="table table-hover">
                                    <thead>
                                        <tr>
                                            <th>Original Filename</th>
                                            <th>Grade Level</th>
                                            <th>Length</th>
                                            <th>Actions</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for summary in summaries %}
                                            <tr>
                                                <td>{{ summary.original_name }}</td>
                                                <td>Grade {{ summary.grade }}</td>
                                                <td>{{ summary.minutes }} minutes</td>
                                                <td>
                                                    <a href="/download/{{ summary.filename }}" class="btn btn-sm btn-primary">Download</a>
                                                </td>
                                            </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        {% else %}
                            <div class="alert alert-info">
                                <p>No summaries have been generated yet.</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        ''')

# Create launch script
def create_launch_script():
    with open('app.py', 'w') as f:
        f.write('''
import os
from flask import Flask, render_template
from pdf_summarizer_ui import app, create_templates

if __name__ == "__main__":
    # Create templates if they don't exist
    if not os.path.exists('templates'):
        create_templates()
    
    # Run the Flask app
    app.run(debug=True)
''')

# Create necessary files and folders when this script is run
if __name__ == "__main__":
    create_templates()
    create_launch_script()
    
    print("PDF Summarizer UI has been initialized.")
    print("Run 'python app.py' to start the application.")
    
    # Run the Flask app
    app.run(debug=True)