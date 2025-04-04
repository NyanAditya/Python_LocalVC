import os
import re
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import keras_nlp
from tensorflow import keras
import torch
from ipywidgets import FileUpload, Button, IntSlider, RadioButtons, Output, VBox, HBox, Label, Layout
from IPython.display import display, clear_output
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    print("Warning: Could not download NLTK punkt resource automatically.")

##########################################
# Configuration and Model Initialization #
##########################################

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configure OCR options
USE_IMAGE_OCR = True  # Set to True for enhanced symbol and image detection
OCR_RESOLUTION = 300
pytesseract.pytesseract.tesseract_cmd = r'tesseract'

# Load the Gemma 2B model
print("Loading Gemma 2B model...")
model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
print("Model loaded successfully!")

# Initialize stemmer for duplicate detection
stemmer = PorterStemmer()

#########################
# Utility Functions     #
#########################

MATH_SYMBOLS = {
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty',
    '≠': '\\neq', '≤': '\\leq', '≥': '\\geq', '±': '\\pm', '→': '\\to',
    '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega',
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon'
}

def get_stemmed_key(sentence, num_words=5):
    """Create a stemmed key for duplicate detection."""
    words = re.findall(r'\w+', sentence.lower())[:num_words]
    return ' '.join([stemmer.stem(word) for word in words])

def complete_sentence(fragment):
    """Complete sentence fragments using the model."""
    prompt = f"Complete this sentence to make it grammatically correct and meaningful. Keep it concise. Fragment: '{fragment}'\nCompleted sentence:"
    try:
        completed = model.generate(prompt, max_length=len(prompt)+50, temperature=0.1)
        return completed[len(prompt):].strip()
    except:
        return fragment  # Return original if completion fails

def preprocess_image_for_math_ocr(image):
    """Preprocess image for improved math symbol OCR."""
    if image.mode != 'L':
        image = image.convert('L')
    image_array = np.array(image)
    threshold = np.mean(image_array) * 0.9
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def extract_text_from_pdf(pdf_path, detect_math=True):
    """Improved text extraction with better image handling."""
    print(f"Extracting text from {pdf_path}...")
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num+1}/{len(pdf.pages)}...")
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
            if USE_IMAGE_OCR and page.images:
                for img_num, img in enumerate(page.images):
                    try:
                        x0 = max(0, img['x0'])
                        top = max(0, img['top'])
                        x1 = min(page.width, img['x1'])
                        bottom = min(page.height, img['bottom'])
                        bbox = (x0, top, x1, bottom)
                        cropped = page.crop(bbox)
                        pil_image = cropped.to_image(resolution=OCR_RESOLUTION).original
                        if detect_math:
                            pil_image = preprocess_image_for_math_ocr(pil_image)
                        ocr_text = pytesseract.image_to_string(pil_image, config='--psm 6 --oem 3')
                        if ocr_text.strip():
                            if detect_math:
                                for symbol, latex in MATH_SYMBOLS.items():
                                    ocr_text = ocr_text.replace(symbol, f" {latex} ")
                            extracted_text += " " + ocr_text + "\n"
                    except Exception as e:
                        print(f"Error processing image {img_num} on page {page_num+1}: {str(e)}")
                        continue
    print(f"Extracted {len(extracted_text)} characters.")
    return extracted_text

def detect_math_content(text):
    """Detect mathematical content with improved patterns."""
    math_indicators = [
        r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative|function|integral)\b',
        r'[=><≤≥≠\+\-\*\/\^]',
        r'\b[a-zA-Z]\([a-zA-Z0-9]+\)',
        r'\$.*?\$',
        r'\\[a-zA-Z]+'
    ]
    for pattern in math_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    for symbol in MATH_SYMBOLS.keys():
        if symbol in text:
            return True
    return False

def clean_text(text):
    """Enhanced text cleaning with improved patterns."""
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<=\w)([A-Z])', r'. \1', text)
    return text.strip()

def split_text_into_chunks(text, chunk_size=800):
    """Enhanced text splitting with sentence-aware chunking."""
    try:
        sentences = nltk.sent_tokenize(text)
    except:
        print("Warning: NLTK sentence tokenizer failed. Using simple sentence splitting.")
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_length = len(sentence)
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sent_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    if len(chunks) > 20:
        chunks = [chunks[i] for i in range(0, len(chunks), 2)]
    
    print(f"Split text into {len(chunks)} chunks")
    return chunks

def determine_reading_level(grade):
    """Determine reading level with age mapping."""
    age = grade + 6
    if 1 <= grade <= 3:
        return "lower", f"elementary (grade {grade}, age {age})"
    elif 4 <= grade <= 6:
        return "middle", f"middle school (grade {grade}, age {age})"
    else:
        return "higher", f"high school (grade {grade}, age {age})"

############################################
# Enhanced Summary Generation              #
############################################

def generate_summary(text_chunks, grade_level, duration, has_math=False):
    """Enhanced two-pass summary generation with word count control."""
    # Define word count targets based on duration
    word_targets = {
        10: (1300, 1500),
        20: (1500, 2600),
        30: (2600, 4000)
    }
    min_words, max_words = word_targets.get(duration, (1300, 1500))
    
    prompts = {
        "lower": {
            "standard": (
                "Create a simple summary for young students (grades 1-3) with these rules:\n"
                "1. Extract and highlight key insights from the document.\n"
                "2. Provide an overview of what the document is about at the start.\n"
                "3. Use short, complete sentences\n4. Start each point with '- '\n"
                "5. Avoid repeating information\n6. Include one hands-on activity\n"
                "7. Group related ideas together\n8. Ensure sentences end with proper punctuation\n"
                "Begin with '# Summary' followed by 1 line break."
            ),
            "math": (
                "Explain math concepts for young students (grades 1-3) with:\n"
                "1. Extract and highlight key insights from the document.\n"
                "2. Provide an overview of what the document is about at the start.\n"
                "3. Simple analogies\n4. Step-by-step explanations\n"
                "5. Real-world examples\n6. One practice activity\n"
                "Format: '# Summary' followed by bullet points."
            )
        },
        "middle": {
            "standard": (
                "Create a structured summary for middle school (grades 4-6) with:\n"
                "1. Extract and highlight key insights from the document.\n"
                "2. Provide an overview of what the document is about at the start.\n"
                "3. Clear section organization\n4. Complete sentences\n"
                "5. Avoid redundancy\n6. Include relevant examples\n"
                "7. One practical activity\nFormat:\n# Summary\n\n## Section\n- Points"
            ),
            "math": (
                "Explain math content for middle school (grades 4-6) with:\n"
                "1. Extract and highlight key insights from the document.\n"
                "2. Provide an overview of what the document is about at the start.\n"
                "3. Conceptual explanations\n4. Worked examples\n"
                "5. Practical applications\n6. Practice problem\n"
                "Format: '# Summary' with sections and bullets"
            )
        },
        "higher": {
            "standard": (
                "Create comprehensive summary for high school (grades 7-12) with:\n"
                "1. Extract and highlight key insights from the document.\n"
                "2. Provide an overview of what the document is about at the start.\n"
                "3. Thematic sections\n4. Academic vocabulary\n"
                "5. No repetition\n6. Real-world applications\n"
                "7. Research/analysis task\nFormat:\n# Summary\n\n## Topic\n- Points"
            ),
            "math": (
                "Explain advanced math for high school (grades 7-12) with:\n"
                "1. Extract and highlight key insights from the document.\n"
                "2. Provide an overview of what the document is about at the start.\n"
                "3. Formal definitions\n4. Proof sketches\n"
                "5. Practical applications\n6. Challenge problem\n"
                "Format: Structured sections with bullets"
            )
        }
    }
    
    # First pass with context tracking
    chunk_summaries = []
    previous_points = []
    max_context_points = 5
    
    for i, chunk in enumerate(text_chunks):
        context = "\nPrevious key points:\n- " + "\n- ".join(previous_points[-max_context_points:]) if previous_points else ""
        prompt = prompts[grade_level]["math" if has_math else "standard"] + context
        
        try:
            generated = model.generate(prompt + "\nText:\n" + chunk, max_length=1500, temperature=0.7)
            summary = generated[len(prompt)+7:].strip()
            
            bullets = [b.strip() for b in re.findall(r'-\s+(.+?)(?=\n-|\n\n|$)', summary)]
            if not bullets:
                bullets = [s.strip() for s in re.split(r'[.!?](?:\s|$)', summary) if s.strip()]
            
            previous_points.extend(bullets)
            chunk_summaries.extend(bullets)
            
            print(f"Processed chunk {i+1}/{len(text_chunks)} - {len(bullets)} points added")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            simple_lines = chunk.split('.')[:3]
            chunk_summaries.extend([line.strip() + '.' for line in simple_lines if line.strip()])
    
    # Second pass consolidation with word count control
    consolidation_prompt = (
        f"Create final {grade_level} level summary by:\n"
        "1. Removing duplicates\n2. Grouping related points\n"
        "3. Ensuring complete sentences\n4. Adding section headers\n"
        "5. Including relevant activity\n"
        "Format:\n# Summary\n\n## Main Topic\n- Points\n\n## Next Topic\n- Points"
    )
    
    try:
        final_summary = model.generate(
            consolidation_prompt + "\nPoints:\n- " + "\n- ".join(chunk_summaries),
            max_length=int(max_words * 1.4),  # 40% buffer for token-to-word conversion
            temperature=0.3
        )
        processed_summary = enhanced_post_process(
            final_summary[len(consolidation_prompt):].strip(),
            grade_level,
            has_math,
            min_words,
            max_words
        )
    except Exception as e:
        print(f"Consolidation error: {str(e)}")
        processed_summary = "# Summary\n\n"
        for i, point in enumerate(chunk_summaries[:20]):
            processed_summary += f"- {point}\n"
        processed_summary += "\n## Suggested Activity\n"
        processed_summary += create_activity(grade_level, has_math)
        processed_summary = processed_summary[:max_words*6]  # Approximate word limit
    
    return processed_summary

def enhanced_post_process(summary, grade_level, has_math, min_words, max_words):
    """Advanced post-processing with word count validation."""
    summary = re.sub(r'#+\s*Summary\s*#+', '# Summary', summary, flags=re.IGNORECASE)
    
    sections = {}
    current_section = "Main Points"
    sections[current_section] = []
    
    for line in summary.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('##'):
            current_section = line.strip('#').strip()
            sections[current_section] = []
        elif line.startswith('-'):
            sections[current_section].append(line)
    
    processed = ["# Summary"]
    seen_stems = set()
    
    for section, bullets in sections.items():
        if section != "Main Points":
            processed.append(f"\n## {section}")
        
        for bullet in bullets:
            if not re.search(r'[.!?]$', bullet):
                bullet = complete_sentence(bullet)
            
            stem_key = get_stemmed_key(bullet)
            if stem_key in seen_stems:
                continue
            seen_stems.add(stem_key)
            
            bullet = re.sub(r'^-\s*', '- ', bullet)
            if bullet[-1] not in {'.', '!', '?'}:
                bullet += '.'
            if bullet[2:3].isalpha():
                bullet = "- " + bullet[2].upper() + bullet[3:]
            
            processed.append(bullet)
    
    if not any("activity" in s.lower() for s in sections):
        processed.append("\n## Suggested Activity")
        processed.append(create_activity(grade_level, has_math))
    
    # Word count management
    full_text = ' '.join(processed)
    words = full_text.split()
    current_count = len(words)
    
    if current_count > max_words:
        print(f"Trimming from {current_count} to {max_words} words")
        keep_start = int(max_words * 0.2)
        keep_end = max_words - keep_start
        words = words[:keep_start] + words[-keep_end:]
    elif current_count < min_words:
        print(f"Note: Summary is {current_count} words (minimum target: {min_words})")
    
    return ' '.join(words[:max_words])

def create_activity(grade_level, has_math):
    """Generate context-appropriate activities."""
    activities = {
        "lower": {
            True: "- Math Game: Use small objects to practice basic operations.",
            False: "- Drawing Activity: Illustrate the main concepts visually."
        },
        "middle": {
            True: "- Math Challenge: Solve real-world problems using these concepts.",
            False: "- Experiment: Conduct a simple related science experiment."
        },
        "higher": {
            True: "- Research Project: Investigate practical applications.",
            False: "- Analysis Task: Compare different aspects critically."
        }
    }
    return activities[grade_level][has_math]

######################################
# Interface and Execution            #
######################################

output = Output()
uploader = FileUpload(accept='.pdf', multiple=False, description='Upload PDF')
grade_slider = IntSlider(min=1, max=12, step=1, value=6, description='Grade Level:')
duration_buttons = RadioButtons(
    options=[('10 minutes', 10), ('20 minutes', 20), ('30 minutes', 30)],
    description='Duration:',
    disabled=False
)
summarize_button = Button(description='Generate Summary', disabled=True)
summary_output = Output()

def on_upload_change(change):
    if change['type'] == 'change' and change['name'] == 'value' and uploader.value:
        with output:
            clear_output()
            print("File uploaded successfully!")
        summarize_button.disabled = False

uploader.observe(on_upload_change, names='value')

def on_summarize_clicked(b):
    if not uploader.value:
        with summary_output:
            print("Please upload a PDF file first.")
        return
    
    grade = grade_slider.value
    duration = duration_buttons.value
    grade_level, grade_name = determine_reading_level(grade)
    file_info = uploader.value[0]
    filename = file_info['name']
    
    with summary_output:
        clear_output()
        print(f"Processing {filename} for grade {grade} ({grade_name})...")
        
        try:
            pdf_path = os.path.join(os.getcwd(), filename)
            with open(pdf_path, 'wb') as f:
                f.write(file_info['content'])
            
            raw_text = extract_text_from_pdf(pdf_path)
            has_math = detect_math_content(raw_text)
            cleaned_text = clean_text(raw_text)
            chunks = split_text_into_chunks(cleaned_text)
            
            word_targets = {
                10: (1300, 1500),
                20: (1500, 2600),
                30: (2600, 4000)
            }
            min_words, max_words = word_targets.get(duration, (1300, 1500))
            print(f"\nGenerating {duration}-minute summary ({min_words}-{max_words} words)...")
            
            summary = generate_summary(chunks, grade_level, duration, has_math)
            word_count = len(summary.split())
            
            # Improved output file naming
            output_file = f"{filename.split('.')[0]}_grade{grade}_duration{duration}_summary.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print("\n✅ Summary Generated Successfully! ✅")
            print(f"Word count: {word_count} words")
            print("\nPreview:\n" + "\n".join(summary.split('\n')[:15]))
            print(f"\nFull summary saved to: {output_file}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

summarize_button.on_click(on_summarize_clicked)

# Display interface
display(VBox([
    Label(value="📚 Enhanced PDF Summarizer 🎓", 
          layout=Layout(justify_content='center', font_weight='bold', font_size='20px')),
    HBox([uploader, grade_slider]),
    HBox([duration_buttons, summarize_button]),
    output,
    summary_output
]))