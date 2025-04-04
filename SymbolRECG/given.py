
import re
import torch
import os
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ipywidgets import FileUpload, Button, IntSlider, RadioButtons, Output, VBox, HBox, Label, Layout
from IPython.display import display, clear_output

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Summarization Model
print("Loading DistilBART model for faster summarization...")
model_name = "sshleifer/distilbart-cnn-12-6"  # Smaller, faster model optimized for GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
print(f"Model loaded successfully on {device}!")

# Global variable to store the PDF path
pdf_path = None

# Set OCR resolution
OCR_RESOLUTION = 300

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'tesseract'  # Update this path if needed

# Map of common math symbols to their LaTeX representation - used only when specifically generating LaTeX output
MATH_SYMBOLS = {
    '∫': '\\int',     # Integral
    '∑': '\\sum',     # Summation
    '∏': '\\prod',    # Product
    '√': '\\sqrt',    # Square root
    '∞': '\\infty',   # Infinity
    '≠': '\\neq',     # Not equal
    '≤': '\\leq',     # Less than or equal
    '≥': '\\geq',     # Greater than or equal
    '±': '\\pm',      # Plus-minus
    '→': '\\to',      # Right arrow
    '∂': '\\partial', # Partial derivative
    '∇': '\\nabla',   # Nabla/del
    'π': '\\pi',      # Pi
    'θ': '\\theta',   # Theta
    'λ': '\\lambda',  # Lambda
    'μ': '\\mu',      # Mu
    'σ': '\\sigma',   # Sigma
    'ω': '\\omega',   # Omega
    'α': '\\alpha',   # Alpha
    'β': '\\beta',    # Beta
    'γ': '\\gamma',   # Gamma
    'δ': '\\delta',   # Delta
    'ε': '\\epsilon', # Epsilon
    '∈': '\\in',      # Element of
    '⊂': '\\subset',  # Subset
    '⊆': '\\subseteq',# Subset or equal
    '∪': '\\cup',     # Union
    '∩': '\\cap',     # Intersection
    '⟨': '\\langle',  # Left angle bracket
    '⟩': '\\rangle',  # Right angle bracket
}

def extract_text_from_pdf(pdf_path, detect_math=True):
    """Extract text from a PDF file, including OCR for images."""
    print(f"Extracting text from {pdf_path}...")
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"Processing page {page_num+1}/{len(pdf.pages)}...")
            
            # Extract regular text
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
            
            # Process images on the page using basic OCR
            if page.images:
                for img_num, img in enumerate(page.images):
                    try:
                        # Clamp the bounding box to page dimensions
                        x0 = max(0, img['x0'])
                        top = max(0, img['top'])
                        x1 = min(page.width, img['x1'])
                        bottom = min(page.height, img['bottom'])
                        bbox = (x0, top, x1, bottom)
                        
                        cropped = page.crop(bbox)
                        pil_image = cropped.to_image(resolution=OCR_RESOLUTION).original
                        
                        # Basic OCR without special processing
                        ocr_text = pytesseract.image_to_string(pil_image)
                        
                        if ocr_text.strip():
                            extracted_text += " " + ocr_text + "\n"
                    except Exception as e:
                        print(f"Error processing image {img_num} on page {page_num+1}: {str(e)}")
                        continue
    
    print(f"Extracted {len(extracted_text)} characters of text.")
    return extracted_text


def detect_math_content(text):
    """Detect if the document contains mathematical content."""
    math_indicators = [
        r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|differentiate|integrate|derivative|function)\b',
        r'[=><≤≥≠\+\-\*\/\^]',
        r'\b[a-zA-Z]\([a-zA-Z0-9]+\)',  # Function notation like f(x)
        r'\b[a-zA-Z]\s*=',  # Variable assignment
        r'\([a-zA-Z0-9\+\-\*\/\^]+\)',  # Expressions in parentheses
    ]
    
    for pattern in math_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check for known math symbols
    for symbol in MATH_SYMBOLS.keys():
        if symbol in text:
            return True
    
    return False

def clean_text(text):
    """Clean extracted text by removing unwanted spaces, characters, and common OCR errors."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors with periods and commas
    text = re.sub(r'\.(\w)', r'. \1', text)  # Fix period followed by word
    text = re.sub(r'\,(\w)', r', \1', text)  # Fix comma followed by word
    
    # Fix sentence spacing
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove repeated punctuation
    text = re.sub(r'([.!?,;])\1+', r'\1', text)
    
    # Remove strange characters that commonly appear in OCR
    text = re.sub(r'\(cid:\d+\)', '', text)
    
    # Fix paragraph breaks
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def split_text_into_chunks(text, chunk_size=1024):
    """Split text into chunks of a specified size, respecting paragraph breaks when possible."""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size and we already have content,
        # complete the current chunk and start a new one
        if current_length + len(paragraph) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # If a single paragraph exceeds chunk size, split it by sentences
        if len(paragraph) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if current_length + len(sentence) > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
        else:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1  # +1 for space

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def get_age_from_grade(grade):
    """Approximate age based on grade level."""
    return grade + 5

def determine_reading_level(grade):
    """Determine appropriate reading level based on grade."""
    if 1 <= grade <= 3:
        return "lower", f"elementary (grade {grade}, age {get_age_from_grade(grade)})"
    elif 4 <= grade <= 6:
        return "middle", f"middle elementary (grade {grade}, age {get_age_from_grade(grade)})"
    else:
        return "higher", f"secondary (grade {grade}, age {get_age_from_grade(grade)})"


def generate_summary(text_chunks, grade_level, duration, has_math=False):
    """Generate a summary based on grade level and duration with focus on key concepts."""
    # Enhanced prompts to focus on key concepts and age-appropriate language, with math options.
    # The middle-level math prompt has been enhanced to require detailed, step-by-step explanations
    # that result in a longer summary (e.g., 4000-4500 words for a 30-minute reading time).
    prompts = {
        "lower": {
            "standard": "Extract and summarize the main ideas and key points from the following text "
                        "for young elementary students (grades 1-3). Focus on the most important concepts, "
                        "use simple vocabulary, short sentences, and explain any difficult terms. "
                        "Organize the summary with clear headings and bullet points where appropriate: ",
            
            "math": "Create a simple summary of the following mathematical content for young elementary "
                    "students (grades 1-3). Explain the math concepts in very basic terms, using concrete "
                    "examples where possible. Break down equations into simple steps, and use visual "
                    "descriptions to help understanding. Focus on 'what' rather than 'why' for complex concepts: "
        },
        
        "middle": {
            "standard": "Create a well-structured educational summary of the following text "
                        "for middle elementary students (grades 4-6). Focus on key concepts, important facts, "
                        "and main ideas. Use clear examples and organize the content with appropriate "
                        "headings and subheadings to help with understanding: ",
            
            "math": "Create a detailed, student-friendly explanation of the following mathematical content for "
                    "middle elementary students (grades 4-6). For a 30-minute explanation, ensure the summary is "
                    "around 4000-4500 words. Explain key mathematical concepts and principles, simplify equations when possible, "
                    "and connect abstract ideas to everyday examples. Include step-by-step explanations for processes and definitions "
                    "of important terms. Use clear headings, bullet points, and verbal descriptions of diagrams where applicable: "
        },
        
        "higher": {
            "standard": "Create a comprehensive educational summary of the following text "
                        "for secondary/high school students. Extract the most important concepts, principles, "
                        "and facts. Maintain academic vocabulary but explain complex ideas clearly. "
                        "Organize the content logically with proper headings and subheadings: ",
            
            "math": "Create a comprehensive educational summary of the following mathematical content "
                    "for secondary/high school students. Preserve the mathematical notation where appropriate, "
                    "but provide clear explanations of theorems, formulas, and processes. Define specialized "
                    "terms, explain the significance of key concepts, and highlight connections between "
                    "different mathematical ideas: "
        }
    }
    
    prompt_type = "math" if has_math else "standard"
    prompt = prompts.get(grade_level, prompts["middle"]).get(prompt_type)
    
    # Define summary length based on time (in words)
    length_map = {
        10: (1300, 1500),
        20: (2500, 3000),
        30: (4000, 4500)
    }
    min_words, max_words = length_map.get(duration, (1300, 1500))
    
    print(f"Generating {duration}-minute summary ({min_words}-{max_words} words) for {grade_level} level...")
    print(f"Using {prompt_type} prompt type based on content detection...")
    total_summary = ""
    
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)}...")
        input_text = prompt + chunk
            
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        min_tokens = min(750, int(min_words / len(text_chunks) * 1.3))
        max_tokens = min(1024, int(max_words / len(text_chunks) * 1.3))
        
        summary_ids = model.generate(
            input_ids,
            num_beams=2,  # Reduced for speed on GPU
            min_length=min_tokens,
            max_length=max_tokens,
            length_penalty=1.5,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        total_summary += chunk_summary + " "
        
        current_words = len(total_summary.split())
        print(f"Current summary: {current_words} words")
        if current_words >= max_words:
            break
    
    return post_process_summary(total_summary, grade_level, min_words, max_words, has_math)

def post_process_summary(summary, grade_level, min_words, max_words, has_math=False):
    """Post-process the generated summary to improve readability and structure it with bullet points."""
    # 1. Remove duplicate sentences (same as before)
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    unique_sentences = []
    seen_sentences = set() # Use a set for faster duplicate checking

    for sentence in sentences:
        sentence = sentence.strip()
        # Skip empty sentences or very short ones
        if len(sentence) < 10: # Increased minimum length to avoid fragments
            continue

        # Normalize for comparison (lowercase, remove punctuation)
        normalized_sentence = ''.join(filter(str.isalnum, sentence.lower()))

        # Check for near duplicates (using simplified check for this example)
        is_duplicate = False
        if normalized_sentence in seen_sentences:
            is_duplicate = True
        else:
            # Check for high overlap with recent sentences if needed (more robust check)
            for existing in unique_sentences[-5:]: # Check last 5 sentences
                existing_normalized = ''.join(filter(str.isalnum, existing.lower()))
                if len(normalized_sentence) > 0 and len(existing_normalized) > 0:
                    # Simple substring check or more complex similarity could go here
                    if normalized_sentence in existing_normalized or existing_normalized in normalized_sentence:
                        # Basic check if one is contained in the other - adjust if needed
                        if abs(len(normalized_sentence) - len(existing_normalized)) < 20: # Only if lengths are similar
                            is_duplicate = True
                            break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
            seen_sentences.add(normalized_sentence)

    # 2. Structure the summary with bullet points
    structured_summary_points = []
    for sentence in unique_sentences:
        # Prepend each sentence with a markdown bullet point
        structured_summary_points.append(f"* {sentence}")

    # Join points with newlines
    processed_summary = "\n".join(structured_summary_points)

    # 3. Add an appropriate title based on context
    title = ""
    if has_math:
        if grade_level == "lower":
            title = "# Simple Math Ideas"
        elif grade_level == "middle":
            title = "# Key Mathematical Concepts"
        else:  # higher
            title = "# Mathematical Concepts Overview"
    else:
        if grade_level == "lower":
            title = "# Main Ideas for Young Readers"
        elif grade_level == "middle":
            title = "# Summary of Key Points"
        else: # higher
            title = "# Comprehensive Summary Points"

    final_summary = title + "\n\n" + processed_summary

    # 4. Ensure summary is within word count limits (apply to the final structured text)
    words = final_summary.split() # Split final text including title and bullets
    current_word_count = len(words)

    if current_word_count > max_words:
        # Trim words from the end of the bulleted list content, keeping the title
        # Estimate how many bullet points to keep
        avg_words_per_point = (current_word_count - len(title.split())) / len(structured_summary_points) if structured_summary_points else 1
        points_to_keep = int((max_words - len(title.split())) / avg_words_per_point)
        
        trimmed_points = structured_summary_points[:max(1, points_to_keep)] # Keep at least one point
        processed_summary = "\n".join(trimmed_points)
        final_summary = title + "\n\n" + processed_summary
        
        # Final trim if still slightly over
        words = final_summary.split()
        if len(words) > max_words:
            final_summary = ' '.join(words[:max_words]) + "..." # Add ellipsis to indicate trim

        print(f"Trimmed summary to approximately {max_words} words using point structure.")
        current_word_count = len(final_summary.split()) # Recalculate count

    elif current_word_count < min_words:
        print(f"Warning: The generated summary ({current_word_count} words) is shorter than the desired minimum length ({min_words} words).")

    return final_summary

def save_summary(summary, filename, grade_level_name, duration):
    """Save the generated summary to a file."""
    output_filename = f"{filename.split('.')[0]}_{grade_level_name}_{duration}min_summary.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    return output_filename

# Create widgets for the interface
output = Output()
uploader = FileUpload(accept='.pdf', multiple=False, description='Upload PDF')
grade_slider = IntSlider(min=1, max=12, step=1, value=6, description='Grade Level:')
duration_buttons = RadioButtons(
    options=[('10 minutes', 10), ('20 minutes', 20), ('30 minutes', 30)],
    description='Duration:',
    disabled=False
)
upload_button = Button(description='Upload PDF')
summarize_button = Button(description='Generate Summary', disabled=True)
summary_output = Output()

# File upload handler using change event
def on_upload_change(change):
    global pdf_path
    if change['type'] == 'change' and change['name'] == 'value' and uploader.value:
        with output:
            clear_output()
            print("File uploaded successfully!")
        
        # Access the first (and only) file's dictionary
        file_info = uploader.value[0]
        filename = file_info['name']
        content = file_info['content']

        pdf_path = os.path.join(os.getcwd(), filename)
        with open(pdf_path, 'wb') as f:
            f.write(content)

        print(f"File saved at: {pdf_path}")
        summarize_button.disabled = False

uploader.observe(on_upload_change, names='value')

# Manual upload button handler
def on_upload_clicked(b):
    global pdf_path
    if uploader.value:
        with output:
            clear_output()
            file_info = uploader.value[0]
            filename = file_info['name']
            content = file_info['content']
            
            print(f"Processing file: {filename} (size: {len(content)} bytes)")
            pdf_path = os.path.join(os.getcwd(), filename)
            with open(pdf_path, 'wb') as f:
                f.write(content)
            print(f"File uploaded and saved as: {pdf_path}")
        summarize_button.disabled = False
    else:
        with output:
            clear_output()
            print("No files uploaded yet.")

upload_button.on_click(on_upload_clicked)

# Generate summary when the button is clicked
def on_summarize_clicked(b):
    global pdf_path
    if pdf_path is None:
        with summary_output:
            clear_output()
            print("No PDF file has been uploaded. Please upload a file first.")
        return
    
    grade = grade_slider.value
    duration = duration_buttons.value
    grade_level, grade_level_name = determine_reading_level(grade)
    
    with summary_output:
        clear_output()
        print(f"Starting summarization process for grade {grade} ({grade_level_name})...")
        print(f"Summary for {duration} minute reading time...")
        try:
            raw_text = extract_text_from_pdf(pdf_path, detect_math=False)
            
            has_math = detect_math_content(raw_text)
            if has_math:
                print("Mathematical content detected! Optimizing processing...")
            
            cleaned_text = clean_text(raw_text)
            text_chunks = split_text_into_chunks(cleaned_text)
            print(f"Text split into {len(text_chunks)} chunks for summarization.")
            
            summary = generate_summary(text_chunks, grade_level, duration, has_math=has_math)
            filename = os.path.basename(pdf_path)
            output_file = save_summary(summary, filename, grade_level_name, duration)
            
            print("\n🔹 **Generated Summary 🔹\n")
            print(f"Preview (first 300 words):")
            print('-' * 80)
            preview_words = summary.split()[:300]
            print(' '.join(preview_words) + "...")
            print('-' * 80)
            print(f"\nFull summary ({len(summary.split())} words) saved to: {output_file}")
        
        except Exception as e:
            print(f"Error during summarization: {str(e)}")

summarize_button.on_click(on_summarize_clicked)

# Create and display the layout
display(VBox([
    Label(value="📚 PDF Summarizer for Education 📚", 
          layout=Layout(display='flex', justify_content='center', font_weight='bold', font_size='20px')),
    uploader,
    upload_button,
    HBox([grade_slider, duration_buttons]),
    summarize_button,
    output,
    summary_output
]))