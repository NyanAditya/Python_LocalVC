# --- Start of Combined Code ---
import re
import torch
import os
import pdfplumber
# import pytesseract # REMOVED
import easyocr      # ADDED
from PIL import Image
import numpy as np  # ADDED (needed for image conversion for EasyOCR)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ipywidgets import FileUpload, Button, IntSlider, RadioButtons, Output, VBox, HBox, Label, Layout
from IPython.display import display, clear_output
import traceback

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Summarization Model
print("Loading DistilBART model for faster summarization...")
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
print(f"Model loaded successfully on {device}!")

# --- Initialize EasyOCR Reader ---
print("Initializing EasyOCR Reader (this may take a moment on first run)...")
# Use gpu=True if torch/CUDA is available and configured, otherwise False
# Add other languages if needed, e.g., ['en', 'fr']
try:
    ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
    print("✅ EasyOCR Reader initialized successfully.")
except Exception as e:
    print(f"⚠️ Error initializing EasyOCR: {e}")
    print("   Will attempt to proceed without OCR on images.")
    ocr_reader = None # Set reader to None if initialization fails
# --- End EasyOCR Initialization ---


# Global variable to store the PDF path
pdf_path = None

# Set OCR resolution (still relevant for image quality fed to EasyOCR)
OCR_RESOLUTION = 300

# --- REMOVED TESSERACT CONFIGURATION BLOCK ---

# Map of common math symbols (kept for math detection)
MATH_SYMBOLS = {
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty', '≠': '\\neq',
    '≤': '\\leq', '≥': '\\geq', '±': '\\pm', '→': '\\to', '∂': '\\partial', '∇': '\\nabla',
    'π': '\\pi', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega',
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon', '∈': '\\in',
    '⊂': '\\subset', '⊆': '\\subseteq', '∪': '\\cup', '∩': '\\cap', '⟨': '\\langle', '⟩': '\\rangle',
}

# --- MODIFIED extract_text_from_pdf ---
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file, including OCR for images using EasyOCR."""
    global ocr_reader # Access the global reader
    print(f"Extracting text from {pdf_path}...")
    extracted_text = ""
    if ocr_reader is None:
        print("⚠️ EasyOCR Reader not available. Skipping OCR on images.")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"PDF has {total_pages} pages.")
            for page_num, page in enumerate(pdf.pages):
                print(f"Processing page {page_num+1}/{total_pages}...")

                # Extract regular text
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"

                # Process images on the page using EasyOCR if available
                if page.images and ocr_reader: # Only process if reader is initialized
                    print(f"  Found {len(page.images)} image areas on page {page_num+1}. Processing OCR with EasyOCR...")
                    for img_num, img in enumerate(page.images):
                        try:
                            # Clamp the bounding box
                            x0, top, x1, bottom = max(0, img['x0']), max(0, img['top']), min(page.width, img['x1']), min(page.height, img['bottom'])
                            if x1 <= x0 or bottom <= top: continue # Skip invalid bbox

                            bbox = (x0, top, x1, bottom)
                            cropped = page.crop(bbox)
                            pil_image = cropped.to_image(resolution=OCR_RESOLUTION).original

                            # Convert PIL image to NumPy array needed by EasyOCR
                            img_np = np.array(pil_image)

                            # Perform OCR using EasyOCR
                            # detail=0 returns just text, paragraph=True tries to combine lines
                            ocr_results = ocr_reader.readtext(img_np, detail=0, paragraph=True)

                            # Join the detected text paragraphs/lines
                            ocr_text = "\n".join(ocr_results)

                            if ocr_text and ocr_text.strip():
                                # print(f"    EasyOCR Result (img {img_num+1}): '{ocr_text[:50].strip()}...'") # Debug
                                extracted_text += " " + ocr_text.strip() + "\n"

                        # Specific exceptions for EasyOCR or image processing can be added here if needed
                        except Exception as e:
                            print(f"  Error processing image {img_num+1} with EasyOCR on page {page_num+1}: {type(e).__name__} - {str(e)}")
                            continue # Skip to next image
                elif page.images and not ocr_reader:
                     print(f"  Skipping {len(page.images)} image areas on page {page_num+1} (EasyOCR unavailable).")


            print(f"Finished processing pages. Extracted approx. {len(extracted_text)} characters.")

    except Exception as e:
        print(f"Error opening or processing PDF '{pdf_path}': {str(e)}")
        raise

    return extracted_text
# --- END MODIFIED extract_text_from_pdf ---


# --- detect_math_content, clean_text, split_text_into_chunks functions remain the same ---
def detect_math_content(text):
    """Detect if the document contains mathematical content using refined indicators."""
    if not text: return False
    strong_indicators = [
        r'[\+\-\*\/=<>≤≥≠±]', r'∫|∑|∏|√|∞|∂|∇|∈|⊂|⊆|∪|∩',
        r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|matrix|vector|integral|derivative)\b',
        r'\^\s*\{?[0-9\.\-a-zA-Z]+\}?', r'_\s*\{?[0-9a-zA-Z]+\}?', r'\\frac|\\sqrt|\\sum|\\int|\\lim',
        r'[a-zA-Z]\s*\(.*\)\s*=', r'\b(sin|cos|tan|log|exp)\b\(',
    ]
    for pattern in strong_indicators:
        if re.search(pattern, text, re.IGNORECASE): return True
    for symbol in MATH_SYMBOLS.keys():
        if symbol in text: return True
    if re.search(r'\b[a-zA-Z]+\([a-zA-Z0-9,\s]+\)', text): return True
    return False

def clean_text(text):
    """Clean extracted text by removing unwanted spaces, characters, and fixing common OCR/extraction issues."""
    if not text: return ""
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'^\s+|\s+$', '', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])(?=\w)', r'\1 ', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'[\f\v]', '', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_text_into_chunks(text, chunk_size=700, overlap=100): # Kept smaller chunk size
    """Split text into potentially overlapping chunks of a specified token size, respecting sentences."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    if total_tokens == 0: return []
    chunks = []
    start_token = 0
    while start_token < total_tokens:
        end_token = min(start_token + chunk_size, total_tokens)
        current_chunk_tokens = tokens[start_token:end_token]
        current_chunk_text = tokenizer.decode(current_chunk_tokens)
        if end_token < total_tokens and (total_tokens - start_token) > chunk_size:
             search_area_start = min(start_token + chunk_size + overlap, total_tokens)
             search_text = tokenizer.decode(tokens[start_token:search_area_start])
             sentence_end_indices = [m.start() for m in re.finditer(r'[.!?]\s+', search_text)]
             if sentence_end_indices:
                  best_end_pos = -1
                  for idx in reversed(sentence_end_indices):
                      estimated_tokens = len(tokenizer.encode(search_text[:idx+1], add_special_tokens=False))
                      if estimated_tokens <= chunk_size:
                          best_end_pos = idx + 1
                          break
                  if best_end_pos != -1:
                      adjusted_chunk_text = search_text[:best_end_pos]
                      adjusted_tokens = tokenizer.encode(adjusted_chunk_text, add_special_tokens=False)
                      if len(adjusted_tokens) > 0:
                         current_chunk_text = adjusted_chunk_text
                         end_token = start_token + len(adjusted_tokens)
        chunks.append(current_chunk_text)
        next_start_token = end_token - overlap
        start_token = max(start_token + 1, next_start_token)
        if start_token >= end_token: start_token = end_token
    chunks = [chunk for chunk in chunks if chunk.strip()]
    print(f"Split text into {len(chunks)} chunks (target size: {chunk_size} tokens, overlap: {overlap} tokens).")
    return chunks

# --- get_age_from_grade, determine_reading_level, generate_summary, post_process_summary, save_summary functions remain the same ---
def get_age_from_grade(grade):
    return grade + 5

def determine_reading_level(grade):
    grade = int(grade); age = get_age_from_grade(grade)
    if 1 <= grade <= 3: return "lower", f"early_elementary_gr{grade}_age{age}"
    elif 4 <= grade <= 6: return "middle", f"late_elementary_gr{grade}_age{age}"
    elif 7 <= grade <= 9: return "higher", f"middle_school_gr{grade}_age{age}"
    else: return "higher", f"high_school_gr{grade}_age{age}"

def generate_summary(text_chunks, grade_level, duration, has_math=False):
    level_description, _ = determine_reading_level(grade_slider.value)
    prompts = { "lower": { "standard": f"Summarize the key points from this text for a {get_age_from_grade(grade_slider.value)}-year-old (Grade {grade_slider.value}). Use simple words and short sentences. Focus on the main ideas. Explain any big words simply:\n\n", "math": f"Explain the main math ideas in this text simply for a {get_age_from_grade(grade_slider.value)}-year-old (Grade {grade_slider.value}). Describe any steps clearly. Use easy words. What are the most important math things mentioned?:\n\n" }, "middle": { "standard": f"Create an educational summary of this text for a student in Grade {grade_slider.value} (around {get_age_from_grade(grade_slider.value)} years old). Identify the main concepts, important facts, and key takeaways. Use clear language and provide context or simple examples where helpful:\n\n", "math": f"Create an educational summary explaining the mathematical concepts in this text for a Grade {grade_slider.value} student ({get_age_from_grade(grade_slider.value)} years old). Define key terms, explain formulas or processes step-by-step if possible, and state the main mathematical points clearly:\n\n" }, "higher": { "standard": f"Generate a concise, comprehensive summary of the following text suitable for a Grade {grade_slider.value} student ({get_age_from_grade(grade_slider.value)} years old). Focus on the core arguments, significant findings, key concepts, and principles. Maintain clarity and accuracy:\n\n", "math": f"Generate a comprehensive summary of the mathematical content in this text for a Grade {grade_slider.value} student ({get_age_from_grade(grade_slider.value)} years old). Clearly explain the main theorems, definitions, formulas, and methodologies discussed. Highlight the significance and application of the concepts presented:\n\n" } }
    prompt_type = "math" if has_math else "standard"
    base_prompt = prompts.get(level_description, prompts["middle"]).get(prompt_type, prompts["middle"]["standard"])
    words_per_minute_map = { "lower": 100, "middle": 130, "higher": 150 }; wpm = words_per_minute_map.get(level_description, 130)
    target_words = wpm * duration; min_words = int(target_words * 0.85); max_words = int(target_words * 1.15)
    avg_tokens_per_word = 1.4; model_max_gen_length = model.config.max_length
    min_tokens_per_chunk = max(30, int((min_words * avg_tokens_per_word * 0.8) / len(text_chunks)))
    max_tokens_per_chunk = min(model_max_gen_length // 2, int((max_words * avg_tokens_per_word * 1.2) / len(text_chunks)))
    print(f"Generating {duration}-minute summary ({min_words}-{max_words} target words) for {level_description} level ({'math focused' if has_math else 'standard'})."); print(f"Model max generation length: {model_max_gen_length}. Targeting {min_tokens_per_chunk}-{max_tokens_per_chunk} output tokens per chunk.")
    total_summary = ""; total_processed_words = 0
    if device == "cuda":
        try: torch.cuda.empty_cache(); print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        except Exception as e: print(f"Could not check GPU memory: {e}")
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)}...")
        input_text = base_prompt + chunk
        try: input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=model.config.max_position_embeddings, truncation=True).to(device)
        except Exception as e: print(f"  Error tokenizing chunk {i+1}: {e}"); continue
        if total_processed_words >= max_words * 1.1: print("Reached target word count estimate. Stopping generation early."); break
        try:
            current_min_length = min(min_tokens_per_chunk, max_tokens_per_chunk - 10); current_max_length = max(current_min_length + 10, max_tokens_per_chunk)
            summary_ids = model.generate( input_ids, num_beams=4, min_length=current_min_length, max_length=current_max_length, length_penalty=1.5, early_stopping=True, no_repeat_ngram_size=3 )
            chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            total_summary += chunk_summary + "\n\n"; total_processed_words = len(total_summary.split())
            print(f"  Chunk {i+1} summary generated. Total words approx: {total_processed_words}")
            if device == "cuda" and i % 5 == 0: torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError: print(f"  ❌ CUDA Out of Memory Error on chunk {i+1}!"); print("     Try reducing chunk_size or overlap."); torch.cuda.empty_cache(); break
        except Exception as e: print(f"  Error generating summary for chunk {i+1}: {type(e).__name__} - {str(e)}"); continue
    print(f"Finished generating raw summary. Total words approx: {total_processed_words}")
    return post_process_summary(total_summary, level_description, min_words, max_words, has_math)

def post_process_summary(summary, grade_level_desc, min_words, max_words, has_math=False):
    if not summary or not summary.strip(): print("Warning: Raw summary was empty before post-processing."); return "Could not generate a valid summary."
    potential_points = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n+', summary) if s.strip()]
    if not potential_points: print("Warning: No points found after splitting the raw summary."); return "Summary content was empty or could not be structured."
    unique_points = []; seen_points_normalized = set(); min_point_length = 15
    for point in potential_points:
        normalized_point = ''.join(filter(str.isalnum, point.lower()))
        if not normalized_point or len(point) < min_point_length: continue
        if normalized_point not in seen_points_normalized: unique_points.append(point); seen_points_normalized.add(normalized_point)
    if not unique_points: print("Warning: No valid points remained after filtering duplicates/short fragments."); return "Summary content was filtered out during processing."
    structured_summary_points = []
    for point in unique_points:
        if re.match(r'^[\*\-\•]\s+', point): structured_summary_points.append(point)
        else: structured_summary_points.append(f"* {point}")
    title = ""; grade_val = grade_slider.value; age = get_age_from_grade(grade_val)
    if has_math: title = f"# Math Summary (Grade {grade_val} / Age {age})"
    else: title = f"# Summary (Grade {grade_val} / Age {age})"
    processed_summary = title + "\n\n" + "\n".join(structured_summary_points)
    words = processed_summary.split(); current_word_count = len(words)
    if current_word_count > max_words:
        print(f"Combined summary ({current_word_count} words) exceeds target ({max_words}). Trimming points...")
        points_to_keep = []; word_count_so_far = len(title.split())
        for point in structured_summary_points:
             point_word_count = len(point.split())
             if word_count_so_far + point_word_count <= max_words: points_to_keep.append(point); word_count_so_far += point_word_count
             else:
                 remaining_words = max_words - word_count_so_far
                 if remaining_words > 10:
                     point_words = point.split(); prefix = "";
                     if point_words[0] in ['*', '-', '•']: prefix = point_words[0] + " "; point_words = point_words[1:]
                     truncated_point_text = ' '.join(point_words[:remaining_words])
                     points_to_keep.append(f"{prefix}{truncated_point_text}..."); word_count_so_far += remaining_words + 1
                 break
        processed_summary = title + "\n\n" + "\n".join(points_to_keep); final_word_count = len(processed_summary.split())
        print(f"Trimmed summary to {final_word_count} words.")
    elif current_word_count < min_words: print(f"Warning: Final summary ({current_word_count} words) is shorter than the desired minimum ({min_words} words).")
    return processed_summary

def save_summary(summary, filename, grade_level_name, duration):
    base_filename = os.path.splitext(filename)[0]; safe_basename = re.sub(r'[^\w\-]+', '_', base_filename)
    output_filename = f"{safe_basename}_{grade_level_name}_{duration}min_summary.txt"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f: f.write(summary)
        print(f"Summary successfully saved to: {output_filename}"); return output_filename
    except Exception as e: print(f"Error saving summary to file '{output_filename}': {str(e)}"); return None

# --- UI Widgets (remain the same) ---
output = Output()
uploader = FileUpload( accept='.pdf', multiple=False, description='Upload PDF', layout=Layout(width='auto'), button_style='primary' )
grade_slider = IntSlider(min=1, max=12, step=1, value=8, description='Grade Level:', continuous_update=False)
duration_buttons = RadioButtons( options=[('10 min read', 10), ('20 min read', 20), ('30 min read', 30)], value=10, description='Target Time:', disabled=False, layout=Layout(width='auto') )
summarize_button = Button( description='Generate Summary', disabled=True, button_style='success', tooltip='Click to generate the summary after uploading a PDF', icon='check', layout=Layout(width='auto', margin='10px 0 0 0') )
summary_output = Output()

# --- Event Handlers (remain the same, still use global pdf_path) ---
def on_upload_change(change):
    global pdf_path
    summarize_button.disabled = True; pdf_path = None
    if change['new']:
        with output:
            clear_output(wait=True)
            try:
                file_info = change['new'][0]; filename = file_info['name']; content = file_info['content']
                print(f"Processing uploaded file: '{filename}' ({len(content)} bytes)")
                temp_pdf_path = os.path.join(os.getcwd(), filename)
                with open(temp_pdf_path, 'wb') as f: f.write(content)
                pdf_path = temp_pdf_path
                print(f"✅ File successfully saved locally as: {pdf_path}")
                summarize_button.disabled = False
            except Exception as e: print(f"❌ Error saving/processing uploaded file: {e}"); print("\nUpload Error Traceback:"); traceback.print_exc()
    elif not change['new'] and change['old']:
         with output: clear_output(wait=True); print("File selection cleared.")
uploader.observe(on_upload_change, names='value')

def on_summarize_clicked(b):
    global pdf_path, ocr_reader # Make sure ocr_reader is accessible if needed inside
    summarize_button.disabled = True; summarize_button.description = 'Processing...'; summarize_button.icon = 'hourglass-half'
    if pdf_path is None or not os.path.exists(pdf_path):
        with summary_output: clear_output(wait=True); print("⚠️ Error: No valid PDF file path found. Please upload a PDF file again.")
        summarize_button.disabled = False; summarize_button.description = 'Generate Summary'; summarize_button.icon = 'check'; return
    grade = grade_slider.value; duration = duration_buttons.value
    level_description, grade_level_name_for_file = determine_reading_level(grade)
    with summary_output:
        clear_output(wait=True); print(f"🚀 Starting summarization process for: {os.path.basename(pdf_path)}"); print(f"Target: Grade {grade} ({level_description}), {duration} min read time."); print("-" * 30)
        output_file = None # Initialize output_file
        summary = ""      # Initialize summary
        try:
            raw_text = extract_text_from_pdf(pdf_path) # Calls the modified version with EasyOCR
            if not raw_text or not raw_text.strip(): summary = "Error: No text extracted from PDF."
            else:
                has_math = detect_math_content(raw_text); print(f"Mathematical content detected: {'Yes' if has_math else 'No'}")
                print("Cleaning extracted text..."); cleaned_text = clean_text(raw_text)
                if not cleaned_text: summary = "Error: Cleaned text is empty."
                else:
                    print("Splitting text into manageable chunks for the model...")
                    text_chunks = split_text_into_chunks(cleaned_text, chunk_size=700, overlap=100)
                    if not text_chunks: summary = "Error: Text splitting resulted in no chunks."
                    else:
                        print("Generating summary using the AI model...")
                        summary = generate_summary(text_chunks, level_description, duration, has_math=has_math)
                        print("\nSaving the generated summary...")
                        filename = os.path.basename(pdf_path)
                        output_file = save_summary(summary, filename, grade_level_name_for_file, duration)

            print("\n" + "="*10 + " Summary Generation Complete " + "="*10)
            if summary.startswith("Error:"): print(f"\n❌ {summary}")
            elif output_file:
                print(f"\n✅ Full summary saved to: {output_file}"); print("\n📄 Summary Preview (first ~200 words):"); print('-' * 40)
                preview_words = summary.split()[:200]; print(' '.join(preview_words) + ("..." if len(summary.split()) > 200 else "")); print('-' * 40)
            else: print("\n⚠️ Summary generated but failed to save to file."); print("\n📄 Summary Content:"); print('-' * 40); print(summary); print('-' * 40)
        # Removed specific TesseractNotFoundError catch block
        except Exception as e: print(f"\n❌ An unexpected critical error occurred during summarization:"); print(f"Error Type: {type(e).__name__}"); print(f"Error Details: {str(e)}"); print("\nTraceback:"); traceback.print_exc()
        finally:
            summarize_button.disabled = False; summarize_button.description = 'Generate Summary'; summarize_button.icon = 'check'
            if device == "cuda":
                try: torch.cuda.empty_cache(); print(f"\nFinal GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                except Exception as e: print(f"Could not clear/check final GPU memory: {e}")
summarize_button.on_click(on_summarize_clicked)

# --- Display UI (remain the same) ---
controls = VBox([ HBox([grade_slider, duration_buttons], layout=Layout(justify_content='space-around')), summarize_button ])
app_layout = VBox([ Label(value="📚 PDF Summarizer for Education 📚", layout=Layout(display='flex', justify_content='center', font_weight='bold', font_size='20px', margin='10px')), Label(value="1. Upload a PDF document:", layout=Layout(margin='5px 0 0 0')), uploader, output, Label(value="2. Select Target Audience and Reading Time:", layout=Layout(margin='15px 0 0 0')), controls, Label(value="3. Summary Output:", layout=Layout(margin='15px 0 0 0')), summary_output ], layout=Layout(border='1px solid #ccc', padding='15px', width='80%'))
display(app_layout)

# --- End of Combined Code ---