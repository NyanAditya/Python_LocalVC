import os
import re
# pdfplumber # Not strictly needed if only using fitz, but keep if used somewhere
import pytesseract
from PIL import Image
import numpy as np
import torch
from nltk.stem import PorterStemmer
import nltk
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback
import tempfile
import io
from flask import Flask, request, render_template, jsonify, send_from_directory

# --- NLTK Download ---
try:
    print("Checking/downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)
    print("NLTK punkt check complete.")
except Exception as e:
    print(f"Warning: Could not download NLTK punkt resource automatically. Error: {e}")
    # The script will try to fall back later if needed

# --- Configuration & Model Loading (Load ONCE when Flask starts) ---
print("Starting Flask App Setup...")

# Tesseract Path
tesseract_cmd_path = None
tesseract_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_cmd_path = path
        print(f"Using Tesseract at: {path}")
        break
if not tesseract_cmd_path:
    print("Warning: Tesseract executable not found. OCR will be unavailable.")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# OCR Global Flag (will be set per request)
# We don't need a global flag here, pass it as parameter or check within function context
# USE_IMAGE_OCR = False # Remove this global flag
OCR_RESOLUTION = 300

# Load Models
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" # Or your desired model
EMBEDDER_MODEL_NAME = 'all-MiniLM-L6-v2'
tokenizer = None
model = None
embedder = None
stemmer = PorterStemmer() # Can be loaded anytime

try:
    print(f"Loading LLM model: {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME).to(device)
    print("LLM Model loaded successfully!")
except Exception as e:
    print(f"FATAL ERROR: Could not load LLM model {LLM_MODEL_NAME}: {e}")
    # Optionally exit or handle appropriately if models are essential
    # exit(1)

try:
    print(f"Loading Sentence Embedder: {EMBEDDER_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME)
    print("Embedder loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load Sentence Transformer {EMBEDDER_MODEL_NAME}: {e}")
    # Decide if the app can run without the embedder (e.g., disable deduplication)

# Check if essential models loaded
if not tokenizer or not model:
     print("Essential models (tokenizer/LLM) failed to load. Exiting.")
     exit(1)
if not embedder:
     print("Warning: Embedder failed to load. Semantic deduplication will be disabled.")

print("Model loading complete.")

# Math Symbols (Keep as before)
MATH_SYMBOLS = {
    '∫': '\\int', '∑': '\\sum', '∏': '\\prod', '√': '\\sqrt', '∞': '\\infty',
    '≠': '\\neq', '≤': '\\leq', '≥': '\\geq', '±': '\\pm', '→': '\\to',
    '∂': '\\partial', '∇': '\\nabla', 'π': '\\pi', 'θ': '\\theta',
    'λ': '\\lambda', 'μ': '\\mu', 'σ': '\\sigma', 'ω': '\\omega',
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon'
}

# --- Utility Functions (Copied from previous version, ensure NO Streamlit usage) ---
# --- Ensure all functions below are correctly defined as in your final CLI version ---
# --- Remove any `st.session_state` or `st.warning/error` calls inside these ---
# --- Replace them with standard `print` for logging ---

def get_stemmed_key(sentence, num_words=5):
    words = re.findall(r'\w+', sentence.lower())[:num_words]
    return ' '.join([stemmer.stem(word) for word in words])

def complete_sentence(fragment):
    if not model or not tokenizer:
        print("Warning: LLM not loaded, cannot complete sentence.")
        return fragment + "."
    prompt = f"Complete this sentence to make it grammatically correct and meaningful. Keep it concise. Fragment: '{fragment}'\nCompleted sentence:"
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        completed_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completed_part = completed_full[len(prompt):].strip()
        if completed_part and len(completed_part) > 2:
            if re.match(r'^[A-Za-z0-9\s]', completed_part):
                completed_part = re.sub(r'<\|eot_id\|>$', '', completed_part).strip()
                if completed_part[-1].isalnum(): completed_part += '.'
                return completed_part
            else:
                 print(f"Warning: Unusual start to completed sentence: '{completed_part}'. Returning fragment.")
                 return fragment + "."
        else:
            print(f"Warning: LLM completion failed/empty for fragment: '{fragment}'. Returning fragment.")
            return fragment + "."
    except Exception as e:
        print(f"Error during sentence completion for '{fragment}': {e}")
        return fragment + "."

def preprocess_image_for_math_ocr(image):
    if image.mode != 'L': image = image.convert('L')
    image_array = np.array(image)
    threshold = np.mean(image_array) * 0.9
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

# Modified to accept ocr_enabled flag
def extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=False):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    print(f"Extracting text from {pdf_path} (OCR Enabled: {ocr_enabled})...")
    extracted_text = ""
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        for page_num, page in enumerate(doc):
            print(f"Processing page {page_num+1}/{num_pages}...")
            text = page.get_text("text")
            if text: extracted_text += text + "\n"

            if ocr_enabled and tesseract_cmd_path and page.get_images(full=True):
                print(f"  - Page {page_num+1}: Found images, performing OCR...")
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        processed_pil_image = preprocess_image_for_math_ocr(pil_image) if detect_math else pil_image
                        ocr_text = pytesseract.image_to_string(processed_pil_image, config='--psm 6 --oem 3')
                        if ocr_text.strip():
                            if detect_math:
                                for symbol, latex in MATH_SYMBOLS.items():
                                    ocr_text = ocr_text.replace(symbol, f" {latex} ")
                            extracted_text += " [OCR_IMG] " + ocr_text.strip() + " [/OCR_IMG]\n"
                    except pytesseract.TesseractNotFoundError:
                         print("Error: Tesseract not found during OCR (should have been checked earlier). Disabling OCR for this request.")
                         ocr_enabled = False # Disable for remainder of this specific extraction
                         break # Stop trying OCR for this page
                    except Exception as e:
                        print(f"Error processing image {img_index} on page {page_num+1}: {str(e)}")
        doc.close()
        # Header/footer removal
        lines = extracted_text.split('\n')
        if len(lines) > 2:
            if len(lines[0].strip()) < 15 and re.match(r'^[\s\d\W]*$', lines[0].strip()): lines = lines[1:]
            if len(lines) > 1 and len(lines[-1].strip()) < 15 and re.match(r'^[\s\d\W]*$', lines[-1].strip()): lines = lines[:-1]
        extracted_text = '\n'.join(lines)
        print(f"Extracted {len(extracted_text)} characters.")
        return extracted_text
    except Exception as e:
        print(f"Text extraction failed: {str(e)}")
        traceback.print_exc()
        raise # Re-raise the exception to be caught by the Flask handler

def detect_math_content(text):
    math_keywords = [r'\b(equation|formula|theorem|lemma|proof|calculus|algebra|derivative|function|integral|vector|matrix|variable|constant|graph|plot)\b']
    math_symbols = r'[=><≤≥≠\+\-\*\/\^∫∑∏√∞≠≤≥±→∂∇πθλμσωαβγδε]'
    function_notation = r'\b[a-zA-Z]\s?\([a-zA-Z0-9,\s]+\)'
    latex_delimiters = r'\$.*?\$|\\\(.*?\\\)|\\[a-zA-Z]+(\{.*?\})*'
    patterns = math_keywords + [math_symbols, function_notation, latex_delimiters]
    if re.search('|'.join(math_keywords), text, re.IGNORECASE): return True
    text_sample = text[:50000]
    for pattern in [math_symbols, function_notation, latex_delimiters]:
        if re.search(pattern, text_sample): return True
    return False

def clean_text(text):
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])(\w)', r'\1 \2', text)
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'\b[^\s\w]{1,2}\b', '', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()

def split_text_into_chunks(text, chunk_size=800, overlap=50):
    try: sentences = nltk.sent_tokenize(text)
    except Exception:
        print("Warning: NLTK splitting failed, using regex fallback.")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences: sentences = [text]
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        sent_length = len(sentence.split())
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_words, overlap_sentences = 0, []
            for s in reversed(current_chunk):
                overlap_words += len(s.split())
                overlap_sentences.insert(0, s)
                if overlap_words >= overlap: break
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sent_length
    if current_chunk: chunks.append(' '.join(current_chunk))
    # Merge small chunks
    refined_chunks, i = [], 0
    while i < len(chunks):
        if len(chunks[i].split()) < chunk_size * 0.3 and i + 1 < len(chunks):
            refined_chunks.append(chunks[i] + " " + chunks[i+1]); i += 2
        else: refined_chunks.append(chunks[i]); i += 1
    chunks = refined_chunks
    # Limit chunks
    MAX_CHUNKS = 20
    if len(chunks) > MAX_CHUNKS:
        print(f"Warning: Reducing chunks from {len(chunks)} to {MAX_CHUNKS}")
        step = max(1, len(chunks) // MAX_CHUNKS)
        chunks = [chunks[i] for i in range(0, len(chunks), step)][:MAX_CHUNKS]
    print(f"Split into {len(chunks)} chunks (~{chunk_size} words, overlap {overlap}).")
    return chunks

def determine_reading_level(grade):
    if not isinstance(grade, int) or not (1 <= grade <= 12): grade = 6
    age = grade + 5
    if 1 <= grade <= 3: level, desc = "lower", f"early elementary (grades {grade}, ~age {age}-{age+1})"
    elif 4 <= grade <= 6: level, desc = "middle", f"late elem./middle school (grades {grade}, ~age {age}-{age+1})"
    elif 7 <= grade <= 9: level, desc = "higher", f"junior high/early high (grades {grade}, ~age {age}-{age+1})"
    else: level, desc = "higher", f"high school (grades {grade}, ~age {age}-{age+1})"
    return level, desc

# --- Prompts Dictionary (Copy from your final working version) ---
prompts = {
    "lower": { # Grades 1-3
        "standard": (
            "You are summarizing text for a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Explain the main ideas using VERY simple words and short sentences.\n"
            "2. Each sentence should start on a new line with a hyphen '- '.\n"
            "3. Focus on the most basic concepts. Avoid complex details.\n"
            "4. Group similar ideas together if possible.\n"
            "5. Include ONE simple, fun activity suggestion related to the text at the end, under '## Fun Activity'.\n"
            "6. Start the summary with the heading '# Simple Summary'.\n"
            "7. Do not repeat information.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math topic to a young child (grades 1-3, ages 6-8).\n"
            "Instructions:\n"
            "1. Use very simple words and analogies (like comparing numbers to toys).\n"
            "2. Break down steps simply: 'First, you do this. Next, you do that.'\n"
            "3. Use examples with small numbers or simple shapes.\n"
            "4. Include ONE simple practice question or drawing activity at the end under '## Practice Time'.\n"
            "5. Start the explanation with the heading '# Math Fun'.\n"
            "6. Format explanations clearly, maybe using bullet points '- ' for steps or examples.\n"
            "Text to explain:\n{text}"
        )
    },
    "middle": { # Grades 4-6
        "standard": (
            "You are summarizing text for a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Identify the main topics and summarize the key information for each.\n"
            "2. Use clear, complete sentences. Avoid jargon where possible, or briefly explain it.\n"
            "3. Organize the summary with headings for main topics (e.g., '## Topic Name').\n"
            "4. Use bullet points '- ' for key details under each heading.\n"
            "5. Include ONE practical activity or thought question related to the text at the end under '## Try This'.\n"
            "6. Start the summary with the heading '# Summary'.\n"
            "7. Ensure the summary flows logically and avoids redundancy.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining a math concept to a student in grades 4-6 (ages 9-11).\n"
            "Instructions:\n"
            "1. Explain the core math concept clearly.\n"
            "2. Provide a step-by-step example of how to solve a typical problem.\n"
            "3. Briefly mention a real-world application or why it's useful.\n"
            "4. Include ONE practice problem (with answer separate or omitted) at the end under '## Practice Problem'.\n"
            "5. Start the explanation with the heading '# Math Explained'.\n"
            "6. Use headings (e.g., '## Concept', '## Example', '## Why it Matters') and bullet points '- ' for clarity.\n"
            "Text to explain:\n{text}"
        )
    },
    "higher": { # Grades 7-12
        "standard": (
            "You are creating a comprehensive summary for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Identify key themes, arguments, and supporting evidence in the text.\n"
            "2. Structure the summary logically, perhaps by theme or section, using clear headings ('## Theme/Section').\n"
            "3. Use appropriate academic vocabulary but ensure clarity.\n"
            "4. Synthesize information; do not just list points. Avoid repetition.\n"
            "5. Mention real-world implications, applications, or connections where relevant.\n"
            "6. Include ONE thought-provoking question, research idea, or analysis task related to the content at the end under '## Further Thinking'.\n"
            "7. Start the summary with the heading '# Comprehensive Summary'.\n"
            "Text to summarize:\n{text}"
        ),
        "math": (
            "You are explaining an advanced math topic for a high school student (grades 7-12, ages 12-18).\n"
            "Instructions:\n"
            "1. Provide concise definitions of key terms or concepts.\n"
            "2. Briefly outline the logic or sketch a proof if applicable.\n"
            "3. Include a non-trivial worked example demonstrating the concept or technique.\n"
            "4. Mention practical applications or connections to other fields (science, engineering, finance, etc.).\n"
            "5. Include ONE challenging problem or extension idea at the end under '## Challenge'.\n"
            "6. Start the explanation with the heading '# Advanced Math Concepts'.\n"
            "7. Use appropriate mathematical notation (like LaTeX placeholders: \\int, \\sum, etc. if present in source text) and structure with headings/subheadings.\n"
            "Text to explain:\n{text}"
        )
    }
}
# --- End Prompts ---

def model_generate(prompt_text, max_new_tokens=1024, temperature=0.6):
    if not model or not tokenizer: return "Error: LLM not available."
    model_context_limit = getattr(tokenizer, 'model_max_length', 4096)
    max_prompt_tokens = model_context_limit - max_new_tokens - 100
    print(f"Generating response (max_new_tokens={max_new_tokens}, temp={temperature})...")
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_tokens).to(device)
        if inputs['input_ids'].shape[1] >= max_prompt_tokens:
            print(f"Warning: Prompt potentially truncated to fit model context window ({max_prompt_tokens} tokens).")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                                 do_sample=True if temperature > 0 else False)
        prompt_token_length = inputs['input_ids'].shape[1]
        generated_text = tokenizer.decode(outputs[0][prompt_token_length:], skip_special_tokens=True)
        generated_text = re.sub(r'<\|eot_id\|>', '', generated_text).strip()
        print("...Generation complete.")
        return generated_text
    except Exception as e:
        print(f"Error during model generation: {e}")
        traceback.print_exc()
        # Return an error string that can be identified by the caller
        return f"Error: Model generation failed - {str(e)}"

# --- generate_summary Function (Use your final version) ---
def generate_summary(text_chunks, grade_level_category, grade_level_desc, duration_minutes, has_math=False):
    if duration_minutes == 10: min_words, max_words = 1300, 1500
    elif duration_minutes == 20: min_words, max_words = 2600, 3000
    elif duration_minutes == 30: min_words, max_words = 3900, 4000
    else: min_words, max_words = 1300, 1500 # Default
    print(f"Targeting summary: {grade_level_desc}, {duration_minutes} mins ({min_words}-{max_words} words).")
    full_text = ' '.join(text_chunks)
    full_text_tokens_estimate = len(tokenizer.encode(full_text))
    MAX_POSSIBLE_TOKENS = 4096
    estimated_target_max_tokens = int(max_words * 1.5)
    model_context_limit = getattr(tokenizer, 'model_max_length', 4096)
    safe_generation_limit = min(MAX_POSSIBLE_TOKENS, model_context_limit // 2)
    max_new_tokens_summary = min(estimated_target_max_tokens, safe_generation_limit)
    max_new_tokens_summary = max(max_new_tokens_summary, 256)
    print(f"Calculated max_new_tokens for summary: {max_new_tokens_summary} (capped at {safe_generation_limit})")
    prompt_instruction_buffer = 600
    required_tokens_for_single_pass = full_text_tokens_estimate + max_new_tokens_summary + prompt_instruction_buffer
    can_summarize_all_at_once = required_tokens_for_single_pass < (model_context_limit * 0.9)
    if max_new_tokens_summary > 2048: can_summarize_all_at_once = False
    if full_text_tokens_estimate > (model_context_limit * 0.7): can_summarize_all_at_once = False

    initial_summary = ""
    if can_summarize_all_at_once and len(text_chunks) > 0 :
        print("Attempting summary generation in a single pass.")
        prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
        try: prompt = prompt_template.format(text=full_text)
        except KeyError: prompt = f"Summarize:\n{full_text}"
        prompt += (f"\n\nGenerate a detailed summary aiming for {min_words} to {max_words} words...") # Add detailed instructions
        initial_summary = model_generate(prompt, max_new_tokens=max_new_tokens_summary, temperature=0.7)
    elif len(text_chunks) > 0:
        print(f"Text too long or requires large output. Summarizing {len(text_chunks)} chunks iteratively.")
        chunk_summaries = []
        running_summary_context = ""
        max_new_tokens_chunk = min( (max_new_tokens_summary // len(text_chunks)) + 150, safe_generation_limit // 2)
        max_new_tokens_chunk = max(max_new_tokens_chunk, 150)
        for i, chunk in enumerate(text_chunks):
            print(f"Summarizing chunk {i + 1}/{len(text_chunks)}...")
            prompt_template = prompts[grade_level_category]["math" if has_math else "standard"]
            context = f"\n\nContext:\n{running_summary_context[-600:]}" if running_summary_context else ""
            try: chunk_prompt = prompt_template.format(text=chunk) + context
            except KeyError: chunk_prompt = f"Summarize part {i+1}:\n{chunk}\n{context}"
            chunk_prompt += (f"\n\nSummarize this part ({max_new_tokens_chunk} tokens max)...") # Add detailed instructions
            chunk_summary = model_generate(chunk_prompt, max_new_tokens=max_new_tokens_chunk, temperature=0.5)
            if chunk_summary.startswith("Error:"): continue # Skip failed chunks
            chunk_summary = re.sub(r'^.*?#+\s*.*?\s*#*', '', chunk_summary)
            chunk_summary = '. '.join(s.strip() for s in chunk_summary.split('.') if len(s.strip()) > 15)
            if chunk_summary:
                if not chunk_summary.endswith('.'): chunk_summary += '.'
                chunk_summaries.append(chunk_summary)
                running_summary_context += chunk_summary + "\n"
        if not chunk_summaries: return "Error: No valid chunk summaries generated."
        print("Consolidating chunk summaries...")
        max_tokens_consolidation = min(int(max_words * 1.6), safe_generation_limit)
        max_tokens_consolidation = max(max_tokens_consolidation, 512)
        consolidation_prompt = (f"Consolidate these pieces into one detailed summary for {grade_level_desc} ({min_words}-{max_words} words)...\n\n" # Add detailed instructions
                                f"Pieces:\n\n" + "\n\n---\n\n".join(chunk_summaries) + f"\n\nFinal summary:")
        initial_summary = model_generate(consolidation_prompt, max_new_tokens=max_tokens_consolidation, temperature=0.7)
    else: return "Error: Text processing resulted in zero chunks."

    # Check for generation errors before proceeding
    if initial_summary.startswith("Error:"):
        return initial_summary # Propagate the error

    # Iterative lengthening
    current_summary = initial_summary
    continuation_prompt_template = ("This summary is too short. Continue elaborating on existing points...\n"
                                    "Summary (End Portion):\n...{last_part}\n\nDetailed Continuation:")
    attempts, max_lengthening_attempts = 0, 5
    while len(current_summary.split()) < min_words and attempts < max_lengthening_attempts:
        print(f"Summary at {len(current_summary.split())} words; extending (Attempt {attempts + 1}).")
        last_part = ' '.join(current_summary.split()[-300:])
        tokens_to_add = min(500, max_new_tokens_summary)
        prompt = continuation_prompt_template.format(last_part=last_part)
        new_part = model_generate(prompt, max_new_tokens=tokens_to_add, temperature=0.7)
        if new_part.startswith("Error:") or len(new_part.split()) < 5:
            print("Stopping lengthening.")
            break
        current_summary += " " + re.sub(r'^.*?Detailed Continuation:', '', new_part, flags=re.IGNORECASE).strip()
        attempts += 1
    if attempts == max_lengthening_attempts: print("Warning: Reached max lengthening attempts.")

    # Trim
    words = current_summary.split()
    if len(words) > max_words:
        print(f"Trimming from {len(words)} to {max_words} words.")
        current_summary = ' '.join(words[:max_words]) + "..."
    summary = current_summary

    # Post-process
    print("Post-processing summary...")
    processed_summary = enhanced_post_process(summary, grade_level_category, min_words, max_words)

    # Ensure activity
    if not re.search(r'^##\s+(Activity|Practice|Thinking|Challenge|Try This|Fun Activity)', processed_summary, re.IGNORECASE | re.MULTILINE):
        print("Adding missing activity section...")
        activity = generate_activity(processed_summary, grade_level_category, grade_level_desc)
        activity_heading_map = {"lower": "## Fun Activity", "middle": "## Try This", "higher": "## Further Thinking"}
        activity_heading = activity_heading_map.get(grade_level_category, "## Activity Suggestion")
        processed_summary += f"\n\n{activity_heading}\n{activity}"

    final_word_count = len(processed_summary.split())
    print(f"Final summary generated ({final_word_count} words).")
    return processed_summary
# --- End generate_summary ---

# --- enhanced_post_process Function (Use your final version) ---
def enhanced_post_process(summary, grade_level_category, min_words, max_words):
    if summary.startswith("Error:"): return summary # Pass errors through
    print("Running enhanced post-processing...")
    main_heading_text = prompts[grade_level_category]["standard"].splitlines()[0].split("#")[-1].strip()
    summary = re.sub(r'^#+\s*.*?\s*#*', f'# {main_heading_text}', summary, count=1, flags=re.IGNORECASE | re.MULTILINE).strip()
    if not summary.startswith("# "): summary = f'# {main_heading_text}\n\n' + summary

    lines = summary.split('\n')
    processed_lines_data = []
    seen_content = set()
    elaboration_candidates = []

    for line in lines: # First pass
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            processed_lines_data.append({"text": line, "is_bullet": False, "is_heading": True})
            continue
        is_bullet = line.startswith('- ')
        content = (line[2:] if is_bullet else line).strip()
        if not content: continue
        content_key = content[:50].lower()
        if content_key in seen_content: continue
        seen_content.add(content_key)
        is_fragment = not re.search(r'[.!?]$', content) and len(content.split()) > 3 and content[0].isupper()
        if is_fragment: content = complete_sentence(content)
        if content and content[0].islower(): content = content[0].upper() + content[1:]
        if content and content[-1].isalnum(): content += '.'
        processed_lines_data.append({"text": content, "is_bullet": is_bullet, "is_heading": False})
        if len(content.split()) < 15 and not is_bullet: elaboration_candidates.append(content)

    # Optional elaboration
    if elaboration_candidates and len(processed_lines_data) < 30:
        print(f"Attempting elaboration on {len(elaboration_candidates)} short points...")
        elaboration_prompt = ("Elaborate on these points...\n" + "\n".join([f"- {p}" for p in elaboration_candidates]) + "\n\nElaborations:")
        elaborations_text = model_generate(elaboration_prompt, max_new_tokens=min(len(elaboration_candidates)*50, 400), temperature=0.6)
        if not elaborations_text.startswith("Error:"):
            elaboration_lines = [e.strip() for e in elaborations_text.split('\n') if len(e.strip()) > 10]
            for elab_line in elaboration_lines:
                if elab_line[-1].isalnum(): elab_line += '.'
                processed_lines_data.append({"text": elab_line, "is_bullet": True, "is_heading": False}) # Add as bullets

    # Deduplication and final assembly
    final_processed_lines_text = []
    points_for_dedup = []
    line_indices_for_dedup = {}
    for i, line_data in enumerate(processed_lines_data):
        if not line_data["is_heading"]:
            sentences = re.split(r'(?<=[.!?])\s+', line_data["text"])
            for sent in sentences:
                sent = sent.strip()
                if len(sent.split()) > 5:
                    points_for_dedup.append(sent)
                    if sent not in line_indices_for_dedup: line_indices_for_dedup[sent] = []
                    line_indices_for_dedup[sent].append(i)

    unique_points_content = points_for_dedup
    if points_for_dedup and embedder: # Check if embedder is available
        print(f"Running semantic duplicate removal on {len(points_for_dedup)} points...")
        unique_points_content = remove_duplicates_semantic(points_for_dedup)
        print(f"Reduced to {len(unique_points_content)} unique points.")
    elif not embedder:
         print("Skipping semantic dedup (embedder not loaded).")
    else: print("Skipping semantic dedup (no points).")

    kept_line_indices = set()
    unique_content_set = set(unique_points_content)
    # Add headings first
    for i, line_data in enumerate(processed_lines_data):
        if line_data["is_heading"]:
            final_processed_lines_text.append(line_data["text"])
            kept_line_indices.add(i)
    # Add unique content lines
    for sent, indices in line_indices_for_dedup.items():
        if sent in unique_content_set:
            for index in indices:
                if index not in kept_line_indices:
                    line_data = processed_lines_data[index]
                    formatted_line = f"- {line_data['text']}" if line_data['is_bullet'] else line_data['text']
                    final_processed_lines_text.append(formatted_line)
                    kept_line_indices.add(index)

    # Assemble final text with proper spacing
    full_text = ""
    for i, line in enumerate(final_processed_lines_text):
        full_text += line
        is_heading = line.startswith('#')
        is_last = (i == len(final_processed_lines_text) - 1)
        if not is_last:
            is_next_heading = final_processed_lines_text[i+1].startswith('#')
            if is_heading: full_text += "\n\n" # Double after heading
            elif not is_next_heading: full_text += "\n" # Single between non-headings
            else: full_text += "\n" # Single before next heading
    print("Post-processing finished.")
    return full_text.strip()
# --- End enhanced_post_process ---

# --- remove_duplicates_semantic Function (Use your final version) ---
def remove_duplicates_semantic(points, similarity_threshold=0.90):
    if not points or not embedder: return points
    try:
        valid_points = [p for p in points if len(p.split()) > 3]
        if not valid_points: return []
        embeddings = embedder.encode(valid_points, convert_to_tensor=True, show_progress_bar=False)
        unique_indices, kept_embeddings = [], []
        for i in range(len(valid_points)):
            is_duplicate = False
            if kept_embeddings:
                all_kept_embeddings = torch.cat(kept_embeddings, dim=0)
                similarities = util.dot_score(embeddings[i], all_kept_embeddings)[0]
                if torch.max(similarities) > similarity_threshold: is_duplicate = True
            if not is_duplicate:
                unique_indices.append(i)
                kept_embeddings.append(embeddings[i].unsqueeze(0))
        return [valid_points[i] for i in unique_indices]
    except Exception as e:
        print(f"Error during semantic duplicate removal: {e}.")
        traceback.print_exc()
        return points # Fallback to original points on error
# --- End remove_duplicates_semantic ---

# --- generate_activity Function (Use your final version) ---
def generate_activity(summary_text, grade_level_category, grade_level_desc):
    if not model or not tokenizer: return "- Review key points."
    print("Generating relevant activity...")
    activity_prompt_template = ("Suggest ONE simple activity for {grade_desc} based on this summary context:\n"
                                "...{summary}\n\nActivity Suggestion:")
    summary_snippet = summary_text[-1000:]
    prompt = activity_prompt_template.format(grade_desc=grade_level_desc, summary=summary_snippet)
    activity = model_generate(prompt, max_new_tokens=100, temperature=0.7)
    if activity.startswith("Error:"):
         print(f"Activity generation failed: {activity}")
         activity = "" # Reset activity string if generation failed
    else:
        activity = activity.strip().replace("Activity Suggestion:", "").strip()
        activity = re.sub(r'^[\-\*\s]+', '', activity)

    if activity:
        activity = activity[0].upper() + activity[1:]
        activity = f"- {activity}"
        if activity[-1].isalnum(): activity += '.'
        return activity
    else:
        print("Warning: Failed to generate activity text.")
        fallbacks = {"lower": "- Draw a picture!", "middle": "- Explain it to someone.", "higher": "- Find one related topic."}
        return fallbacks.get(grade_level_category, "- Review the summary.")
# --- End generate_activity ---

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# Serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    """API endpoint to handle PDF summarization."""
    if 'pdfFile' not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400

    file = request.files['pdfFile']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    # --- Get form data ---
    try:
        grade = int(request.form.get('grade', 6))
        duration = int(request.form.get('duration', 20))
        ocr_enabled = request.form.get('ocr') == 'true' # Check checkbox value
        chunk_size = int(request.form.get('chunkSize', 500))
        overlap = int(request.form.get('overlap', 50))
    except ValueError:
        return jsonify({"error": "Invalid form data (grade, duration, etc.)."}), 400

    # --- Save temporary file ---
    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            file.save(tmp_file) # Save uploaded file object to temp file
            pdf_path = tmp_file.name
        print(f"PDF saved temporarily to: {pdf_path}")

        # --- Run Summarization Logic ---
        grade_level_category, grade_level_desc = determine_reading_level(grade)
        print(f"Processing for: {grade_level_desc}")

        # Pass ocr_enabled flag here
        raw_text = extract_text_from_pdf(pdf_path, detect_math=True, ocr_enabled=ocr_enabled)

        if not raw_text or len(raw_text.strip()) < 50:
            return jsonify({"error": "No significant text extracted from PDF."}), 400

        has_math = detect_math_content(raw_text)
        print(f"Math content detected: {has_math}")

        print("Cleaning text...")
        cleaned_text = clean_text(raw_text)

        print("Splitting text...")
        chunks = split_text_into_chunks(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return jsonify({"error": "Failed to split text into chunks."}), 500

        print("Generating summary...")
        summary = generate_summary(chunks, grade_level_category, grade_level_desc, duration, has_math)

        # Check if summary generation itself returned an error
        if summary.startswith("Error:"):
             print(f"Summarization process failed: {summary}")
             return jsonify({"error": summary}), 500

        word_count = len(summary.split())
        print(f"Summary generated. Word count: {word_count}")

        return jsonify({"summary": summary, "word_count": word_count})

    except FileNotFoundError as e:
         print(f"Error: {e}")
         return jsonify({"error": str(e)}), 404
    except pytesseract.TesseractNotFoundError:
         err_msg = "Tesseract OCR Engine not found on server."
         print(f"Error: {err_msg}")
         return jsonify({"error": err_msg}), 500
    except Exception as e:
        print(f"--- An Unexpected Error Occurred ---")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred on the server: {str(e)}"}), 500
    finally:
        # --- Clean up temporary file ---
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
                print(f"Temporary file removed: {pdf_path}")
            except Exception as e:
                print(f"Error removing temporary file {pdf_path}: {e}")


# --- Run Flask App ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    # Use host='0.0.0.0' to make it accessible on your network
    # Use debug=True only for development, NEVER for production
    app.run(host='0.0.0.0', port=5000, debug=False)