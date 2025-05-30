from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from pydub import AudioSegment
import whisper
from transformers import pipeline, AutoTokenizer
import torch
import nltk
import time
import logging
from collections import Counter
import re

# Initialize colorful logging
logging.basicConfig(
    level=logging.INFO,
    format='\033[1;34m%(asctime)s\033[0m - \033[1;32m%(levelname)s\033[0m - \033[1;36m%(message)s\033[0m'
)
logger = logging.getLogger(__name__)

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app, resources={
    r"/process": {"origins": "http://127.0.0.1:5000"},
    r"/": {"origins": "http://127.0.0.1:5000"}
})

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
MAX_INPUT_WORDS = 9000  # Safe limit for summarization model

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SERVER_NAME'] = '127.0.0.1:5000'

# ======================
# NLTK RESOURCE HANDLING
# ======================
def initialize_nltk():
    """Ensure all required NLTK resources are available"""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logger.info(f"✓ NLTK {resource} already installed")
        except LookupError:
            logger.info(f"↓ Downloading NLTK {resource}...")
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"✓ Successfully downloaded {resource}")
            except Exception as e:
                logger.error(f"✗ Failed to download {resource}: {str(e)}")
                raise

initialize_nltk()

# ==============
# MODEL LOADING
# ==============
def load_models():
    """Initialize and return Whisper and summarization models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Using device: {device.upper()}")
    
    try:
        # Whisper model with progress callback
        logger.info("🔊 Loading Whisper model...")
        whisper_model = whisper.load_model("base").to(device)
        
        # Summarization model with error handling
        logger.info("📝 Loading summarization model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-large-cnn",
            truncation=True,
            model_max_length=1024
        )
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        logger.info("✅ Models loaded successfully")
        return whisper_model, summarizer
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise

whisper_model, summarizer = load_models()

# =================
# UTILITY FUNCTIONS
# =================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def print_chunk_progress(current, total, start_time):
    """Display progress bar for chunk processing"""
    progress = (current / total) * 100
    elapsed = time.time() - start_time
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    
    print(
        f"\r🔧 Processing chunk {current}/{total} "
        f"[{'=' * int(progress/5)}{' ' * (20 - int(progress/5))}] "
        f"{progress:.1f}% | "
        f"Elapsed: {elapsed:.1f}s | "
        f"ETA: {eta:.1f}s",
        end='',
        flush=True
    )
    if current == total:
        print()

def clean_text(text):
    """Clean and normalize text for processing"""
    # Remove excessive whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    # Basic punctuation normalization
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    return text

# ======================
# CORE PROCESSING LOGIC
# ======================
def transcribe_audio(audio_path):
    """Transcribe audio with visual chunking progress and error handling"""
    try:
        start_time = time.time()
        filename = os.path.basename(audio_path)
        logger.info(f"🎧 Starting transcription for {filename}")
        
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000  # in seconds
        logger.info(f"⏱️  Audio duration: {duration:.2f} seconds")
        
        # Process in chunks if longer than 2 minutes
        if duration > 120:
            chunk_length = 60000  # 60 seconds per chunk
            chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
            full_transcript = ""
            
            logger.info(f"✂️  Splitting into {len(chunks)} chunks...")
            chunk_start_time = time.time()
            
            for i, chunk in enumerate(chunks, 1):
                chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], f"chunk_{i}.mp3")
                chunk.export(chunk_path, format="mp3")
                print_chunk_progress(i, len(chunks), chunk_start_time)
                
                try:
                    result = whisper_model.transcribe(chunk_path)
                    full_transcript += result["text"] + " "
                except Exception as e:
                    logger.error(f"❌ Chunk {i} failed: {str(e)}")
                    full_transcript += f"[Chunk {i} transcription failed] "
                finally:
                    os.remove(chunk_path)
            
            transcript = clean_text(full_transcript)
        else:
            logger.info("🔍 Processing as single chunk...")
            result = whisper_model.transcribe(audio_path)
            transcript = clean_text(result["text"])
        
        word_count = len(transcript.split())
        logger.info(f"✅ Transcription completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"📝 Transcript length: {word_count} words")
        return transcript
    
    except Exception as e:
        logger.error(f"❌ Transcription failed: {str(e)}")
        raise

def extract_key_points(transcript, num_points=5):
    """Extract key points from complete transcript with multiple fallbacks"""
    if not transcript.strip():
        return ["No transcript available for key points"]
    
    try:
        sentences = nltk.sent_tokenize(transcript)
        
        # Fallback if standard tokenization fails
        if len(sentences) <= 1:
            sentences = [s.strip() + '.' for s in transcript.split('. ') if s.strip()]
        
        # If still insufficient, use paragraph breaks
        if len(sentences) <= 1:
            sentences = [p.strip() for p in transcript.split('\n') if p.strip()]
        
        # Select diverse sentences
        if len(sentences) <= num_points:
            return sentences[:num_points]
        
        step = max(1, len(sentences) // num_points)
        key_points = [sentences[i] for i in range(0, len(sentences), step)][:num_points]
        return [kp for kp in key_points if kp.strip()]
    
    except Exception as e:
        logger.error(f"❌ Key points extraction failed: {str(e)}")
        # Ultimate fallback - return beginning of text
        return [transcript[:200] + "..." if len(transcript) > 200 else transcript]

def generate_summary(transcript):
    """Generate summary with multiple fallback strategies"""
    if not transcript.strip():
        return "No transcript available for summary"
    
    start_time = time.time()
    words = transcript.split()
    word_count = len(words)
    
    # Skip summary if too short
    if word_count < 50:
        return "Transcript too short for meaningful summary"
    
    # Attempt 1: Full BART summarization
    try:
        # Prepare safe input text
        input_text = ' '.join(words[:MAX_INPUT_WORDS])
        
        # Calculate lengths (100-200 words)
        max_len = 400
        min_len = 200
        
        logger.info(f"📊 Attempting full summary from {len(input_text.split())} words...")
        
        summary = summarizer(
            input_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        
        logger.info(f"✅ Summary generated in {time.time() - start_time:.2f} seconds")
        return summary
    
    except Exception as e:
        logger.warning(f"⚠️ Primary summary failed: {str(e)}")
        # Attempt 2: Extractive summarization
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            
            sentences = nltk.sent_tokenize(transcript)
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform(sentences)
            
            # Cluster sentences and pick from each cluster
            n_clusters = min(5, len(sentences))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            
            # Get closest sentence to each cluster center
            summary_sentences = []
            for i in range(n_clusters):
                idx = (kmeans.labels_ == i).nonzero()[0]
                if len(idx) > 0:
                    summary_sentences.append(sentences[idx[0]])
            
            fallback_summary = ' '.join(summary_sentences[:5])
            logger.info("✅ Generated fallback extractive summary")
            return fallback_summary
            
        except Exception as e:
            logger.warning(f"⚠️ Extractive summary failed: {str(e)}")
            # Attempt 3: Simple sentence selection
            sentences = [s.strip() for s in transcript.split('. ') if s.strip()]
            if len(sentences) > 5:
                return ' '.join(sentences[:5]) + " [...]"
            return ' '.join(sentences) if sentences else "Could not generate summary"

# ============
# API ROUTES
# ============
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    file_path = None
    try:
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logger.info(f"📤 Received file: {file.filename} ({os.path.getsize(file_path)/1024/1024:.2f} MB)")
        
        # Process audio
        transcript = transcribe_audio(file_path)
        key_points = extract_key_points(transcript)
        summary = generate_summary(transcript)
        
        return jsonify({
            "transcript": transcript,
            "key_points": key_points,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        return jsonify({
            "error": str(e),
            "transcript": "",
            "key_points": ["Processing error occurred"],
            "summary": ""
        }), 500
        
    finally:
        # Clean up uploaded file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info("🧹 Cleaned up temporary files")

if __name__ == '__main__':
    logger.info("\n🚀 Starting Podcast Summarizer API...")
    app.run(host='127.0.0.1', port=5000, debug=True)