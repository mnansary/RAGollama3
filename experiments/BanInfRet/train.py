import torch
import numpy as np
import random
import json
import os
import warnings
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import sentence_transformers
from sentence_transformers import SentenceTransformer, losses, models, LoggingHandler, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util import cos_sim
from pathlib import Path
import transformers
from data_utils import generate_training_examples

# Suppress FutureWarning for clean_up_tokenization_spaces
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# ----------------- Config -------------------
EMBEDDING_MODEL = "l3cube-pune/bengali-sentence-similarity-sbert"
MODEL_SAVE_PATH = "./d3_baninfret"
SAVE_DIR = "./dataset_splits"
TRAIN_DIRS = ["/home/vpa/deliverables/D2/BanQA/data", 
              "/home/vpa/deliverables/D3/BanQA/data", 
              "/home/vpa/deliverables/D4/BanQA/data"]
TRAIN_SAMPLES = 10000
VAL_RATIO = 0.1
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 256
NUM_EPOCHS = 1
WARMUP_STEPS = 100
LR = 1e-5
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Reproducibility -------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ----------------- Logging -------------------
logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Metrics file setup
METRICS_FILE = os.path.join(MODEL_SAVE_PATH, "training_metrics.json")
metrics_log = []

# Log library versions for debugging
logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"Transformers version: {transformers.__version__}")
logging.info(f"Sentence-Transformers version: {sentence_transformers.__version__}")

# Check NVML initialization
if torch.cuda.is_available():
    try:
        torch.cuda.get_device_properties(0)
        logging.info("CUDA device initialized successfully")
    except Exception as e:
        logging.warning(f"NVML initialization issue: {e}")

# ----------------- Similarity Score Function -------------------
def get_similarity_score(query: str, passage: str, model: SentenceTransformer) -> float:
    """
    Compute cosine similarity between a query and a passage using the embedding model.
    Returns a float between -1 and 1.
    """
    query_embedding = model.encode(query, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
    passage_embedding = model.encode(passage, convert_to_tensor=True, device=DEVICE, show_progress_bar=False)
    
    # Compute cosine similarity
    similarity = cos_sim(query_embedding.unsqueeze(0), passage_embedding.unsqueeze(0)).item()
    return similarity

# ----------------- Model -------------------
# Initialize base model for computing validation scores
base_model = SentenceTransformer(EMBEDDING_MODEL).to(DEVICE)

word_embedding_model = models.Transformer(EMBEDDING_MODEL, max_seq_length=MAX_SEQ_LENGTH)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(DEVICE)

# Freeze lower layers to preserve pre-trained weights
for param in word_embedding_model.auto_model.parameters():
    param.requires_grad = False

# Dynamically access transformer layers
try:
    layers = word_embedding_model.auto_model.encoder.layer
except AttributeError:
    layers = getattr(word_embedding_model.auto_model, 'layers', None)
    if layers is None:
        logging.error("Could not identify transformer layers. Check model architecture.")
        raise AttributeError("Unsupported model architecture")

# Unfreeze last 2 layers
for layer in layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Log model architecture
logging.info(f"Model architecture: {word_embedding_model.auto_model}")

# ----------------- Data -------------------
train_data, val_data = generate_training_examples(
    directories=TRAIN_DIRS,
    total_samples=TRAIN_SAMPLES,
    save_dir=SAVE_DIR,
    validation_split=VAL_RATIO,
    seed=SEED,
    base_model=base_model
)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Create evaluator with disabled progress bar
val_evaluator = None
if val_data:
    queries = [ex.texts[0] for ex in val_data]
    docs = [ex.texts[1] for ex in val_data]
    scores = [ex.label for ex in val_data]
    val_evaluator = EmbeddingSimilarityEvaluator(queries, docs, scores, main_similarity="cosine", show_progress_bar=False)

# ----------------- Metrics Callback -------------------
def metrics_callback(score, epoch, steps):
    """Callback to log training loss, evaluation scores, and average cosine similarity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if val_data:
        cosine_similarities = [
            get_similarity_score(ex.texts[0], ex.texts[1], model) for ex in val_data
        ]
        avg_cosine_similarity = float(np.mean(cosine_similarities))
    else:
        avg_cosine_similarity = None
    
    metrics = {
        "timestamp": timestamp,
        "epoch": epoch,
        "steps": steps,
        "train_loss": float(model._last_loss_value) if hasattr(model, '_last_loss_value') else None,
        "eval_score": float(score) if score is not None else None,
        "avg_cosine_similarity": avg_cosine_similarity
    }
    metrics_log.append(metrics)
    
    # Save metrics to file
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)

# ----------------- Training -------------------
steps_per_epoch = len(train_dataloader)
EVAL_STEPS = max(1, steps_per_epoch // 2)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=NUM_EPOCHS,
    evaluation_steps=EVAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    output_path=MODEL_SAVE_PATH,
    save_best_model=True,
    use_amp=True,
    optimizer_params={'lr': LR},
    callback=metrics_callback,
    show_progress_bar=True 
)