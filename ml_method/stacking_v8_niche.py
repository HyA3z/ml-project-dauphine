"""
Stacking v8 — Niche Segmentation (Cascade Classification)
=================================================================
v7 → v8 : Statistical noise reduction via domain specialization

  [PHASE 1] CASCADE CLASSIFICATION (Fallback Logic)
      Priority 1: Hashtag mapping → 6 niches (Cooking, Study, Comedy,
                   Fitness, Tech, Finance)
      Priority 2: Keyword scan in caption (multilingual FR/EN)
      Priority 3: Heritage — majority niche of the creator_id

  [PHASE 2] COMPARATIVE ANALYSIS BY NICHE
      - target_log distribution by niche (mean, std, N)
      - Differential feature importance per segment
      - Detection of discriminating features per niche

  [PHASE 3] SPECIALIZED MODELS PER NICHE
      - A dedicated XGBoost + LightGBM per niche (chronological split)
      - Local vs global R² meta-comparison (0.3332)
      - Target: R² ≥ 0.40 on at least one segment

  [KEPT] Entire v7 pipeline: Target Encoding OOF, Feature Selection,
         Momentum, NLP, CV, Cross-features, Stacking L2 Optuna

Record to beat (global) : R² = 0.3332 | MAE = 0.2922
Target v8               : R² ≥ 0.40 local on ≥ 1 niche
"""

import ast
import os
import io
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from PIL import Image, ImageFilter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from textblob import TextBlob
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from collections import Counter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RECORD_R2   = 0.3332
RECORD_MAE  = 0.2922
N_OPTUNA_TRIALS   = 40
N_OOF_FOLDS       = 5
TOP_K_FEATURES    = 40
MIN_NICHE_SAMPLES = 80   # minimum threshold to train a specialized model

print("=" * 70)
print("  Stacking v8 — Niche Segmentation (Cascade Classification)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. LOADING & CREATOR_ID
# ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data_thumbnail.csv")

# cleaned_data_thumbnail.csv has no webVideoUrl column.
# cleaned_data_subtittles.csv is line-by-line identical (verified: 100% match
# on followers, video_rank, duration, hour, target_log, explosion_score).
# We map webVideoUrl by index to extract the true creator_id.
#   DO NOT use groupby("followers") : 69 follower values correspond
#     to several different creators → 39.5% of videos with incorrect momentum.
_df_sub = pd.read_csv("cleaned_data_subtittles.csv")
df["creator_id"] = _df_sub["webVideoUrl"].str.extract(r'tiktok\.com/@([^/]+)/video')
del _df_sub

# Sort by creator then chronological (essential for momentum + target encoding)
df = df.sort_values(["creator_id", "video_rank"]).reset_index(drop=True)

print(f"\n {df['creator_id'].nunique()} creators | {len(df):,} videos")

# ─────────────────────────────────────────────────────────────────────
# [PHASE 1] CASCADE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  [PHASE 1] CASCADE CLASSIFICATION — 6 Niches")
print("=" * 70)

# ── Mapping dictionary : hashtags → niche ───────────────────────
HASHTAG_MAP = {
    # Cooking / Food
    "cooking":    "Cooking", "recipe":      "Cooking", "food":        "Cooking",
    "foodtok":    "Cooking", "chef":        "Cooking", "baking":      "Cooking",
    "kitchen":    "Cooking", "meal":        "Cooking", "eat":         "Cooking",
    "dinner":     "Cooking", "lunch":       "Cooking", "breakfast":   "Cooking",
    "vegan":      "Cooking", "healthy":     "Cooking", "nutrition":   "Cooking",
    "recette":    "Cooking", "cuisine":     "Cooking", "manger":      "Cooking",
    "foodie":     "Cooking", "tasty":       "Cooking", "yummy":       "Cooking",
    "cookwithme": "Cooking", "cooktok":     "Cooking", "homecooking": "Cooking",

    # Study / Education
    "study":         "Study", "studytok":     "Study", "studywithme":  "Study",
    "studymotivation":"Study","school":       "Study", "college":      "Study",
    "university":    "Study", "exam":         "Study", "homework":     "Study",
    "learning":      "Study", "education":    "Study", "student":      "Study",
    "revision":      "Study", "notes":        "Study", "flashcards":   "Study",
    "studyaesthetic":"Study", "pomodoro":     "Study", "studycheck":   "Study",
    "etude":         "Study", "cours":        "Study",

    # Comedy / Entertainment
    "comedy":     "Comedy", "funny":       "Comedy", "humor":       "Comedy",
    "meme":       "Comedy", "lol":         "Comedy", "joke":        "Comedy",
    "prank":      "Comedy", "skit":        "Comedy", "relatable":   "Comedy",
    "trending":   "Comedy", "viral":       "Comedy", "entertainment":"Comedy",
    "comedytok":  "Comedy", "laughing":    "Comedy", "sketch":      "Comedy",
    "parody":     "Comedy", "funnytiktok": "Comedy",

    # Fitness / Sport
    "fitness":    "Fitness", "workout":     "Fitness", "gym":         "Fitness",
    "exercise":   "Fitness", "sport":       "Fitness", "training":    "Fitness",
    "fitnessmotivation":"Fitness","bodybuilding":"Fitness","cardio":   "Fitness",
    "running":    "Fitness", "yoga":        "Fitness", "pilates":     "Fitness",
    "fitspo":     "Fitness", "musculation": "Fitness", "crossfit":    "Fitness",
    "hiit":       "Fitness", "weightloss":  "Fitness", "fitnesscheck":"Fitness",
    "gymtok":     "Fitness",

    # Tech / Gaming / Digital
    "tech":         "Tech", "technology":   "Tech", "coding":       "Tech",
    "programming":  "Tech", "software":     "Tech", "ai":           "Tech",
    "developer":    "Tech", "gaming":       "Tech", "gamer":        "Tech",
    "techtok":      "Tech", "apple":        "Tech", "android":      "Tech",
    "python":       "Tech", "javascript":   "Tech", "startup":      "Tech",
    "innovation":   "Tech", "robotics":     "Tech", "cybersecurity":"Tech",
    "informatique": "Tech",

    # Finance / Investing / Business
    "finance":       "Finance", "investing":    "Finance", "stocks":      "Finance",
    "crypto":        "Finance", "bitcoin":      "Finance", "fintech":     "Finance",
    "money":         "Finance", "wealth":       "Finance", "business":    "Finance",
    "entrepreneur":  "Finance", "sidehustle":   "Finance", "passive":     "Finance",
    "trading":       "Finance", "bourse":       "Finance", "investment":  "Finance",
    "financetok":    "Finance", "budget":       "Finance", "frugal":      "Finance",
    "richlife":      "Finance",
}

# ── Caption keyword dictionary (FR + EN) ─────────────────────────
CAPTION_KW = {
    "Cooking": [
        r"\brecipe\b", r"\bcooking\b", r"\bchef\b", r"\bbaking\b", r"\bfood\b",
        r"\bmeal\b", r"\bdinner\b", r"\blunch\b", r"\bbreakfast\b", r"\bvegan\b",
        r"\brecette\b", r"\bcuisine\b", r"\bmanger\b", r"\bgr(e|é)dients?\b",
        r"\bingredients?\b", r"\bcook\b", r"\bdelicious\b", r"\btasty\b",
    ],
    "Study": [
        r"\bstudy\b", r"\bstudying\b", r"\bschool\b", r"\bcollege\b",
        r"\bexam\b", r"\bhomework\b", r"\bnotes\b", r"\blearning\b",
        r"\bstudent\b", r"\buniversity\b", r"\bétude\b", r"\bcours\b",
        r"\bdevoirs\b", r"\brévision\b", r"\blecture\b", r"\bflashcard\b",
    ],
    "Comedy": [
        r"\bfunny\b", r"\blol\b", r"\bhaha\b", r"\bcomédie\b", r"\bcomedy\b",
        r"\bjoke\b", r"\bprank\b", r"\bskit\b", r"\bmeme\b", r"\bhumor\b",
        r"\blaugh\b", r"\brelatable\b", r"\bwhen you\b", r"\bpov\b",
    ],
    "Fitness": [
        r"\bworkout\b", r"\bgym\b", r"\bfitness\b", r"\bexercise\b",
        r"\btraining\b", r"\bcardio\b", r"\bryoga\b", r"\brunning\b",
        r"\bpilates\b", r"\bbodybuilding\b", r"\bfat\s?loss\b",
        r"\bmusculation\b", r"\bséance\b", r"\bentraînement\b",
    ],
    "Tech": [
        r"\bcoding\b", r"\bprogramming\b", r"\bcode\b", r"\bgaming\b",
        r"\btechnology\b", r"\bai\b", r"\bapple\b", r"\biphone\b",
        r"\bsoftware\b", r"\bdeveloper\b", r"\btech\b", r"\bbot\b",
        r"\binformatique\b", r"\balgorithm\b", r"\bdéveloppeur\b",
    ],
    "Finance": [
        r"\bstocks?\b", r"\bcrypto\b", r"\bbitcoin\b", r"\binvest",
        r"\btrading\b", r"\bmoney\b", r"\bbourse\b", r"\bfinance\b",
        r"\bbusiness\b", r"\bentrepreneur\b", r"\bwealth\b",
        r"\bbudget\b", r"\bpassive\s?income\b", r"\bargent\b",
    ],
}

# ─────────────────────────────────────────────────────────────────────
# Classification functions
# ─────────────────────────────────────────────────────────────────────
def parse_hashtags(raw):
    try:
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if isinstance(tags, list):
            return [t["name"].lower() for t in tags if isinstance(t, dict) and "name" in t]
    except Exception:
        pass
    return []

def classify_from_hashtags(tags):
    """Priority 1: direct mapping hashtag → niche."""
    votes = Counter()
    for tag in tags:
        clean = re.sub(r"[^a-z0-9]", "", tag.lower())
        if clean in HASHTAG_MAP:
            votes[HASHTAG_MAP[clean]] += 1
    if votes:
        return votes.most_common(1)[0][0]
    return None

def classify_from_caption(text):
    """Priority 2: keyword scan in the caption."""
    if not isinstance(text, str) or text.strip() == "":
        return None
    text_lower = text.lower()
    scores = {}
    for niche, patterns in CAPTION_KW.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[niche] = score
    if scores:
        return max(scores, key=scores.get)
    return None

def classify_cascade(row, creator_majority):
    """Full cascade: hashtags → caption → creator heritage."""
    tags = parse_hashtags(row["hashtag"])

    # Priority 1: hashtags
    niche = classify_from_hashtags(tags)
    if niche:
        return niche, "hashtag"

    # Priority 2: caption
    niche = classify_from_caption(row["caption"])
    if niche:
        return niche, "caption"

    # Priority 3: creator heritage
    cid = row["creator_id"]
    if cid in creator_majority:
        return creator_majority[cid], "creator_heritage"

    return "Unknown", "unclassified"

# ── Pre-calculation for heritage (Priority 3) ──────────────────────────
# We first classify via P1+P2, then we calculate the majority niche per creator
print("\n    Pass 1 — P1+P2 Classification (without heritage)...")

p1p2_niches = []
for _, row in df.iterrows():
    tags  = parse_hashtags(row["hashtag"])
    niche = classify_from_hashtags(tags)
    if niche is None:
        niche = classify_from_caption(row["caption"])
    p1p2_niches.append(niche)

df["_niche_p1p2"] = p1p2_niches

# Calculate the majority niche per creator (on P1/P2 classified videos)
creator_majority = {}
for cid, grp in df[df["_niche_p1p2"].notna()].groupby("creator_id"):
    counts = grp["_niche_p1p2"].value_counts()
    if len(counts) > 0 and counts.iloc[0] >= 2:   # at least 2 videos in the niche
        creator_majority[cid] = counts.index[0]

print(f"    Heritage available for {len(creator_majority)} creators")

# ── Pass 2 : final classification with heritage ────────────────────
print("    Pass 2 : final classification with heritage (P3)...")

niche_results = df.apply(lambda row: classify_cascade(row, creator_majority), axis=1)
df["niche"]        = niche_results.apply(lambda x: x[0])
df["niche_source"] = niche_results.apply(lambda x: x[1])
df.drop(columns=["_niche_p1p2"], inplace=True)

# ── Classification report ─────────────────────────────────────────
print("\n" + "─" * 70)
print("    CLASSIFICATION REPORT")
print("─" * 70)
niche_counts  = df["niche"].value_counts()
source_counts = df["niche_source"].value_counts()

print(f"\n  {'Niche':<15} {'N videos':>10} {'%':>8}")
print(f"  {'─'*38}")
for niche, cnt in niche_counts.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {niche:<15} {cnt:>10,} {pct:>7.1f}%  {bar}")

print(f"\n  Classification source :")
for src, cnt in source_counts.items():
    print(f"    {src:<22} : {cnt:>5,}  ({cnt/len(df)*100:.1f}%)")

classified_pct = (df["niche"] != "Unknown").sum() / len(df) * 100
print(f"\n   Classification rate : {classified_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 2. FULL FEATURE ENGINEERING (identical to v7)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("    Feature Engineering (v7 pipeline)")
print("─" * 70)

df["_tags"]      = df["hashtag"].apply(parse_hashtags)
df["n_hashtags"] = df["_tags"].apply(len)
df["has_fyp"]    = df["_tags"].apply(lambda t: int(any("fyp"    in x for x in t)))
df["has_viral"]  = df["_tags"].apply(lambda t: int(any("viral"  in x for x in t)))
df["has_foryou"] = df["_tags"].apply(lambda t: int(any("foryou" in x for x in t)))
df.drop(columns=["_tags"], inplace=True)

df["caption_len"]           = df["caption"].fillna("").str.len()
df["has_emoji"]             = df["caption"].fillna("").apply(lambda s: int(any(ord(c) > 127 for c in s)))
df["has_question"]          = df["caption"].fillna("").str.contains(r"\?").astype(int)
df["has_exclamation"]       = df["caption"].fillna("").str.contains(r"!").astype(int)
df["viral_potential"]       = df["hist_p90_views"] / (df["hist_median_views"] + 1)
df["engagement_total_hist"] = df["hist_like_rate"] + df["hist_comment_rate"] + df["hist_share_rate"]
df["is_peak_hour"]          = df["hour"].between(17, 22).astype(int)
df["follower_tier"]         = np.log1p(df["followers"])
df["views_efficiency_trend"]= df["hist_p70_views"] / (df["hist_median_views"] + 1)

# Niche one-hot (numeric feature for models)
niche_dummies = pd.get_dummies(df["niche"], prefix="niche").astype(int)
df = pd.concat([df, niche_dummies], axis=1)
NICHE_COLS = [c for c in df.columns if c.startswith("niche_") and c != "niche_source"]

# ─────────────────────────────────────────────────────────────────────
# COMPUTER VISION (cache)
# ─────────────────────────────────────────────────────────────────────
print("\n  Loading CV features (PIL cache)...")
CV_CACHE_PATH = "cv_features_cache.csv"
SAVE_EVERY    = 100
t_cv = time.time()

def extract_image_features(url, timeout=5):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("L")
        img_small = img.resize((64, 64))
        arr = np.array(img_small, dtype=np.float32)
        brightness = float(arr.mean())
        contrast   = float(arr.std())
        lap        = img_small.filter(ImageFilter.FIND_EDGES)
        complexity = float(np.array(lap, dtype=np.float32).var())
        return brightness, contrast, complexity
    except Exception:
        return None

urls = df["coverUrl"].fillna("").tolist()
n    = len(urls)

if os.path.exists(CV_CACHE_PATH):
    cache_df = pd.read_csv(CV_CACHE_PATH, index_col="row_idx")
    done_idx = set(cache_df.index.tolist())
    print(f"    Cache found : {len(done_idx):,} images | {n - len(done_idx)} remaining")
else:
    cache_df = pd.DataFrame(columns=["img_brightness", "img_contrast", "img_complexity"])
    cache_df.index.name = "row_idx"
    done_idx = set()
    print(f"    Starting from scratch ({n:,} images)")

pending = [i for i in range(n) if i not in done_idx]
errors  = 0
batch   = {}

for count, i in enumerate(pending):
    url  = urls[i]
    feat = None if not url else extract_image_features(url)
    batch[i] = feat if feat is not None else (np.nan, np.nan, np.nan)
    if feat is None:
        errors += 1
    if (count + 1) % SAVE_EVERY == 0 or count == len(pending) - 1:
        batch_df = pd.DataFrame.from_dict(
            batch, orient="index",
            columns=["img_brightness", "img_contrast", "img_complexity"]
        )
        batch_df.index.name = "row_idx"
        cache_df = pd.concat([cache_df, batch_df])
        cache_df.to_csv(CV_CACHE_PATH)
        batch = {}
        print(f"   [{len(done_idx)+count+1:>5}/{n}] errors={errors} | {time.time()-t_cv:.0f}s")

cache_df = pd.read_csv(CV_CACHE_PATH, index_col="row_idx").sort_index()
for col in ["img_brightness", "img_contrast", "img_complexity"]:
    cache_df[col] = cache_df[col].fillna(cache_df[col].median())
df["img_brightness"] = cache_df["img_brightness"].values
df["img_contrast"]   = cache_df["img_contrast"].values
df["img_complexity"] = cache_df["img_complexity"].values
CV_COLS = ["img_brightness", "img_contrast", "img_complexity"]
print(f"\n   CV features ready in {time.time()-t_cv:.0f}s")

# ─────────────────────────────────────────────────────────────────────
# NLP (TextBlob)
# ─────────────────────────────────────────────────────────────────────
print("\n  NLP Features (TextBlob)...")
t_nlp = time.time()

def get_sentiment(text):
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0

captions   = df["caption"].fillna("")
sentiments = captions.apply(get_sentiment)
df["sentiment_polarity"]     = sentiments.apply(lambda x: x[0])
df["sentiment_subjectivity"] = sentiments.apply(lambda x: x[1])
df["emotional_intensity"]    = df["sentiment_polarity"].abs()
df["word_count"]             = captions.apply(lambda s: len(s.split()))
NLP_COLS = ["sentiment_polarity", "sentiment_subjectivity", "emotional_intensity", "word_count"]
print(f"   NLP in {time.time()-t_nlp:.1f}s")

# ─────────────────────────────────────────────────────────────────────
# MOMENTUM
# ─────────────────────────────────────────────────────────────────────
print("\n  Momentum...")

def rolling_slope(series, window=5):
    def slope(arr):
        if pd.Series(arr).isna().any() or len(arr) < 2:
            return np.nan
        return np.polyfit(np.arange(len(arr)), arr, 1)[0]
    return series.rolling(window, min_periods=2).apply(slope, raw=True)

def count_streak(series):
    def streak(arr):
        n = 0
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] > arr[i-1]: n += 1
            else: break
        return n
    return series.rolling(5, min_periods=1).apply(streak, raw=True)

grp     = df.groupby("creator_id")["explosion_score"]
shifted = grp.shift(1)

df["momentum_3"]      = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df["trend_slope"]     = shifted.groupby(df["creator_id"]).transform(lambda s: rolling_slope(s, window=5))
df["consistency"]     = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=2).std())
df["momentum_ratio"]  = df["momentum_3"] / (df["hist_median_views"] + 1)
df["trend_direction"] = np.sign(df["trend_slope"].fillna(0)).astype(int)
df["volatility_tier"] = df["consistency"] / (df["hist_median_views"] + 1)
df["momentum_7"]      = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(7, min_periods=2).mean())
df["accel"]           = df["momentum_3"] - df["momentum_7"]
df["peak_score"]      = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=1).max())
recent_min            = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=1).min())
df["recovery"]        = (df["momentum_3"] - recent_min) / (df["peak_score"] - recent_min + 1)
df["streak_up"]       = shifted.groupby(df["creator_id"]).transform(lambda s: count_streak(s))
df["momentum_norm"]   = df["momentum_3"] / (df["hist_p90_views"] + 1)

MOMENTUM_COLS = [
    "momentum_3", "trend_slope", "consistency", "momentum_ratio",
    "trend_direction", "volatility_tier", "momentum_7", "accel",
    "peak_score", "recovery", "streak_up", "momentum_norm",
]
df[MOMENTUM_COLS] = df[MOMENTUM_COLS].fillna(df[MOMENTUM_COLS].median())

# ─────────────────────────────────────────────────────────────────────
# CROSS-FEATURES (v4 + NLP + CV + Niche)
# ─────────────────────────────────────────────────────────────────────
print("  Cross-features...")
df["mom3_x_tier"]      = df["momentum_3"]    * df["follower_tier"]
df["accel_x_viral"]    = df["accel"]          * df["viral_potential"]
df["recovery_x_hist"]  = df["recovery"]       * df["hist_median_views"]
df["streak_x_engage"]  = df["streak_up"]      * df["engagement_total_hist"]
df["peak_x_p90"]       = df["peak_score"]     / (df["hist_p90_views"] + 1)
df["mom7_x_consist"]   = df["momentum_7"]     / (df["consistency"] + 0.1)
df["trend_x_duration"] = df["trend_slope"]    * df["duration"]
df["norm_x_tier"]      = df["momentum_norm"]  * df["follower_tier"]
df["intensity_x_momentum"] = df["emotional_intensity"] * df["momentum_3"]
df["polarity_x_viral"]     = df["sentiment_polarity"]  * df["viral_potential"]
df["subj_x_tier"]          = df["sentiment_subjectivity"] * df["follower_tier"]
df["contrast_x_momentum"]  = df["img_contrast"]    * df["momentum_3"]
df["bright_x_viral"]       = df["img_brightness"]  * df["viral_potential"]
df["complex_x_tier"]       = df["img_complexity"]  * df["follower_tier"]

CV_CROSS_COLS  = ["contrast_x_momentum", "bright_x_viral", "complex_x_tier"]
NLP_CROSS_COLS = ["intensity_x_momentum", "polarity_x_viral", "subj_x_tier"]
CROSS_COLS = [
    "mom3_x_tier", "accel_x_viral", "recovery_x_hist", "streak_x_engage",
    "peak_x_p90", "mom7_x_consist", "trend_x_duration", "norm_x_tier",
] + NLP_CROSS_COLS + CV_CROSS_COLS

df[CROSS_COLS] = df[CROSS_COLS].replace([np.inf, -np.inf], np.nan).fillna(df[CROSS_COLS].median())

# ─────────────────────────────────────────────────────────────────────
# FULL FEATURE SET
# ─────────────────────────────────────────────────────────────────────
STATIC_FEATURES = [
    "followers", "duration", "hour", "weekday", "musicOriginal",
    "hist_median_views", "hist_p70_views", "hist_p90_views",
    "hist_like_rate", "hist_comment_rate", "hist_share_rate",
    "n_hashtags", "has_fyp", "has_viral", "has_foryou",
    "caption_len", "has_emoji", "has_question", "has_exclamation",
    "viral_potential", "engagement_total_hist", "is_peak_hour",
    "follower_tier", "views_efficiency_trend",
]
ALL_FEATURES = STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS + NLP_COLS + CV_COLS + NICHE_COLS
TARGET = "target_log"
print(f"\n  Raw feature set : {len(ALL_FEATURES)} features ({len(NICHE_COLS)} niche one-hot)")

# ─────────────────────────────────────────────────────────────────────
# STRICT CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────
train_mask = df["video_rank"].between(11, 26)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

print(f"\n  Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")
print(f"   Global record : R²={RECORD_R2} | MAE={RECORD_MAE}\n")

# ─────────────────────────────────────────────────────────────────────
# TARGET ENCODING OOF (anti-leakage)
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  [LEVIER A] Target Encoding OOF (anti-leakage)")
print("─" * 70)

SMOOTH_K    = 5
global_mean = df.loc[train_mask, TARGET].mean()

# Causal expanding window encoding: for each video of rank R,
# we encode with the smoothed average of ranks < R from the same creator.
# Identical to corrected v9 — only possible leak-free encoding on this dataset
# (ranks 11-30 only, no external history available).
def expanding_te(group, smooth_k, gm):
    vals = group[TARGET].values
    te   = np.full(len(vals), gm)
    cs, cc = 0.0, 0
    for i in range(len(vals)):
        te[i] = (cs + smooth_k * gm) / (cc + smooth_k)
        cs += vals[i]; cc += 1
    return pd.Series(te, index=group.index)

df["creator_target_mean"] = (
    df.groupby("creator_id", group_keys=False)
      .apply(lambda g: expanding_te(g, SMOOTH_K, global_mean))
)

# Recalculating masks after sorting (sort_values above)
train_mask = df["video_rank"].between(11, 26)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

te_corr = np.corrcoef(df.loc[train_mask, "creator_target_mean"],
                      df.loc[train_mask, TARGET])[0, 1]
assert te_corr < 0.999, f"  Leakage detected (corr={te_corr:.4f})"
print(f"   creator_target_mean (expanding, k={SMOOTH_K}) | corr={te_corr:.4f}")

ALL_FEATURES_V8 = ALL_FEATURES + ["creator_target_mean"]

# ─────────────────────────────────────────────────────────────────────
# FEATURE SELECTION TOP-K (Fast CatBoost)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"  [LEVIER B] Feature Selection — Top {TOP_K_FEATURES}")
print("─" * 70)

X_train_full = df.loc[train_mask, ALL_FEATURES_V8].astype(float)
y_train      = df.loc[train_mask, TARGET]
X_val_full   = df.loc[val_mask,   ALL_FEATURES_V8].astype(float)
y_val        = df.loc[val_mask,   TARGET]
X_test_full  = df.loc[test_mask,  ALL_FEATURES_V8].astype(float)
y_test       = df.loc[test_mask,  TARGET]

cat_selector = CatBoostRegressor(
    iterations=3000, learning_rate=0.05, depth=6,
    early_stopping_rounds=100, random_state=42, verbose=0
)
cat_selector.fit(X_train_full, y_train, eval_set=(X_val_full, y_val))

fi_series = pd.Series(
    cat_selector.get_feature_importance(),
    index=ALL_FEATURES_V8
).sort_values(ascending=False)

SELECTED_FEATURES = fi_series.head(TOP_K_FEATURES).index.tolist()

print(f"\n  Top {TOP_K_FEATURES} selected features :")
for rank, (feat, imp) in enumerate(fi_series.head(TOP_K_FEATURES).items(), 1):
    tag = ""
    if feat == "creator_target_mean": tag = " ←  TARGET ENC"
    elif feat in NICHE_COLS:          tag = " ←   NICHE"
    elif feat in CV_COLS:             tag = " ←   CV"
    elif feat in MOMENTUM_COLS:       tag = " ←  momentum"
    elif feat in NLP_COLS:            tag = " ←  NLP"
    elif feat in CV_CROSS_COLS:       tag = " ←   CV×"
    elif feat in CROSS_COLS:          tag = " ←  crossed"
    print(f"    #{rank:>2}  {feat:<35} {imp:>6.2f}%{tag}")

X_train = df.loc[train_mask, SELECTED_FEATURES].astype(float)
X_val   = df.loc[val_mask,   SELECTED_FEATURES].astype(float)
X_test  = df.loc[test_mask,  SELECTED_FEATURES].astype(float)

# ─────────────────────────────────────────────────────────────────────
# [PHASE 2] COMPARATIVE ANALYSIS BY NICHE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  [PHASE 2] COMPARATIVE ANALYSIS BY NICHE")
print("=" * 70)

# Descriptive stats by niche
print(f"\n  {'Niche':<12} {'N':>6} {'N_train':>8} {'N_test':>7} {'μ_target':>9} {'σ_target':>9} {'μ_viral':>9}")
print(f"  {'─' * 68}")

niche_stats = {}
for niche in sorted(df["niche"].unique()):
    mask        = df["niche"] == niche
    n_total     = mask.sum()
    n_train     = (mask & train_mask).sum()
    n_test      = (mask & test_mask).sum()
    mu_target   = df.loc[mask, TARGET].mean()
    std_target  = df.loc[mask, TARGET].std()
    mu_viral    = df.loc[mask, "viral_potential"].mean()
    niche_stats[niche] = {
        "n_total": n_total, "n_train": n_train, "n_test": n_test,
        "mu_target": mu_target, "std_target": std_target, "mu_viral": mu_viral
    }
    print(f"  {niche:<12} {n_total:>6,} {n_train:>8,} {n_test:>7,} {mu_target:>9.3f} {std_target:>9.3f} {mu_viral:>9.2f}")

# Differential feature importance by niche (top 5)
print(f"\n    Differential Feature Importance by niche (top 5, fast CatBoost) :")
niche_fi = {}

for niche in sorted(df["niche"].unique()):
    mask_n  = df["niche"] == niche
    n_train_n = (mask_n & train_mask).sum()
    n_val_n   = (mask_n & val_mask).sum()

    if n_train_n < MIN_NICHE_SAMPLES or n_val_n < 10:
        continue

    Xn_train = df.loc[mask_n & train_mask, SELECTED_FEATURES].astype(float)
    yn_train = df.loc[mask_n & train_mask, TARGET]
    Xn_val   = df.loc[mask_n & val_mask,   SELECTED_FEATURES].astype(float)
    yn_val   = df.loc[mask_n & val_mask,   TARGET]

    cat_n = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=5,
        early_stopping_rounds=50, random_state=42, verbose=0
    )
    cat_n.fit(Xn_train, yn_train, eval_set=(Xn_val, yn_val))
    fi_n = pd.Series(cat_n.get_feature_importance(), index=SELECTED_FEATURES).sort_values(ascending=False)
    niche_fi[niche] = fi_n

    top5 = ", ".join([f"{f}({v:.1f}%)" for f, v in fi_n.head(5).items()])
    print(f"    {niche:<10} : {top5}")

# ─────────────────────────────────────────────────────────────────────
# [PHASE 3] SPECIALIZED MODELS PER NICHE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  [PHASE 3] SPECIALIZED MODELS PER NICHE")
print("=" * 70)

niche_results = {}
NICHES_TO_TRAIN = [
    n for n in sorted(df["niche"].unique())
    if n != "Unknown"
    and (df["niche"] == n).sum() >= MIN_NICHE_SAMPLES
    and ((df["niche"] == n) & train_mask).sum() >= MIN_NICHE_SAMPLES // 2
]

print(f"\n  Eligible niches (≥ {MIN_NICHE_SAMPLES} total videos) : {NICHES_TO_TRAIN}")

def build_meta_features(p_xgb, p_lgbm, p_cat=None):
    if p_cat is not None:
        return np.column_stack([
            p_xgb, p_lgbm, p_cat,
            (p_xgb + p_lgbm + p_cat) / 3,
            np.abs(p_xgb - p_cat),
            np.abs(p_lgbm - p_cat),
            np.abs(p_xgb - p_lgbm),
        ])
    return np.column_stack([
        p_xgb, p_lgbm,
        (p_xgb + p_lgbm) / 2,
        np.abs(p_xgb - p_lgbm),
    ])

for niche in NICHES_TO_TRAIN:
    print(f"\n  {'─'*60}")
    print(f"    Niche : {niche.upper()}")
    print(f"  {'─'*60}")

    mask_n    = df["niche"] == niche
    n_train_n = (mask_n & train_mask).sum()
    n_val_n   = (mask_n & val_mask).sum()
    n_test_n  = (mask_n & test_mask).sum()

    print(f"  Train={n_train_n} | Val={n_val_n} | Test={n_test_n}")

    if n_train_n < 30 or n_val_n < 5 or n_test_n < 5:
        print(f"    Insufficient split, niche ignored")
        continue

    Xn_train = df.loc[mask_n & train_mask, SELECTED_FEATURES].astype(float)
    yn_train = df.loc[mask_n & train_mask, TARGET]
    Xn_val   = df.loc[mask_n & val_mask,   SELECTED_FEATURES].astype(float)
    yn_val   = df.loc[mask_n & val_mask,   TARGET]
    Xn_test  = df.loc[mask_n & test_mask,  SELECTED_FEATURES].astype(float)
    yn_test  = df.loc[mask_n & test_mask,  TARGET]

    niche_model_results = {}
    t0 = time.time()

    # ── Specialized XGBoost ──────────────────────────────────────────
    print(f"    XGBoost niche ({N_OPTUNA_TRIALS} trials)...")

    def xgb_niche_objective(trial):
        params = dict(
            max_depth        = trial.suggest_int  ("max_depth",        3, 7),
            learning_rate    = trial.suggest_float("learning_rate",    0.005, 0.08, log=True),
            n_estimators     = trial.suggest_int  ("n_estimators",     500, 5000, step=500),
            subsample        = trial.suggest_float("subsample",        0.65, 0.95),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.50, 0.90),
            min_child_weight = trial.suggest_int  ("min_child_weight", 1, 15),
            reg_alpha        = trial.suggest_float("reg_alpha",        0.01, 5.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda",       0.5,  10.0, log=True),
            gamma            = trial.suggest_float("gamma",            0.0,  0.5),
        )
        m = XGBRegressor(**params, tree_method="hist", early_stopping_rounds=100,
                         eval_metric="mae", verbosity=0, random_state=42)
        m.fit(Xn_train, yn_train, eval_set=[(Xn_val, yn_val)], verbose=False)
        return -r2_score(yn_val, m.predict(Xn_val))

    xgb_study_n = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    xgb_study_n.optimize(xgb_niche_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_xgb_n = xgb_study_n.best_params

    xgb_n = XGBRegressor(**best_xgb_n, tree_method="hist", early_stopping_rounds=100,
                         eval_metric="mae", verbosity=0, random_state=42)
    xgb_n.fit(Xn_train, yn_train, eval_set=[(Xn_val, yn_val)], verbose=False)
    r2_xgb_n  = r2_score(yn_test, xgb_n.predict(Xn_test))
    mae_xgb_n = mean_absolute_error(yn_test, xgb_n.predict(Xn_test))
    niche_model_results["XGBoost"] = {"r2": r2_xgb_n, "mae": mae_xgb_n, "model": xgb_n}
    print(f"    R²={r2_xgb_n:.4f} | MAE={mae_xgb_n:.4f}  {'PASS > global' if r2_xgb_n > RECORD_R2 else 'FAIL'} {'* ≥ 0.40 !' if r2_xgb_n >= 0.40 else ''}")

    # ── Specialized LightGBM ─────────────────────────────────────────
    print(f"    LightGBM niche ({N_OPTUNA_TRIALS} trials)...")

    def lgbm_niche_objective(trial):
        params = dict(
            max_depth         = trial.suggest_int  ("max_depth",         3, 8),
            learning_rate     = trial.suggest_float("learning_rate",     0.005, 0.08, log=True),
            n_estimators      = trial.suggest_int  ("n_estimators",      500, 5000, step=500),
            num_leaves        = trial.suggest_int  ("num_leaves",        15, 100),
            subsample         = trial.suggest_float("subsample",         0.65, 0.95),
            colsample_bytree  = trial.suggest_float("colsample_bytree",  0.50, 0.90),
            min_child_samples = trial.suggest_int  ("min_child_samples", 5, 50),
            reg_alpha         = trial.suggest_float("reg_alpha",         0.01, 5.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda",        0.5,  10.0, log=True),
        )
        m = LGBMRegressor(**params, random_state=42, verbose=-1)
        cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
        m.fit(Xn_train, yn_train, eval_set=[(Xn_val, yn_val)], callbacks=cb)
        return -r2_score(yn_val, m.predict(Xn_val))

    lgbm_study_n = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    lgbm_study_n.optimize(lgbm_niche_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_lgbm_n = lgbm_study_n.best_params

    lgbm_n = LGBMRegressor(**best_lgbm_n, random_state=42, verbose=-1)
    cb_fit = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    lgbm_n.fit(Xn_train, yn_train, eval_set=[(Xn_val, yn_val)], callbacks=cb_fit)
    r2_lgbm_n  = r2_score(yn_test, lgbm_n.predict(Xn_test))
    mae_lgbm_n = mean_absolute_error(yn_test, lgbm_n.predict(Xn_test))
    niche_model_results["LightGBM"] = {"r2": r2_lgbm_n, "mae": mae_lgbm_n, "model": lgbm_n}
    print(f"    R²={r2_lgbm_n:.4f} | MAE={mae_lgbm_n:.4f}  {'PASS > global' if r2_lgbm_n > RECORD_R2 else 'FAIL'} {'* ≥ 0.40 !' if r2_lgbm_n >= 0.40 else ''}")

    # ── Light Ridge Stacking (XGB + LGBM) ───────────────────────────
    if n_val_n >= 15:
        p_xgb_v  = xgb_n.predict(Xn_val)
        p_lgbm_v = lgbm_n.predict(Xn_val)
        p_xgb_t  = xgb_n.predict(Xn_test)
        p_lgbm_t = lgbm_n.predict(Xn_test)

        meta_v = build_meta_features(p_xgb_v,  p_lgbm_v)
        meta_t = build_meta_features(p_xgb_t,  p_lgbm_t)

        ridge_n = Ridge(alpha=1.0)
        ridge_n.fit(meta_v, yn_val)
        r2_stack_n  = r2_score(yn_test, ridge_n.predict(meta_t))
        mae_stack_n = mean_absolute_error(yn_test, ridge_n.predict(meta_t))
        niche_model_results["Stacking"] = {"r2": r2_stack_n, "mae": mae_stack_n}
        print(f"    Niche Stacking : R²={r2_stack_n:.4f} | MAE={mae_stack_n:.4f}  {'PASS > global' if r2_stack_n > RECORD_R2 else 'FAIL'} {'* ≥ 0.40 !' if r2_stack_n >= 0.40 else ''}")

    elapsed_n = time.time() - t0
    best_r2_n = max(v["r2"] for v in niche_model_results.values())
    niche_results[niche] = {
        "models": niche_model_results,
        "best_r2": best_r2_n,
        "n_train": n_train_n, "n_test": n_test_n,
        "elapsed": elapsed_n
    }

# ─────────────────────────────────────────────────────────────────────
# GLOBAL MODEL v8 (Full L2 Stacking, features + niche one-hot)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("    GLOBAL MODEL v8 (L2 Stacking + Niche One-Hot)")
print("=" * 70)

global_results = {}

# ── Global XGBoost ────────────────────────────────────────────────────
print(f"\n    Global XGBoost ({N_OPTUNA_TRIALS} trials)...")

def xgb_global_objective(trial):
    params = dict(
        max_depth        = trial.suggest_int  ("max_depth",        3, 7),
        learning_rate    = trial.suggest_float("learning_rate",    0.002, 0.05, log=True),
        n_estimators     = trial.suggest_int  ("n_estimators",     3000, 12000, step=1000),
        subsample        = trial.suggest_float("subsample",        0.65, 0.95),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.50, 0.90),
        min_child_weight = trial.suggest_int  ("min_child_weight", 2, 15),
        reg_alpha        = trial.suggest_float("reg_alpha",        0.01, 5.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",       0.5,  10.0, log=True),
        gamma            = trial.suggest_float("gamma",            0.0,  0.5),
    )
    m = XGBRegressor(**params, tree_method="hist", early_stopping_rounds=200,
                     eval_metric="mae", verbosity=0, random_state=42)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return -r2_score(y_val, m.predict(X_val))

t0 = time.time()
xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), pruner=HyperbandPruner())
xgb_study.optimize(xgb_global_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_xgb = xgb_study.best_params

xgb_model = XGBRegressor(**best_xgb, tree_method="hist", early_stopping_rounds=200,
                          eval_metric="mae", verbosity=0, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb  = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"  R²={r2_xgb:.4f} | MAE={mae_xgb:.4f}  {'PASS' if r2_xgb > RECORD_R2 else 'FAIL'} | {time.time()-t0:.0f}s")
global_results["XGBoost"] = {"r2": r2_xgb, "mae": mae_xgb}

# ── Global LightGBM ───────────────────────────────────────────────────
print(f"\n    Global LightGBM ({N_OPTUNA_TRIALS} trials)...")

def lgbm_global_objective(trial):
    params = dict(
        max_depth         = trial.suggest_int  ("max_depth",         3, 8),
        learning_rate     = trial.suggest_float("learning_rate",     0.002, 0.05, log=True),
        n_estimators      = trial.suggest_int  ("n_estimators",      3000, 12000, step=1000),
        num_leaves        = trial.suggest_int  ("num_leaves",        20, 127),
        subsample         = trial.suggest_float("subsample",         0.65, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.50, 0.90),
        min_child_samples = trial.suggest_int  ("min_child_samples", 10, 80),
        reg_alpha         = trial.suggest_float("reg_alpha",         0.01, 5.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",        0.5,  10.0, log=True),
    )
    m = LGBMRegressor(**params, random_state=42, verbose=-1)
    cb = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cb)
    return -r2_score(y_val, m.predict(X_val))

t0 = time.time()
lgbm_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), pruner=HyperbandPruner())
lgbm_study.optimize(lgbm_global_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm = lgbm_study.best_params

lgbm_model = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
cb_fit = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cb_fit)
y_pred_lgbm = lgbm_model.predict(X_test)
r2_lgbm  = r2_score(y_test, y_pred_lgbm)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
print(f"  R²={r2_lgbm:.4f} | MAE={mae_lgbm:.4f}  {'PASS' if r2_lgbm > RECORD_R2 else 'FAIL'} | {time.time()-t0:.0f}s")
global_results["LightGBM"] = {"r2": r2_lgbm, "mae": mae_lgbm}

# ── Global CatBoost ───────────────────────────────────────────────────
print(f"\n    Global CatBoost ({N_OPTUNA_TRIALS} trials)...")

def cat_global_objective(trial):
    params = dict(
        depth               = trial.suggest_int  ("depth",               4, 10),
        learning_rate       = trial.suggest_float("learning_rate",       0.002, 0.05, log=True),
        iterations          = trial.suggest_int  ("iterations",          3000, 12000, step=1000),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg",         1.0, 20.0, log=True),
        subsample           = trial.suggest_float("subsample",           0.65, 0.95),
        colsample_bylevel   = trial.suggest_float("colsample_bylevel",   0.50, 0.90),
        min_data_in_leaf    = trial.suggest_int  ("min_data_in_leaf",    3, 30),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength     = trial.suggest_float("random_strength",     0.5, 5.0),
    )
    m = CatBoostRegressor(**params, early_stopping_rounds=200, random_state=42, verbose=0)
    m.fit(X_train, y_train, eval_set=(X_val, y_val))
    return -r2_score(y_val, m.predict(X_val))

t0 = time.time()
cat_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42), pruner=HyperbandPruner())
cat_study.optimize(cat_global_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_cat = cat_study.best_params

cat_model = CatBoostRegressor(**best_cat, early_stopping_rounds=200, random_state=42, verbose=0)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
y_pred_cat = cat_model.predict(X_test)
r2_cat  = r2_score(y_test, y_pred_cat)
mae_cat = mean_absolute_error(y_test, y_pred_cat)
print(f"  R²={r2_cat:.4f} | MAE={mae_cat:.4f}  {'PASS' if r2_cat > RECORD_R2 else 'FAIL'} | {time.time()-t0:.0f}s")
global_results["CatBoost"] = {"r2": r2_cat, "mae": mae_cat}

# ── Global L2 Stacking ────────────────────────────────────────────────
print("\n    Global L2 OOF Stacking...")
t0 = time.time()

kf = KFold(n_splits=5, shuffle=False)
xgb_s  = XGBRegressor(**best_xgb,  tree_method="hist", verbosity=0, random_state=42)
lgbm_s = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
cat_s  = CatBoostRegressor(**best_cat, random_state=42, verbose=0)

oof_xgb  = cross_val_predict(xgb_s,  X_train, y_train, cv=kf)
oof_lgbm = cross_val_predict(lgbm_s, X_train, y_train, cv=kf)
oof_cat  = cross_val_predict(cat_s,  X_train, y_train, cv=kf)

xgb_s.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
lgbm_s.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
cat_s.fit(X_train, y_train, eval_set=(X_val, y_val))

p_xgb_test  = xgb_s.predict(X_test)
p_lgbm_test = lgbm_s.predict(X_test)
p_cat_test  = cat_s.predict(X_test)

meta_train_l2 = build_meta_features(oof_xgb, oof_lgbm, oof_cat)
meta_test_l2  = build_meta_features(p_xgb_test, p_lgbm_test, p_cat_test)
meta_val_l2   = build_meta_features(xgb_s.predict(X_val), lgbm_s.predict(X_val), cat_s.predict(X_val))

def meta_l2_objective(trial):
    p = dict(
        max_depth     = trial.suggest_int  ("max_depth",     2, 5),
        learning_rate = trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        n_estimators  = trial.suggest_int  ("n_estimators",  500, 5000, step=500),
        subsample     = trial.suggest_float("subsample",     0.7, 1.0),
        reg_alpha     = trial.suggest_float("reg_alpha",     0.01, 5.0, log=True),
        reg_lambda    = trial.suggest_float("reg_lambda",    0.5, 10.0, log=True),
    )
    m = XGBRegressor(**p, tree_method="hist", early_stopping_rounds=100,
                     eval_metric="mae", verbosity=0, random_state=77)
    m.fit(meta_train_l2, y_train, eval_set=[(meta_val_l2, y_val)], verbose=False)
    return -r2_score(y_val, m.predict(meta_val_l2))

meta_xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=77))
meta_xgb_study.optimize(meta_l2_objective, n_trials=20, show_progress_bar=False)
best_meta_xgb = meta_xgb_study.best_params

xgb_l2 = XGBRegressor(**best_meta_xgb, tree_method="hist", early_stopping_rounds=100,
                       eval_metric="mae", verbosity=0, random_state=77)
xgb_l2.fit(meta_train_l2, y_train, eval_set=[(meta_val_l2, y_val)], verbose=False)

cat_l2 = CatBoostRegressor(depth=4, learning_rate=0.01, iterations=3000,
                            l2_leaf_reg=5, subsample=0.8,
                            early_stopping_rounds=100, random_state=77, verbose=0)
cat_l2.fit(meta_train_l2, y_train, eval_set=(meta_val_l2, y_val))

meta_l2_val  = np.column_stack([xgb_l2.predict(meta_val_l2),  cat_l2.predict(meta_val_l2)])
meta_l2_test = np.column_stack([xgb_l2.predict(meta_test_l2), cat_l2.predict(meta_test_l2)])

ridge_l2 = Ridge(alpha=1.0)
ridge_l2.fit(meta_l2_val, y_val)
y_pred_l2 = ridge_l2.predict(meta_l2_test)
r2_l2   = r2_score(y_test, y_pred_l2)
mae_l2  = mean_absolute_error(y_test, y_pred_l2)
print(f"  R²={r2_l2:.4f} | MAE={mae_l2:.4f}  {'PASS' if r2_l2 > RECORD_R2 else 'FAIL'}  (Δ {r2_l2 - RECORD_R2:+.4f}) | {time.time()-t0:.0f}s")
global_results["Stacking-L2"] = {"r2": r2_l2, "mae": mae_l2}

# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY — v8 (Niche Segmentation)")
print("=" * 70)

print(f"\n  ── Global Model v8 (niche one-hot) ──")
print(f"  {'Model':<22} {'R²':>8} {'Δ R²':>9} {'MAE':>8}")
print(f"  {'─'*52}")
for name, res in global_results.items():
    dr2  = res["r2"] - RECORD_R2
    icon = "* " if res["r2"] == max(r["r2"] for r in global_results.values()) else "  "
    print(f"  {icon} {name:<20} {res['r2']:>8.4f} {dr2:>+9.4f} {res['mae']:>8.4f}")
print(f"  {'─'*52}")
print(f"     {'Record v7':20} {RECORD_R2:>8.4f}  {'':>9} {RECORD_MAE:>8.4f}")

print(f"\n  ── Specialized Models by Niche ──")
print(f"  {'Niche':<12} {'N_train':>8} {'Best R²':>9} {'Δ global':>9} {'Status'}")
print(f"  {'─'*60}")
best_niche_r2 = 0.0
best_niche    = None
for niche, res in sorted(niche_results.items(), key=lambda x: x[1]["best_r2"], reverse=True):
    delta = res["best_r2"] - RECORD_R2
    if res["best_r2"] >= 0.40:
        status = " TARGET ≥ 0.40 ACHIEVED !"
    elif res["best_r2"] > RECORD_R2:
        status = f" PASS +{delta:.4f}"
    else:
        status = f" FAIL {delta:.4f}"
    print(f"  {niche:<12} {res['n_train']:>8,} {res['best_r2']:>9.4f} {delta:>+9.4f}  {status}")
    if res["best_r2"] > best_niche_r2:
        best_niche_r2 = res["best_r2"]
        best_niche    = niche

print(f"\n   Best niche : {best_niche} (R²={best_niche_r2:.4f})")
if best_niche_r2 >= 0.40:
    print(f"   TARGET R² ≥ 0.40 ACHIEVED on niche '{best_niche}' !")
elif best_niche_r2 > RECORD_R2:
    print(f"   Local improvement confirmed : +{best_niche_r2 - RECORD_R2:.4f}")
else:
    print(f"   Persistent plateau — ideas: increase dataset, try KMeans clustering")

# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION v8
# ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 18))
fig.patch.set_facecolor("#0d0d0d")
fig.suptitle(
    "Stacking v8 — Niche Segmentation | Comparative Analysis",
    fontsize=16, color="white", fontweight="bold", y=0.98
)

P = dict(
    ax_bg="#1a1a1a", grid="#2d2d2d", text="#f0f0f0",
    gold="#ffbe0b", record="#ff4d6d", target="#00b4d8",
)
NICHE_COLORS = {
    "Cooking": "#ff6b6b", "Study": "#4ecdc4", "Comedy": "#ffe66d",
    "Fitness": "#a8e6cf", "Tech": "#88d8b0", "Finance": "#c7f2a4",
    "Unknown": "#666666",
}

gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# ── Plot 1 : Niche distribution ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(P["ax_bg"])
niches_sorted = niche_counts.sort_values(ascending=True)
colors_pie    = [NICHE_COLORS.get(n, "#888") for n in niches_sorted.index]
bars = ax1.barh(niches_sorted.index, niches_sorted.values, color=colors_pie, alpha=0.85)
for bar, val in zip(bars, niches_sorted.values):
    ax1.text(val + 10, bar.get_y() + bar.get_height()/2,
             f"{val:,}  ({val/len(df)*100:.0f}%)", va="center",
             color=P["text"], fontsize=8)
ax1.set_title("Niche Distribution", color=P["text"], fontweight="bold")
ax1.tick_params(colors=P["text"])
ax1.xaxis.grid(True, color=P["grid"])
for sp in ax1.spines.values(): sp.set_edgecolor(P["grid"])
ax1.set_xlabel("Number of videos", color=P["text"])

# ── Plot 2 : Classification source ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(P["ax_bg"])
src_labels = source_counts.index.tolist()
src_vals   = source_counts.values.tolist()
src_colors = ["#3a86ff", "#06d6a0", "#ffbe0b", "#ef476f"][:len(src_labels)]
wedges, texts, autotexts = ax2.pie(
    src_vals, labels=src_labels, colors=src_colors,
    autopct="%1.0f%%", pctdistance=0.8, startangle=90,
    textprops={"color": P["text"], "fontsize": 9}
)
for at in autotexts: at.set_color("#0d0d0d")
ax2.set_title("Classification Source", color=P["text"], fontweight="bold")

# ── Plot 3 : μ target_log per niche ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(P["ax_bg"])
niches_list = list(niche_stats.keys())
mu_vals     = [niche_stats[n]["mu_target"] for n in niches_list]
std_vals    = [niche_stats[n]["std_target"] for n in niches_list]
nc          = [NICHE_COLORS.get(n, "#888") for n in niches_list]
xs          = np.arange(len(niches_list))
ax3.bar(xs, mu_vals, yerr=std_vals, color=nc, alpha=0.85, capsize=4, error_kw={"ecolor": "#888"})
ax3.axhline(df[TARGET].mean(), color=P["gold"], linewidth=1.5, linestyle="--", label="Global mean")
ax3.set_xticks(xs)
ax3.set_xticklabels(niches_list, rotation=30, ha="right", color=P["text"], fontsize=8)
ax3.set_title("μ target_log by Niche (± σ)", color=P["text"], fontweight="bold")
ax3.tick_params(colors=P["text"])
ax3.yaxis.grid(True, color=P["grid"])
ax3.legend(fontsize=7, framealpha=0.2, labelcolor=P["text"])
for sp in ax3.spines.values(): sp.set_edgecolor(P["grid"])

# ── Plot 4 : Global v8 R² vs record ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(P["ax_bg"])
gnames  = list(global_results.keys())
gr2vals = [global_results[n]["r2"] for n in gnames]
gc      = ["#3a86ff", "#06d6a0", "#8338ec", "#ef476f"][:len(gnames)]
bars = ax4.bar(gnames, gr2vals, color=gc, alpha=0.88)
best_g_idx = np.argmax(gr2vals)
bars[best_g_idx].set_edgecolor(P["gold"]); bars[best_g_idx].set_linewidth(2.5)
ax4.axhline(RECORD_R2, color=P["record"], linewidth=1.5, linestyle="--", label=f"Record v7 ({RECORD_R2})")
ax4.axhline(0.40, color=P["target"], linewidth=1.0, linestyle=":", label="Target 0.40")
for bar, val in zip(bars, gr2vals):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}",
             ha="center", va="bottom", color=P["text"], fontsize=8, fontweight="bold")
ax4.set_title("Global Models v8 R²", color=P["text"], fontweight="bold")
ax4.tick_params(colors=P["text"], labelsize=8)
ax4.yaxis.grid(True, color=P["grid"])
ax4.legend(fontsize=7, framealpha=0.2, labelcolor=P["text"])
for sp in ax4.spines.values(): sp.set_edgecolor(P["grid"])
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20, ha="right")

# ── Plot 5 : Specialized R² per niche ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1:])
ax5.set_facecolor(P["ax_bg"])
if niche_results:
    niche_names = list(niche_results.keys())
    for offset, model_type in enumerate(["XGBoost", "LightGBM", "Stacking"]):
        r2s = []
        for n in niche_names:
            r2s.append(niche_results[n]["models"].get(model_type, {}).get("r2", np.nan))
        xs = np.arange(len(niche_names)) + offset * 0.25
        valid = [(x, r) for x, r in zip(xs, r2s) if not np.isnan(r)]
        if valid:
            xv, rv = zip(*valid)
            mc = ["#3a86ff", "#06d6a0", "#ffbe0b"][offset]
            ax5.bar(xv, rv, width=0.22, color=mc, alpha=0.85, label=model_type)
    ax5.axhline(RECORD_R2, color=P["record"], linewidth=1.5, linestyle="--", label=f"Global record ({RECORD_R2})")
    ax5.axhline(0.40, color=P["target"], linewidth=1.0, linestyle=":", label="Target 0.40")
    ax5.set_xticks(np.arange(len(niche_names)) + 0.25)
    ax5.set_xticklabels(niche_names, color=P["text"], fontsize=9)
    ax5.set_title("Specialized Models R² by Niche vs Global", color=P["text"], fontweight="bold")
    ax5.tick_params(colors=P["text"])
    ax5.yaxis.grid(True, color=P["grid"])
    ax5.legend(fontsize=8, framealpha=0.2, labelcolor=P["text"])
    for sp in ax5.spines.values(): sp.set_edgecolor(P["grid"])

# ── Plot 6 : R² gain summary per niche ──────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor(P["ax_bg"])
all_niche_data = sorted(
    [(n, res["best_r2"] - RECORD_R2) for n, res in niche_results.items()],
    key=lambda x: x[1], reverse=True
)
if all_niche_data:
    nn, nd = zip(*all_niche_data)
    bar_c  = [NICHE_COLORS.get(n, "#888") for n in nn]
    xs = np.arange(len(nn))
    ax6.bar(xs, nd, color=bar_c, alpha=0.85)
    ax6.axhline(0, color=P["record"], linewidth=1.2, linestyle="--")
    ax6.axhline(0.40 - RECORD_R2, color=P["target"], linewidth=1.0, linestyle=":", label="Target +0.40")
    for x, val, name in zip(xs, nd, nn):
        ax6.text(x, val + (0.001 if val >= 0 else -0.003),
                 f"{val:+.4f}", ha="center", va="bottom" if val >= 0 else "top",
                 color=P["text"], fontsize=9, fontweight="bold")
    ax6.set_xticks(xs)
    ax6.set_xticklabels(nn, color=P["text"], fontsize=10)
    ax6.set_ylabel("ΔR² vs Global (0.3332)", color=P["text"])
    ax6.set_title("R² Gain of Specialized Models vs Global Record", color=P["text"], fontweight="bold", fontsize=12)
    ax6.tick_params(colors=P["text"])
    ax6.yaxis.grid(True, color=P["grid"])
    ax6.legend(fontsize=8, framealpha=0.2, labelcolor=P["text"])
    for sp in ax6.spines.values(): sp.set_edgecolor(P["grid"])

plt.savefig("stacking_v8_niche.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n Chart saved → stacking_v8_niche.png")
plt.close()

# ── CSV Export of results by niche ────────────────────────────────
niche_report_rows = []
for niche, res in niche_results.items():
    for model_name, mres in res["models"].items():
        niche_report_rows.append({
            "niche": niche,
            "model": model_name,
            "r2": mres["r2"],
            "mae": mres["mae"],
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "delta_r2_vs_global": mres["r2"] - RECORD_R2,
            "beats_global": mres["r2"] > RECORD_R2,
            "reaches_040": mres["r2"] >= 0.40,
        })
if niche_report_rows:
    pd.DataFrame(niche_report_rows).sort_values("r2", ascending=False).to_csv(
        "niche_results_v8.csv", index=False
    )
    print(" CSV Report → niche_results_v8.csv")

# Export of the classified dataframe
df[["creator_id", "video_rank", "niche", "niche_source", TARGET]].to_csv(
    "df_classified_v8.csv", index=False
)
print(" Classified dataset → df_classified_v8.csv")
print("\n Stacking v8 completed.\n")