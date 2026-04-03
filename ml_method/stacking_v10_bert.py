"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TIKTOK VIRALITY — STACKING v10 : BERT HYBRID                   ║
║           BERT Embeddings + Hybrid Speech/Non-Speech Stacking             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  What's new in v10 vs v9:                                                     ║
║  • TF-IDF + LSA 30D -> BERT (paraphrase-multilingual-MiniLM-L12-v2)        ║
║    + SVD 20D (fit on train only, anti-leakage)                      ║
║  • Embedding cache -> bert_embeddings_cache.npy (avoids re-encoding)       ║
║  • Hybrid stacking:                                                       ║
║        - GLOBAL model: all videos (tabular features + BERT)    ║
║        - SPEECH model: spoken segment only (34.6%)                  ║
║        - Meta-learner: combines global + speech via has_speech             ║
║  • Top-45 feature selection preserved (fast CatBoost)                   ║
║  • All v9 fixes preserved:                                  ║
║        - creator_id extracted from webVideoUrl                                       ║
║        - Expanding window target encoding (causal, anti-leakage)            ║
║        - early_stopping_rounds removed from cross_val_predict                  ║
║  Record to beat: R² = 0.3310 (global) | R² = 0.4036 (Comedy)           ║
║  Goal v10: R² >= 0.40 global | R² >= 0.45 spoken segment               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import ast
import re
import time
import warnings
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RECORD_R2_GLOBAL  = 0.3310        # v8 corrigé
RECORD_R2_COMEDY  = 0.4036        # v8 corrigé
RECORD_MAE        = 0.2907
TARGET            = "target_log"
N_OPTUNA_TRIALS   = 40
N_OOF_FOLDS       = 5
TOP_K_FEATURES    = 45
N_BERT_SVD        = 20            # composantes SVD post-BERT (réduction de 384→20)
BERT_MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
BERT_CACHE_PATH   = "bert_embeddings_cache.npy"
BERT_BATCH_SIZE   = 64
MIN_NICHE_SAMPLES = 80
SMOOTH_K          = 5

print("=" * 72)
print("  STACKING v10 — BERT HYBRID (Speech + Non-Speech)")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data_subtittles.csv")
# Extract creator_id before sorting to enable creator grouping
df["creator_id"] = df["webVideoUrl"].str.extract(r'tiktok\.com/@([^/]+)/video')
# Sort by creator then chronologically: each creator's videos remain
# grouped, which is essential for momentum and OOF target encoding.
# Warning: Do not sort by followers alone: it disperses videos from the same
#     creator and breaks target encoding correlation (corr ~= 0.04).
df = df.sort_values(["creator_id", "video_rank"]).reset_index(drop=True)
df["creator_id_int"] = df["creator_id"].astype("category").cat.codes

# Sanity check for creator_id_int
n_creators = df["creator_id_int"].nunique()
print(f"{n_creators} creators | {len(df):,} videos")
assert n_creators > 100, "Warning: creator_id_int suspicious — check sorting"

# ─────────────────────────────────────────────────────────────────────────────
# 2. CASCADE CLASSIFICATION (identical to v8 — preserved in full)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 1] CASCADE CLASSIFICATION — 6 Niches (preserved from v8)")
print("=" * 72)

HASHTAG_MAP = {
    "cooking":"Cooking","recipe":"Cooking","food":"Cooking","foodtok":"Cooking",
    "chef":"Cooking","baking":"Cooking","kitchen":"Cooking","meal":"Cooking",
    "eat":"Cooking","dinner":"Cooking","lunch":"Cooking","breakfast":"Cooking",
    "vegan":"Cooking","healthy":"Cooking","nutrition":"Cooking","recette":"Cooking",
    "cuisine":"Cooking","manger":"Cooking","foodie":"Cooking","tasty":"Cooking",
    "yummy":"Cooking","cookwithme":"Cooking","cooktok":"Cooking","homecooking":"Cooking",
    "study":"Study","studytok":"Study","studywithme":"Study","studymotivation":"Study",
    "school":"Study","college":"Study","university":"Study","exam":"Study",
    "homework":"Study","learning":"Study","education":"Study","student":"Study",
    "revision":"Study","notes":"Study","flashcards":"Study","studyaesthetic":"Study",
    "pomodoro":"Study","studycheck":"Study","etude":"Study","cours":"Study",
    "comedy":"Comedy","funny":"Comedy","humor":"Comedy","meme":"Comedy",
    "lol":"Comedy","joke":"Comedy","prank":"Comedy","skit":"Comedy",
    "relatable":"Comedy","trending":"Comedy","viral":"Comedy","entertainment":"Comedy",
    "comedytok":"Comedy","laughing":"Comedy","sketch":"Comedy","parody":"Comedy",
    "funnytiktok":"Comedy",
    "fitness":"Fitness","workout":"Fitness","gym":"Fitness","exercise":"Fitness",
    "sport":"Fitness","training":"Fitness","fitnessmotivation":"Fitness",
    "bodybuilding":"Fitness","cardio":"Fitness","running":"Fitness","yoga":"Fitness",
    "pilates":"Fitness","fitspo":"Fitness","musculation":"Fitness","crossfit":"Fitness",
    "hiit":"Fitness","weightloss":"Fitness","fitnesscheck":"Fitness","gymtok":"Fitness",
    "tech":"Tech","technology":"Tech","coding":"Tech","programming":"Tech",
    "software":"Tech","ai":"Tech","developer":"Tech","gaming":"Tech","gamer":"Tech",
    "techtok":"Tech","apple":"Tech","android":"Tech","python":"Tech",
    "javascript":"Tech","startup":"Tech","innovation":"Tech","robotics":"Tech",
    "cybersecurity":"Tech","informatique":"Tech",
    "finance":"Finance","investing":"Finance","stocks":"Finance","crypto":"Finance",
    "bitcoin":"Finance","fintech":"Finance","money":"Finance","wealth":"Finance",
    "business":"Finance","entrepreneur":"Finance","sidehustle":"Finance",
    "passive":"Finance","trading":"Finance","bourse":"Finance","investment":"Finance",
    "financetok":"Finance","budget":"Finance","frugal":"Finance","richlife":"Finance",
}

CAPTION_KW = {
    "Cooking":[r"\brecipe\b",r"\bcooking\b",r"\bchef\b",r"\bbaking\b",r"\bfood\b",
               r"\bmeal\b",r"\bdinner\b",r"\blunch\b",r"\bbreakfast\b",r"\bvegan\b",
               r"\brecette\b",r"\bcuisine\b",r"\bmanger\b",r"\bingredients?\b",
               r"\bcook\b",r"\bdelicious\b",r"\btasty\b"],
    "Study":[r"\bstudy\b",r"\bstudying\b",r"\bschool\b",r"\bcollege\b",
             r"\bexam\b",r"\bhomework\b",r"\bnotes\b",r"\blearning\b",
             r"\bstudent\b",r"\buniversity\b",r"\bétude\b",r"\bcours\b",
             r"\bdevoirs\b",r"\brévision\b",r"\blecture\b",r"\bflashcard\b"],
    "Comedy":[r"\bfunny\b",r"\blol\b",r"\bhaha\b",r"\bcomedy\b",r"\bjoke\b",
              r"\bprank\b",r"\bskit\b",r"\bmeme\b",r"\bhumor\b",r"\blaugh\b",
              r"\brelatable\b",r"\bwhen you\b",r"\bpov\b"],
    "Fitness":[r"\bworkout\b",r"\bgym\b",r"\bfitness\b",r"\bexercise\b",
               r"\btraining\b",r"\bcardio\b",r"\byoga\b",r"\brunning\b",
               r"\bpilates\b",r"\bbodybuilding\b",r"\bfat\s?loss\b",
               r"\bmusculation\b",r"\bséance\b",r"\bentraînement\b"],
    "Tech":[r"\bcoding\b",r"\bprogramming\b",r"\bcode\b",r"\bgaming\b",
            r"\btechnology\b",r"\bai\b",r"\bapple\b",r"\biphone\b",
            r"\bsoftware\b",r"\bdeveloper\b",r"\btech\b",r"\bbot\b",
            r"\binformatique\b",r"\balgorithm\b",r"\bdéveloppeur\b"],
    "Finance":[r"\bstocks?\b",r"\bcrypto\b",r"\bbitcoin\b",r"\binvest",
               r"\btrading\b",r"\bmoney\b",r"\bbourse\b",r"\bfinance\b",
               r"\bbusiness\b",r"\bentrepreneur\b",r"\bwealth\b",
               r"\bbudget\b",r"\bpassive\s?income\b",r"\bargent\b"],
}

def parse_hashtags(raw):
    try:
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if isinstance(tags, list):
            return [t["name"].lower() for t in tags if isinstance(t, dict) and "name" in t]
    except Exception:
        pass
    return []

def classify_from_hashtags(tags):
    votes = Counter()
    for tag in tags:
        clean = re.sub(r"[^a-z0-9]", "", tag.lower())
        if clean in HASHTAG_MAP:
            votes[HASHTAG_MAP[clean]] += 1
    return votes.most_common(1)[0][0] if votes else None

def classify_from_caption(text):
    if not isinstance(text, str) or not text.strip():
        return None
    text_lower = text.lower()
    scores = {}
    for niche, patterns in CAPTION_KW.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > 0:
            scores[niche] = score
    return max(scores, key=scores.get) if scores else None

def classify_cascade(row, creator_majority):
    tags  = parse_hashtags(row["hashtag"])
    niche = classify_from_hashtags(tags)
    if niche:
        return niche, "hashtag"
    niche = classify_from_caption(row["caption"])
    if niche:
        return niche, "caption"
    cid = row["creator_id_int"]
    if cid in creator_majority:
        return creator_majority[cid], "creator_heritage"
    return "Unknown", "unclassified"

print("\n   Pass 1 — P1+P2 Classification...")
p1p2 = []
for _, row in df.iterrows():
    tags  = parse_hashtags(row["hashtag"])
    niche = classify_from_hashtags(tags) or classify_from_caption(row["caption"])
    p1p2.append(niche)
df["_niche_p1p2"] = p1p2

creator_majority = {}
for cid, grp in df[df["_niche_p1p2"].notna()].groupby("creator_id_int"):
    counts = grp["_niche_p1p2"].value_counts()
    if len(counts) > 0 and counts.iloc[0] >= 2:
        creator_majority[cid] = counts.index[0]

print(f"   Heritage available for {len(creator_majority)} creators")
print("   Pass 2 — Final classification with heritage...")

niche_results_cascade = df.apply(lambda r: classify_cascade(r, creator_majority), axis=1)
df["niche"]        = niche_results_cascade.apply(lambda x: x[0])
df["niche_source"] = niche_results_cascade.apply(lambda x: x[1])
df.drop(columns=["_niche_p1p2"], inplace=True)

niche_counts = df["niche"].value_counts()
classified_pct = (df["niche"] != "Unknown").sum() / len(df) * 100
print(f"\n   Classification rate : {classified_pct:.1f}%")
for niche, cnt in niche_counts.items():
    print(f"    {niche:<12} : {cnt:>5,}  ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING v8 (identical pipeline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 2] FEATURE ENGINEERING v8 (pipeline preserved)")
print("=" * 72)

df["_tags"]      = df["hashtag"].apply(parse_hashtags)
df["n_hashtags"] = df["_tags"].apply(len)
df["has_fyp"]    = df["_tags"].apply(lambda t: int(any("fyp"    in x for x in t)))
df["has_viral"]  = df["_tags"].apply(lambda t: int(any("viral"  in x for x in t)))
df["has_foryou"] = df["_tags"].apply(lambda t: int(any("foryou" in x for x in t)))
df.drop(columns=["_tags"], inplace=True)

df["caption_len"]            = df["caption"].fillna("").str.len()
df["has_emoji"]              = df["caption"].fillna("").apply(lambda s: int(any(ord(c) > 127 for c in s)))
df["has_question_cap"]       = df["caption"].fillna("").str.contains(r"\?").astype(int)
df["has_exclamation"]        = df["caption"].fillna("").str.contains(r"!").astype(int)
df["viral_potential"]        = df["hist_p90_views"] / (df["hist_median_views"] + 1)
df["engagement_total_hist"]  = df["hist_like_rate"] + df["hist_comment_rate"] + df["hist_share_rate"]
df["is_peak_hour"]           = df["hour"].between(17, 22).astype(int)
df["follower_tier"]          = np.log1p(df["followers"])
df["views_efficiency_trend"] = df["hist_p70_views"] / (df["hist_median_views"] + 1)

# Niche one-hot
niche_dummies = pd.get_dummies(df["niche"], prefix="niche").astype(int)
df = pd.concat([df, niche_dummies], axis=1)
NICHE_COLS = [c for c in df.columns if c.startswith("niche_") and c != "niche_source"]

# Momentum
print("  Momentum...")

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

grp     = df.groupby("creator_id_int")["explosion_score"]
shifted = grp.shift(1)

df["momentum_3"]      = shifted.groupby(df["creator_id_int"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df["trend_slope"]     = shifted.groupby(df["creator_id_int"]).transform(lambda s: rolling_slope(s, window=5))
df["consistency"]     = shifted.groupby(df["creator_id_int"]).transform(lambda s: s.rolling(5, min_periods=2).std())
df["momentum_ratio"]  = df["momentum_3"] / (df["hist_median_views"] + 1)
df["trend_direction"] = np.sign(df["trend_slope"].fillna(0)).astype(int)
df["volatility_tier"] = df["consistency"] / (df["hist_median_views"] + 1)
df["momentum_7"]      = shifted.groupby(df["creator_id_int"]).transform(lambda s: s.rolling(7, min_periods=2).mean())
df["accel"]           = df["momentum_3"] - df["momentum_7"]
df["peak_score"]      = shifted.groupby(df["creator_id_int"]).transform(lambda s: s.rolling(5, min_periods=1).max())
recent_min            = shifted.groupby(df["creator_id_int"]).transform(lambda s: s.rolling(5, min_periods=1).min())
df["recovery"]        = (df["momentum_3"] - recent_min) / (df["peak_score"] - recent_min + 1)
df["streak_up"]       = shifted.groupby(df["creator_id_int"]).transform(lambda s: count_streak(s))
df["momentum_norm"]   = df["momentum_3"] / (df["hist_p90_views"] + 1)

MOMENTUM_COLS = ["momentum_3","trend_slope","consistency","momentum_ratio",
                 "trend_direction","volatility_tier","momentum_7","accel",
                 "peak_score","recovery","streak_up","momentum_norm"]
df[MOMENTUM_COLS] = df[MOMENTUM_COLS].fillna(df[MOMENTUM_COLS].median())

# NLP caption (TextBlob — conservé v8)
print("   Caption NLP (TextBlob)...")
try:
    from textblob import TextBlob
    def get_sentiment(text):
        try:
            b = TextBlob(str(text))
            return b.sentiment.polarity, b.sentiment.subjectivity
        except:
            return 0.0, 0.0
    sents = df["caption"].fillna("").apply(get_sentiment)
    df["sentiment_polarity"]     = sents.apply(lambda x: x[0])
    df["sentiment_subjectivity"] = sents.apply(lambda x: x[1])
    df["emotional_intensity"]    = df["sentiment_polarity"].abs()
    df["word_count_caption"]     = df["caption"].fillna("").apply(lambda s: len(s.split()))
    NLP_COLS = ["sentiment_polarity","sentiment_subjectivity","emotional_intensity","word_count_caption"]
    print("     TextBlob OK")
except ImportError:
    print("     Warning:  TextBlob missing — NLP caption features set to 0")
    for col in ["sentiment_polarity","sentiment_subjectivity","emotional_intensity","word_count_caption"]:
        df[col] = 0.0
    NLP_COLS = ["sentiment_polarity","sentiment_subjectivity","emotional_intensity","word_count_caption"]

# Cross-features v8
print("  Cross-features v8...")
df["mom3_x_tier"]          = df["momentum_3"]    * df["follower_tier"]
df["accel_x_viral"]        = df["accel"]          * df["viral_potential"]
df["recovery_x_hist"]      = df["recovery"]       * df["hist_median_views"]
df["streak_x_engage"]      = df["streak_up"]      * df["engagement_total_hist"]
df["peak_x_p90"]           = df["peak_score"]     / (df["hist_p90_views"] + 1)
df["mom7_x_consist"]       = df["momentum_7"]     / (df["consistency"] + 0.1)
df["trend_x_duration"]     = df["trend_slope"]    * df["duration"]
df["norm_x_tier"]          = df["momentum_norm"]  * df["follower_tier"]
df["intensity_x_momentum"] = df["emotional_intensity"] * df["momentum_3"]
df["polarity_x_viral"]     = df["sentiment_polarity"]  * df["viral_potential"]

CROSS_COLS_V8 = ["mom3_x_tier","accel_x_viral","recovery_x_hist","streak_x_engage",
                 "peak_x_p90","mom7_x_consist","trend_x_duration","norm_x_tier",
                 "intensity_x_momentum","polarity_x_viral"]
df[CROSS_COLS_V8] = df[CROSS_COLS_V8].replace([np.inf, -np.inf], np.nan).fillna(df[CROSS_COLS_V8].median())

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPEECH FEATURE ENGINEERING (new in v9)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 3] SPEECH FEATURE ENGINEERING (v9 — new)")
print("=" * 72)

# Hook patterns (first 5 words — in English, TikTok EN dataset)
QUESTION_PAT = re.compile(
    r'\b(why|how|what|when|who|which|where|would|could|should|do you|are you|'
    r'is it|can you|did you|have you|will you)\b|\?',
    re.IGNORECASE
)
LIST_PAT = re.compile(
    r'\b(top|best|\d+\s*(?:ways?|tips?|steps?|reasons?|things?|mistakes?|hacks?|'
    r'facts?|secrets?|signs?|rules?|habits?|tricks?))\b',
    re.IGNORECASE
)
URGENCY_PAT = re.compile(
    r'\b(stop|wait|listen|attention|alert|breaking|now|today|immediately|'
    r'never|always|everybody|everyone|nobody)\b',
    re.IGNORECASE
)
PERSONAL_PAT = re.compile(
    r'\b(i |my |me |i\'m|i\'ve|i\'ll|i\'d|we |our |us )\b',
    re.IGNORECASE
)
POI_PAT = re.compile(
    r'\b(pov|tuto|tutorial|recipe|hack|trick|tip|secret|reveal|review|'
    r'react|explain|show|teach|learn|watch)\b',
    re.IGNORECASE
)
POSITIVE_PAT = re.compile(
    r'\b(amazing|incredible|perfect|great|excellent|best|love|awesome|'
    r'beautiful|fantastic|brilliant|outstanding|wonderful|superb)\b',
    re.IGNORECASE
)
NEGATIVE_PAT = re.compile(
    r'\b(horrible|worst|bad|terrible|awful|dangerous|warning|problem|'
    r'fail|wrong|never|stupid|mistake|error)\b',
    re.IGNORECASE
)

print("\n   Building Speech Features...")

# Cleaning subtitles
df["subtitles_clean"] = df["subtitles"].fillna("").str.strip()
df["has_speech"]      = (df["subtitles_clean"] != "").astype(int)

# Function to extract the first n words
def hook_words(text, n=5):
    if not text:
        return ""
    return " ".join(text.split()[:n])

# Computing textual features
df["hook_text"]   = df["subtitles_clean"].apply(hook_words)
df["word_count_speech"] = df["subtitles_clean"].apply(lambda t: len(t.split()) if t else 0)
df["char_count_speech"] = df["subtitles_clean"].str.len()

# Speech rate : mots par seconde
df["speech_rate"] = df["word_count_speech"] / df["duration"].clip(lower=1)
# Capping: outliers on very short videos
df["speech_rate"] = df["speech_rate"].clip(upper=df["speech_rate"].quantile(0.99))

# Hook features (on the first 5 words)
df["is_question"]  = df["hook_text"].apply(lambda t: int(bool(QUESTION_PAT.search(t))))
df["is_list"]      = df["hook_text"].apply(lambda t: int(bool(LIST_PAT.search(t))))
df["is_urgency"]   = df["hook_text"].apply(lambda t: int(bool(URGENCY_PAT.search(t))))
df["is_personal"]  = df["hook_text"].apply(lambda t: int(bool(PERSONAL_PAT.search(t))))
df["is_poi"]       = df["hook_text"].apply(lambda t: int(bool(POI_PAT.search(t))))

# Composite hook score (pondéré par corrélation empirique TikTok)
df["hook_score"] = (
    df["is_list"]     * 2.5 +
    df["is_question"] * 2.0 +
    df["is_urgency"]  * 1.5 +
    df["is_poi"]      * 1.5 +
    df["is_personal"] * 1.0
)

# Sentiment on full text
df["positive_count"]  = df["subtitles_clean"].apply(lambda t: len(POSITIVE_PAT.findall(t)))
df["negative_count"]  = df["subtitles_clean"].apply(lambda t: len(NEGATIVE_PAT.findall(t)))
df["sentiment_ratio"] = (df["positive_count"] - df["negative_count"]) / (df["word_count_speech"].clip(lower=1))

# Zero out numerical features for silent videos
SPEECH_NUMERIC_COLS = ["speech_rate","is_question","is_list","is_urgency","is_personal",
                       "is_poi","hook_score","positive_count","negative_count","sentiment_ratio",
                       "word_count_speech","char_count_speech"]
mask_no_speech = df["has_speech"] == 0
df.loc[mask_no_speech, SPEECH_NUMERIC_COLS] = 0

# NON_VERBAL token for TF-IDF (trees will create a dedicated branch)
df["subtitles_for_tfidf"] = df["subtitles_clean"].apply(
    lambda t: t if t else "NON_VERBAL"
)

print(f"  has_speech : {df['has_speech'].sum():,} videos with speech ({df['has_speech'].mean()*100:.1f}%)")
print(f"   Average speech rate (spoken) : {df.loc[df['has_speech']==1, 'speech_rate'].mean():.2f} mots/s")
print(f"   Question hook : {df.loc[df['has_speech']==1, 'is_question'].mean()*100:.1f}% of spoken videos")
print(f"   List hook    : {df.loc[df['has_speech']==1, 'is_list'].mean()*100:.1f}% of spoken videos")
print(f"   Urgency hook  : {df.loc[df['has_speech']==1, 'is_urgency'].mean()*100:.1f}% of spoken videos")

# ── Speech x Momentum Cross-features (new in v9) ──────────────────────────
df["speech_x_momentum"]   = df["has_speech"]   * df["momentum_3"]
df["hook_x_viral"]        = df["hook_score"]   * df["viral_potential"]
df["rate_x_engagement"]   = df["speech_rate"]  * df["engagement_total_hist"]
df["hook_x_tier"]         = df["hook_score"]   * df["follower_tier"]
df["sentiment_x_momentum"]= df["sentiment_ratio"] * df["momentum_3"]

CROSS_SPEECH_COLS = ["speech_x_momentum","hook_x_viral","rate_x_engagement",
                     "hook_x_tier","sentiment_x_momentum"]
df[CROSS_SPEECH_COLS] = df[CROSS_SPEECH_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. BERT EMBEDDINGS + SVD (v10 — replaces TF-IDF + LSA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  [PHASE 4] BERT Embeddings -> SVD {N_BERT_SVD}D (v10)")
print("=" * 72)

TRAIN_RANK_MIN, TRAIN_RANK_MAX = 11, 26

def load_bert_model():
    """Loads the multilingual BERT model (sentence-transformers)."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  📥 Loading BERT : {BERT_MODEL_NAME}...")
        model = SentenceTransformer(BERT_MODEL_NAME)
        print(f"   BERT loaded (dim={model.get_sentence_embedding_dimension()})")
        return model
    except ImportError:
        raise ImportError(
            "sentence-transformers missing.\n"
            "Install with: pip install sentence-transformers"
        )

def encode_texts_bert(model, texts, batch_size=BERT_BATCH_SIZE):
    """Encodes a list of texts into BERT embeddings, in batches."""
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine-ready, stabilise le SVD
    )
    print(f"   {len(texts):,} texts encoded in {time.time()-t0:.0f}s "
          f"| shape={embeddings.shape}")
    return embeddings

# ── Cache embeddings (évite re-encoding entre runs) ──────────────────────────
texts_for_bert = df["subtitles_for_tfidf"].tolist()
# subtitles_for_tfidf contains either the transcript or "NON_VERBAL"

if os.path.exists(BERT_CACHE_PATH):
    print(f"    BERT cache found : {BERT_CACHE_PATH}")
    bert_embeddings = np.load(BERT_CACHE_PATH)
    if bert_embeddings.shape[0] != len(df):
        print(f"  Warning:  Incomplete cache ({bert_embeddings.shape[0]} vs {len(df)}) — re-encoding")
        bert_model     = load_bert_model()
        bert_embeddings = encode_texts_bert(bert_model, texts_for_bert)
        np.save(BERT_CACHE_PATH, bert_embeddings)
        print(f"   Cache saved -> {BERT_CACHE_PATH}")
    else:
        print(f"   Cache loaded : {bert_embeddings.shape}")
else:
    bert_model      = load_bert_model()
    bert_embeddings = encode_texts_bert(bert_model, texts_for_bert)
    np.save(BERT_CACHE_PATH, bert_embeddings)
    print(f"   Cache saved -> {BERT_CACHE_PATH}")

# ── SVD post-BERT (fit on train only — anti-leakage) ──────────────────
train_mask_bert = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)

svd_bert = TruncatedSVD(n_components=N_BERT_SVD, random_state=42)
bert_train_reduced = svd_bert.fit_transform(bert_embeddings[train_mask_bert])
bert_full_reduced  = svd_bert.transform(bert_embeddings)

explained_var_bert = svd_bert.explained_variance_ratio_.sum()
print(f"   Explained variance ({N_BERT_SVD}D SVD post-BERT) : {explained_var_bert:.1%}")

BERT_COLS = [f"bert_{i}" for i in range(N_BERT_SVD)]
bert_df   = pd.DataFrame(bert_full_reduced, columns=BERT_COLS, index=df.index)
df        = pd.concat([df, bert_df], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL FEATURE SET v10
# ─────────────────────────────────────────────────────────────────────────────
STATIC_FEATURES = [
    "followers","duration","hour","weekday","musicOriginal",
    "hist_median_views","hist_p70_views","hist_p90_views",
    "hist_like_rate","hist_comment_rate","hist_share_rate",
    "n_hashtags","has_fyp","has_viral","has_foryou",
    "caption_len","has_emoji","has_question_cap","has_exclamation",
    "viral_potential","engagement_total_hist","is_peak_hour",
    "follower_tier","views_efficiency_trend",
]

SPEECH_FEATURES = (
    SPEECH_NUMERIC_COLS +
    CROSS_SPEECH_COLS +
    ["has_speech"]
)

ALL_FEATURES_V10 = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS_V8 +
    NLP_COLS + NICHE_COLS + SPEECH_FEATURES + BERT_COLS
)

print(f"\n   v10 total feature set : {len(ALL_FEATURES_V10)} features")
print(f"     of which Speech Features  : {len(SPEECH_FEATURES)}")
print(f"     of which BERT SVD         : {len(BERT_COLS)}")
print(f"     of which Momentum         : {len(MOMENTUM_COLS)}")
print(f"     of which Cross-features   : {len(CROSS_COLS_V8 + CROSS_SPEECH_COLS)}")
print(f"     of which Niche one-hot    : {len(NICHE_COLS)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

print(f"\n   Split: Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")
print(f"     has_speech train : {df.loc[train_mask, 'has_speech'].mean()*100:.1f}%")
print(f"     has_speech test  : {df.loc[test_mask,  'has_speech'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. TARGET ENCODING BY ROLLING RANK (anti-leakage adapted to this dataset)
# ─────────────────────────────────────────────────────────────────────────────
# Structural constraint: the dataset only contains ranks 11-30,
# each creator having exactly 20 videos. There are no "past videos"
# outside train/val/test, making per-fold OOF target encoding impossible:
# GroupKFold isolates each creator entirely in val -> never seen in train
# -> 100% fallback global_mean -> corr=0.
#
# Solution: for each video at rank R, we encode with the smoothed mean of
# videos from the SAME creator at ranks < R (already observed chronologically).
# This is a strictly causal expanding window encoding: rank 12 sees rank 11,
# rank 15 sees ranks 11-14, etc.  No future information is used.
print("\n" + "─" * 72)
print("  [LEVER A] Sliding rank target encoding (causal, anti-leakage)")
print("─" * 72)

global_mean = df.loc[train_mask, TARGET].mean()

# Sort by creator then rank to ensure causal order
df = df.sort_values(["creator_id_int", "video_rank"]).reset_index(drop=True)

# Smoothed expanding mean (Bayesian smoothing) per creator, shifted by 1 rank
def expanding_te(group, smooth_k, global_mean):
    target_vals = group[TARGET].values
    ranks       = group["video_rank"].values
    te_vals     = np.full(len(target_vals), global_mean)
    cumsum   = 0.0
    cumcount = 0
    for i in range(len(target_vals)):
        # We only use strictly earlier ranks
        te_vals[i] = (cumsum + smooth_k * global_mean) / (cumcount + smooth_k)
        cumsum   += target_vals[i]
        cumcount += 1
    return pd.Series(te_vals, index=group.index)

df["creator_target_mean"] = (
    df.groupby("creator_id_int", group_keys=False)
      .apply(lambda g: expanding_te(g, SMOOTH_K, global_mean))
)

# Recomputing masks after re-sort
train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

te_corr = np.corrcoef(df.loc[train_mask, "creator_target_mean"],
                      df.loc[train_mask, TARGET])[0, 1]
# Leakage check: the first video of each creator receives global_mean
# (cumcount=0), so corr < 1.0 is structurally guaranteed.
assert te_corr < 0.999, f"❌ Leakage detected (corr={te_corr:.4f})"
print(f"   creator_target_mean (expanding OOF, k={SMOOTH_K}) | corr={te_corr:.4f}"
      f"  {' OK' if te_corr > 0.05 else 'Warning:  low — normal for rank 11 (1 past video)'}")

ALL_FEATURES_V10_FINAL = ALL_FEATURES_V10 + ["creator_target_mean"]

# ─────────────────────────────────────────────────────────────────────────────
# 9. TOP-K FEATURE SELECTION (fast CatBoost)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print(f"  [LEVER B] Feature Selection — Top {TOP_K_FEATURES}")
print("─" * 72)

X_train_full = df.loc[train_mask, ALL_FEATURES_V10_FINAL].astype(float)
y_train      = df.loc[train_mask, TARGET]
X_val_full   = df.loc[val_mask,   ALL_FEATURES_V10_FINAL].astype(float)
y_val        = df.loc[val_mask,   TARGET]
X_test_full  = df.loc[test_mask,  ALL_FEATURES_V10_FINAL].astype(float)
y_test       = df.loc[test_mask,  TARGET]

cat_selector = CatBoostRegressor(
    iterations=3000, learning_rate=0.05, depth=6,
    early_stopping_rounds=100, random_state=42, verbose=0
)
cat_selector.fit(X_train_full, y_train, eval_set=(X_val_full, y_val))

fi_series = pd.Series(
    cat_selector.get_feature_importance(),
    index=ALL_FEATURES_V10_FINAL
).sort_values(ascending=False)

SELECTED_FEATURES = fi_series.head(TOP_K_FEATURES).index.tolist()

print(f"\n  Top {TOP_K_FEATURES} selected features :")
speech_in_top  = 0
bert_in_top    = 0
for rank, (feat, imp) in enumerate(fi_series.head(TOP_K_FEATURES).items(), 1):
    tag = ""
    if feat == "creator_target_mean": tag = " ←  TARGET ENC"
    elif feat in SPEECH_FEATURES:     tag = " ←  SPEECH"; speech_in_top += 1
    elif feat in BERT_COLS:            tag = " ←  BERT"; bert_in_top += 1
    elif feat in NICHE_COLS:          tag = " ←   NICHE"
    elif feat in MOMENTUM_COLS:       tag = " ←  MOMENTUM"
    elif feat in NLP_COLS:            tag = " ←  NLP"
    elif feat in CROSS_COLS_V8:       tag = " ←  CROSS v8"
    elif feat in CROSS_SPEECH_COLS:   tag = " ←  CROSS speech"
    print(f"    #{rank:>2}  {feat:<35} {imp:>6.2f}%{tag}")

print(f"\n   Speech features in top {TOP_K_FEATURES} : {speech_in_top}")
print(f"   BERT SVD features in top {TOP_K_FEATURES}    : {bert_in_top}")

X_train = df.loc[train_mask, SELECTED_FEATURES].astype(float)
X_val   = df.loc[val_mask,   SELECTED_FEATURES].astype(float)
X_test  = df.loc[test_mask,  SELECTED_FEATURES].astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 10. A/B TEST: SPOKEN vs NON-SPOKEN SEGMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [A/B TEST] IMPACT OF SPEECH ON PREDICTION")
print("=" * 72)

# Base features for the A/B test (without LSA or speech)
BASE_FEATURES_AB = [f for f in SELECTED_FEATURES
                    if f not in BERT_COLS and f not in SPEECH_FEATURES]

ab_r2_results  = {}
ab_mae_results = {}

quick_model = LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=40, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

# ── Group A: Videos WITHOUT speech ────────────────────────────────────────────
mask_no  = df["has_speech"] == 0
df_no    = df[mask_no]
Xn_train = df_no.loc[df_no.index.isin(df.index[train_mask]), BASE_FEATURES_AB].astype(float)
yn_train = df_no.loc[df_no.index.isin(df.index[train_mask]), TARGET]
Xn_test  = df_no.loc[df_no.index.isin(df.index[test_mask]),  BASE_FEATURES_AB].astype(float)
yn_test  = df_no.loc[df_no.index.isin(df.index[test_mask]),  TARGET]

if len(Xn_train) > 50 and len(Xn_test) > 10:
    Xn_val  = df_no.loc[df_no.index.isin(df.index[val_mask]),  BASE_FEATURES_AB].astype(float)
    yn_val  = df_no.loc[df_no.index.isin(df.index[val_mask]),  TARGET]
    cb_ab   = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    quick_model.fit(Xn_train, yn_train, eval_set=[(Xn_val, yn_val)], callbacks=cb_ab)
    r2_no   = r2_score(yn_test, quick_model.predict(Xn_test))
    mae_no  = mean_absolute_error(yn_test, quick_model.predict(Xn_test))
    ab_r2_results[" Without speech (base)"] = r2_no
    ab_mae_results[" Without speech (base)"] = mae_no
    print(f"\n   GROUP A — Without speech  (train={len(Xn_train):,} | test={len(Xn_test):,})")
    print(f"     R² = {r2_no:.4f} | MAE = {mae_no:.4f}")

# ── Group B: With speech — BASE features only ────────────────────────
mask_yes  = df["has_speech"] == 1
df_yes    = df[mask_yes]
Xy_train  = df_yes.loc[df_yes.index.isin(df.index[train_mask]), BASE_FEATURES_AB].astype(float)
yy_train  = df_yes.loc[df_yes.index.isin(df.index[train_mask]), TARGET]
Xy_test   = df_yes.loc[df_yes.index.isin(df.index[test_mask]),  BASE_FEATURES_AB].astype(float)
yy_test   = df_yes.loc[df_yes.index.isin(df.index[test_mask]),  TARGET]

if len(Xy_train) > 30 and len(Xy_test) > 5:
    Xy_val   = df_yes.loc[df_yes.index.isin(df.index[val_mask]), BASE_FEATURES_AB].astype(float)
    yy_val   = df_yes.loc[df_yes.index.isin(df.index[val_mask]), TARGET]
    cb_ab2   = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    quick_model_b = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   num_leaves=40, subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1)
    quick_model_b.fit(Xy_train, yy_train, eval_set=[(Xy_val, yy_val)], callbacks=cb_ab2)
    r2_base   = r2_score(yy_test, quick_model_b.predict(Xy_test))
    mae_base  = mean_absolute_error(yy_test, quick_model_b.predict(Xy_test))
    ab_r2_results[" With speech (base)"] = r2_base
    ab_mae_results[" With speech (base)"] = mae_base
    print(f"\n   GROUP B — With speech, BASE only (train={len(Xy_train):,} | test={len(Xy_test):,})")
    print(f"     R² = {r2_base:.4f} | MAE = {mae_base:.4f}")

delta_ab = 0.0  # défaut si groupe C ne s'exécute pas

# ── Group C: With speech — BASE + SPEECH + LSA ────────────────────────────
SPEECH_SELECTED = [f for f in SELECTED_FEATURES
                   if f in SPEECH_FEATURES or f in BERT_COLS]
FULL_FEATURES_C = BASE_FEATURES_AB + SPEECH_SELECTED

Xy_train_c = df_yes.loc[df_yes.index.isin(df.index[train_mask]), FULL_FEATURES_C].astype(float)
Xy_test_c  = df_yes.loc[df_yes.index.isin(df.index[test_mask]),  FULL_FEATURES_C].astype(float)

if len(Xy_train_c) > 30 and len(Xy_test_c) > 5:
    Xy_val_c   = df_yes.loc[df_yes.index.isin(df.index[val_mask]), FULL_FEATURES_C].astype(float)
    yy_val_c   = df_yes.loc[df_yes.index.isin(df.index[val_mask]), TARGET]

    cb_ab3 = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    quick_model_c = LGBMRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                                   num_leaves=50, subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1)
    quick_model_c.fit(Xy_train_c, yy_train, eval_set=[(Xy_val_c, yy_val_c)], callbacks=cb_ab3)
    r2_full  = r2_score(yy_test, quick_model_c.predict(Xy_test_c))
    mae_full = mean_absolute_error(yy_test, quick_model_c.predict(Xy_test_c))
    ab_r2_results[" With speech (+speech+LSA)"] = r2_full
    ab_mae_results[" With speech (+speech+LSA)"] = mae_full
    delta_ab = r2_full - r2_base if " With speech (base)" in ab_r2_results else 0
    print(f"\n   GROUP C — With speech, BASE + SPEECH + LSA (train={len(Xy_train_c):,} | test={len(Xy_test_c):,})")
    print(f"     R² = {r2_full:.4f} | MAE = {mae_full:.4f}")
    print(f"     Speech Features gain : {delta_ab:+.4f}")
    reached_045 = " TARGET R²>=0.45 REACHED!" if r2_full >= 0.45 else f" {r2_full:.4f} (objectif: 0.45)"
    print(f"     {reached_045}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. GLOBAL STACKING v10 (XGBoost + LightGBM + CatBoost + Ridge L2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 5] GLOBAL STACKING v10 (full features)")
print("=" * 72)

global_results_v10 = {}

def build_meta_features(p1, p2, p3=None):
    if p3 is not None:
        return np.column_stack([
            p1, p2, p3,
            (p1 + p2 + p3) / 3,
            np.abs(p1 - p3),
            np.abs(p2 - p3),
            np.abs(p1 - p2),
        ])
    return np.column_stack([p1, p2, (p1 + p2) / 2, np.abs(p1 - p2)])

# ── XGBoost ────────────────────────────────────────────────────────────────
print(f"\n    XGBoost v10 ({N_OPTUNA_TRIALS} Optuna trials)...")

def xgb_objective(trial):
    params = dict(
        max_depth        = trial.suggest_int  ("max_depth",        3, 7),
        learning_rate    = trial.suggest_float("learning_rate",    0.002, 0.05, log=True),
        n_estimators     = trial.suggest_int  ("n_estimators",     3000, 12000, step=1000),
        subsample        = trial.suggest_float("subsample",        0.65, 0.95),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.50, 0.90),
        min_child_weight = trial.suggest_int  ("min_child_weight", 2, 15),
        reg_alpha        = trial.suggest_float("reg_alpha",        0.01, 5.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",       0.5, 10.0, log=True),
        gamma            = trial.suggest_float("gamma",            0.0, 0.5),
    )
    m = XGBRegressor(**params, tree_method="hist", early_stopping_rounds=200,
                     eval_metric="mae", verbosity=0, random_state=42)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return -r2_score(y_val, m.predict(X_val))

t0 = time.time()
xgb_study = optuna.create_study(direction="minimize",
                                  sampler=TPESampler(seed=42),
                                  pruner=HyperbandPruner())
xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_xgb = xgb_study.best_params

xgb_model = XGBRegressor(**best_xgb, tree_method="hist", early_stopping_rounds=200,
                          eval_metric="mae", verbosity=0, random_state=42)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb  = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"  R²={r2_xgb:.4f} | MAE={mae_xgb:.4f}  {'' if r2_xgb > RECORD_R2_GLOBAL else ''} | {time.time()-t0:.0f}s")
global_results_v10["XGBoost"] = {"r2": r2_xgb, "mae": mae_xgb}

# ── LightGBM ───────────────────────────────────────────────────────────────
print(f"\n    LightGBM v10 ({N_OPTUNA_TRIALS} Optuna trials)...")

def lgbm_objective(trial):
    params = dict(
        max_depth         = trial.suggest_int  ("max_depth",         3, 8),
        learning_rate     = trial.suggest_float("learning_rate",     0.002, 0.05, log=True),
        n_estimators      = trial.suggest_int  ("n_estimators",      3000, 12000, step=1000),
        num_leaves        = trial.suggest_int  ("num_leaves",        20, 127),
        subsample         = trial.suggest_float("subsample",         0.65, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.50, 0.90),
        min_child_samples = trial.suggest_int  ("min_child_samples", 10, 80),
        reg_alpha         = trial.suggest_float("reg_alpha",         0.01, 5.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",        0.5, 10.0, log=True),
    )
    m = LGBMRegressor(**params, random_state=42, verbose=-1)
    cb = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cb)
    return -r2_score(y_val, m.predict(X_val))

t0 = time.time()
lgbm_study = optuna.create_study(direction="minimize",
                                   sampler=TPESampler(seed=42),
                                   pruner=HyperbandPruner())
lgbm_study.optimize(lgbm_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm = lgbm_study.best_params

lgbm_model = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
cb_fit = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cb_fit)
y_pred_lgbm = lgbm_model.predict(X_test)
r2_lgbm  = r2_score(y_test, y_pred_lgbm)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
print(f"  R²={r2_lgbm:.4f} | MAE={mae_lgbm:.4f}  {'' if r2_lgbm > RECORD_R2_GLOBAL else ''} | {time.time()-t0:.0f}s")
global_results_v10["LightGBM"] = {"r2": r2_lgbm, "mae": mae_lgbm}

# ── CatBoost ───────────────────────────────────────────────────────────────
print(f"\n    CatBoost v10 ({N_OPTUNA_TRIALS} Optuna trials)...")

def cat_objective(trial):
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
cat_study = optuna.create_study(direction="minimize",
                                  sampler=TPESampler(seed=42),
                                  pruner=HyperbandPruner())
cat_study.optimize(cat_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_cat = cat_study.best_params

cat_model = CatBoostRegressor(**best_cat, early_stopping_rounds=200, random_state=42, verbose=0)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
y_pred_cat = cat_model.predict(X_test)
r2_cat  = r2_score(y_test, y_pred_cat)
mae_cat = mean_absolute_error(y_test, y_pred_cat)
print(f"  R²={r2_cat:.4f} | MAE={mae_cat:.4f}  {'' if r2_cat > RECORD_R2_GLOBAL else ''} | {time.time()-t0:.0f}s")
global_results_v10["CatBoost"] = {"r2": r2_cat, "mae": mae_cat}

# ── Stacking L2 (OOF) ──────────────────────────────────────────────────────
print("\n  Stacking L2 OOF v10...")
t0 = time.time()

kf = KFold(n_splits=5, shuffle=False)
# For cross_val_predict, early_stopping_rounds is removed (no eval_set available)
xgb_s  = XGBRegressor(
    **{k: v for k, v in best_xgb.items() if k != "early_stopping_rounds"},
    tree_method="hist", verbosity=0, random_state=42)
lgbm_s = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
cat_s  = CatBoostRegressor(
    **{k: v for k, v in best_cat.items() if k != "early_stopping_rounds"},
    random_state=42, verbose=0)

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

meta_train = build_meta_features(oof_xgb, oof_lgbm, oof_cat)
meta_test  = build_meta_features(p_xgb_test, p_lgbm_test, p_cat_test)
meta_val   = build_meta_features(xgb_s.predict(X_val), lgbm_s.predict(X_val), cat_s.predict(X_val))

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
    m.fit(meta_train, y_train, eval_set=[(meta_val, y_val)], verbose=False)
    return -r2_score(y_val, m.predict(meta_val))

meta_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=77))
meta_study.optimize(meta_l2_objective, n_trials=20, show_progress_bar=False)
best_meta  = meta_study.best_params

xgb_l2 = XGBRegressor(**best_meta, tree_method="hist", early_stopping_rounds=100,
                        eval_metric="mae", verbosity=0, random_state=77)
xgb_l2.fit(meta_train, y_train, eval_set=[(meta_val, y_val)], verbose=False)

cat_l2 = CatBoostRegressor(depth=4, learning_rate=0.01, iterations=3000,
                             l2_leaf_reg=5, subsample=0.8,
                             early_stopping_rounds=100, random_state=77, verbose=0)
cat_l2.fit(meta_train, y_train, eval_set=(meta_val, y_val))

meta_l2_val  = np.column_stack([xgb_l2.predict(meta_val),  cat_l2.predict(meta_val)])
meta_l2_test = np.column_stack([xgb_l2.predict(meta_test), cat_l2.predict(meta_test)])

ridge_l2 = Ridge(alpha=1.0)
ridge_l2.fit(meta_l2_val, y_val)
y_pred_l2  = ridge_l2.predict(meta_l2_test)
r2_l2      = r2_score(y_test, y_pred_l2)
mae_l2     = mean_absolute_error(y_test, y_pred_l2)
delta_v10   = r2_l2 - RECORD_R2_GLOBAL
print(f"  R²={r2_l2:.4f} | MAE={mae_l2:.4f}  {'' if r2_l2 > RECORD_R2_GLOBAL else ''}  (Delta vs v9: {delta_v10:+.4f}) | {time.time()-t0:.0f}s")
global_results_v10["Stacking-L2"] = {"r2": r2_l2, "mae": mae_l2}

# ── R² segmenté sur le set test ───────────────────────────────────────────────
print("\n   Segmented R² on test set (Stacking-L2) :")
for seg_name, seg_mask_col in [("With speech (has_speech=1)", "has_speech"),
                                ("Without speech (has_speech=0)", "has_speech")]:
    val_s = 1 if "Avec" in seg_name else 0
    idx_s = df.index[test_mask & (df["has_speech"] == val_s)]
    if len(idx_s) > 10:
        local_idx = [list(df.index[test_mask]).index(i) for i in idx_s if i in df.index[test_mask]]
        y_s  = y_test.values[local_idx]
        yp_s = y_pred_l2[local_idx]
        r2_s = r2_score(y_s, yp_s)
        ab_r2_results[f" {seg_name} (Stacking-L2)"] = r2_s
        goal = " OBJECTIF ≥0.45 !" if r2_s >= 0.45 else (" ≥0.40" if r2_s >= 0.40 else "")
        print(f"     {seg_name:<35}: R²={r2_s:.4f}  {goal}")

# ─────────────────────────────────────────────────────────────────────────────
# 11b. HYBRID SPEECH / NON-SPEECH STACKING (new in v10)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   – GLOBAL model: already trained above (all videos)
#   – SPEECH model: trained only on spoken segment (has_speech=1)
#                      with the same features + BERT SVD
#   – Meta-learner: Ridge that combines global + speech based on has_speech
#     → for silent videos, the meta-learner relies on global
#     → pour les videos with speech, il peut pondérer les deux prédictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 6] HYBRID SPEECH / NON-SPEECH STACKING (v10)")
print("=" * 72)

mask_speech_train = train_mask & (df["has_speech"] == 1)
mask_speech_val   = val_mask   & (df["has_speech"] == 1)
mask_speech_test  = test_mask  & (df["has_speech"] == 1)

n_sp_train = mask_speech_train.sum()
n_sp_test  = mask_speech_test.sum()
print(f"\n  Spoken segment — Train={n_sp_train:,} | Test={n_sp_test:,}")

hybrid_r2 = None
if n_sp_train >= 100 and n_sp_test >= 20:
    Xsp_train = df.loc[mask_speech_train, SELECTED_FEATURES].astype(float)
    ysp_train = df.loc[mask_speech_train, TARGET]
    Xsp_val   = df.loc[mask_speech_val,   SELECTED_FEATURES].astype(float)
    ysp_val   = df.loc[mask_speech_val,   TARGET]
    Xsp_test  = df.loc[mask_speech_test,  SELECTED_FEATURES].astype(float)
    ysp_test  = df.loc[mask_speech_test,  TARGET]

    # ── Model specialized on spoken segment (Optuna 20 trials) ────────────
    print("    CatBoost speech-only (20 trials Optuna)...")
    t0 = time.time()

    def cat_speech_objective(trial):
        params = dict(
            depth            = trial.suggest_int  ("depth",            4, 8),
            learning_rate    = trial.suggest_float("learning_rate",    0.002, 0.05, log=True),
            iterations       = trial.suggest_int  ("iterations",       2000, 8000, step=1000),
            l2_leaf_reg      = trial.suggest_float("l2_leaf_reg",      1.0, 15.0, log=True),
            subsample        = trial.suggest_float("subsample",        0.65, 0.95),
            colsample_bylevel= trial.suggest_float("colsample_bylevel",0.50, 0.90),
            min_data_in_leaf = trial.suggest_int  ("min_data_in_leaf", 3, 20),
        )
        m = CatBoostRegressor(**params, early_stopping_rounds=150,
                              random_state=42, verbose=0)
        m.fit(Xsp_train, ysp_train, eval_set=(Xsp_val, ysp_val))
        return -r2_score(ysp_val, m.predict(Xsp_val))

    sp_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    sp_study.optimize(cat_speech_objective, n_trials=20, show_progress_bar=False)

    cat_speech = CatBoostRegressor(**sp_study.best_params,
                                    early_stopping_rounds=150,
                                    random_state=42, verbose=0)
    cat_speech.fit(Xsp_train, ysp_train, eval_set=(Xsp_val, ysp_val))
    r2_speech_alone = r2_score(ysp_test, cat_speech.predict(Xsp_test))
    print(f"  R² speech-only model : {r2_speech_alone:.4f} "
          f"({'' if r2_speech_alone > RECORD_R2_GLOBAL else ''}) | {time.time()-t0:.0f}s")

    # ── Hybrid meta-learner on test set ─────────────────────────────────
    # Global model predictions on spoken segment
    test_speech_local_idx = [list(df.index[test_mask]).index(i)
                              for i in df.index[mask_speech_test]]
    pred_global_on_speech = y_pred_l2[test_speech_local_idx]
    pred_speech_model     = cat_speech.predict(Xsp_test)

    # Ridge stacking: [pred_global, pred_speech_model] -> target
    # Trained on speech validation
    val_speech_local_idx = [list(df.index[val_mask]).index(i)
                             for i in df.index[mask_speech_val]]
    pred_global_val_sp   = ridge_l2.predict(
        build_meta_features(
            xgb_l2.predict(meta_val[[i for i in val_speech_local_idx]]),
            cat_l2.predict(meta_val[[i for i in val_speech_local_idx]])
        ) if False else  # simplification: using already-computed val predictions
        np.column_stack([
            xgb_l2.predict(meta_val[val_speech_local_idx]),
            cat_l2.predict(meta_val[val_speech_local_idx])
        ])
    )
    pred_speech_val = cat_speech.predict(Xsp_val)

    meta_hybrid_val  = np.column_stack([pred_global_val_sp,  pred_speech_val])
    meta_hybrid_test = np.column_stack([pred_global_on_speech, pred_speech_model])

    ridge_hybrid = Ridge(alpha=1.0)
    ridge_hybrid.fit(meta_hybrid_val, ysp_val)
    pred_hybrid_speech = ridge_hybrid.predict(meta_hybrid_test)

    r2_hybrid_speech = r2_score(ysp_test, pred_hybrid_speech)
    mae_hybrid_speech = mean_absolute_error(ysp_test, pred_hybrid_speech)
    print(f"\n    Hybrid (global + speech) on spoken segment:")
    print(f"     R²={r2_hybrid_speech:.4f} | MAE={mae_hybrid_speech:.4f} "
          f"({'' if r2_hybrid_speech > RECORD_R2_GLOBAL else ''})")
    print(f"     Ridge weights: global={ridge_hybrid.coef_[0]:.3f} | "
          f"speech-only={ridge_hybrid.coef_[1]:.3f}")

    # ── Global hybrid score (speech hybrid + global for non-speech) ─────────
    # Reconstructing the complete prediction vector on the test set
    y_pred_hybrid_full = y_pred_l2.copy()
    for rank_pos, global_idx in enumerate(test_speech_local_idx):
        y_pred_hybrid_full[global_idx] = pred_hybrid_speech[rank_pos]

    hybrid_r2   = r2_score(y_test, y_pred_hybrid_full)
    hybrid_mae  = mean_absolute_error(y_test, y_pred_hybrid_full)
    delta_hybrid = hybrid_r2 - RECORD_R2_GLOBAL
    print(f"\n   GLOBAL HYBRID SCORE (speech+global) :")
    print(f"     R²={hybrid_r2:.4f} | MAE={hybrid_mae:.4f} "
          f"(Delta vs v9: {delta_hybrid:+.4f}) "
          f"{'' if hybrid_r2 > RECORD_R2_GLOBAL else ''}")
    global_results_v10["Hybride"] = {"r2": hybrid_r2, "mae": hybrid_mae}
else:
    print("  Warning: Spoken segment insufficient for hybrid stacking")


print("\n" + "=" * 72)
print("  FINAL SUMMARY — v10 vs v9")
print("=" * 72)

print(f"\n  {'Model':<30} {'R²':>8} {'Delta v8':>9} {'MAE':>8}  Status")
print(f"  {'─'*65}")
for name, res in global_results_v10.items():
    dr2  = res["r2"] - RECORD_R2_GLOBAL
    icon = "" if res["r2"] == max(r["r2"] for r in global_results_v10.values()) else "  "
    goal = ""
    if res["r2"] >= 0.45: goal = " ≥0.45 !"
    elif res["r2"] >= 0.40: goal = " ≥0.40"
    elif res["r2"] > RECORD_R2_GLOBAL: goal = " >"
    else: goal = " ≤"
    print(f"  {icon} {name:<28} {res['r2']:>8.4f} {dr2:>+9.4f} {res['mae']:>8.4f}  {goal}")

print(f"  {'─'*65}")
print(f"     {'Record v8 (global)':<28} {RECORD_R2_GLOBAL:>8.4f} {'':>9} {RECORD_MAE:>8.4f}")
print(f"     {'Record v8 (Comedy)':<28} {RECORD_R2_COMEDY:>8.4f}")

print(f"\n  A/B Test — Speech Features impact:")
for seg, r2 in ab_r2_results.items():
    goal = " " if r2 >= 0.45 else (" " if r2 >= 0.40 else "")
    print(f"     {seg:<45}: R²={r2:.4f}{goal}")

# ─────────────────────────────────────────────────────────────────────────────
# 13. VISUALIZATION v9
# ─────────────────────────────────────────────────────────────────────────────
print("\n   Generating v10 dashboard...")

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#0d0d0d")
fig.suptitle(
    "Stacking v10 — Content-Deep Analysis | Speech Features x TikTok Virality",
    fontsize=15, color="white", fontweight="bold", y=0.98
)

P = dict(ax_bg="#1a1a1a", grid="#2d2d2d", text="#f0f0f0",
         gold="#ffbe0b", record="#ff4d6d", target="#00b4d8",
         speech="#2ecc71", nospeech="#e74c3c", lsa="#9b59b6")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1 : Distribution has_speech × target_log ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(P["ax_bg"])
for sp in ax1.spines.values(): sp.set_color(P["grid"])

for has_s, color, label in [(0, P["nospeech"], f'Sans parole (n={mask_no.sum():,})'),
                              (1, P["speech"],   f'Avec parole (n={mask_yes.sum():,})')]:
    subset = df[df["has_speech"] == has_s][TARGET].dropna()
    ax1.hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
    ax1.axvline(subset.mean(), color=color, linestyle="--", linewidth=2,
                label=f"μ={subset.mean():.2f}")

ax1.set_xlabel("target_log (vues)", color=P["text"])
ax1.set_ylabel("Density", color=P["text"])
ax1.set_title("Distribution: Speech vs Non-verbal", color=P["text"], fontweight="bold")
ax1.tick_params(colors=P["text"])
ax1.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=7)
ax1.yaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 2 : A/B Test R² ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(P["ax_bg"])
for sp in ax2.spines.values(): sp.set_color(P["grid"])

ab_labels = list(ab_r2_results.keys())
ab_vals   = list(ab_r2_results.values())
ab_colors = [P["nospeech"] if "Sans" in l else
             (P["gold"]    if "Stacking" in l else
             (P["speech"]  if "+speech" in l else "#f39c12"))
             for l in ab_labels]

bars = ax2.bar(range(len(ab_labels)), ab_vals, color=ab_colors, alpha=0.85, width=0.6)
for bar, val in zip(bars, ab_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.3f}", ha="center", va="bottom",
             color=P["text"], fontweight="bold", fontsize=9)

ax2.axhline(RECORD_R2_GLOBAL, color=P["record"], linewidth=1.5, linestyle="--", label=f"Record v8 ({RECORD_R2_GLOBAL})")
ax2.axhline(0.40,             color=P["target"], linewidth=1.0, linestyle=":",  label="Target 0.40")
ax2.axhline(0.45,             color=P["gold"],   linewidth=1.5, linestyle="-.", label="Target 0.45")
ax2.set_xticks(range(len(ab_labels)))
ax2.set_xticklabels([l.replace("","").replace("","").replace("","").strip()[:18]
                     for l in ab_labels], rotation=35, ha="right", color=P["text"], fontsize=7)
ax2.set_ylabel("R²", color=P["text"])
ax2.set_title("A/B Test: Speech Impact", color=P["text"], fontweight="bold")
ax2.tick_params(colors=P["text"])
ax2.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=7)
ax2.yaxis.grid(True, color=P["grid"], alpha=0.5, zorder=0)

# ── Panel 3 : R² Stacking v9 vs v8 ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(P["ax_bg"])
for sp in ax3.spines.values(): sp.set_color(P["grid"])

models_compare = {
    "XGBoost v10":    global_results_v10["XGBoost"]["r2"],
    "LightGBM v10":   global_results_v10["LightGBM"]["r2"],
    "CatBoost v10":   global_results_v10["CatBoost"]["r2"],
    "Stacking-L2 v10":global_results_v10["Stacking-L2"]["r2"],
    "Record v8":     RECORD_R2_GLOBAL,
    "Comedy v8":     RECORD_R2_COMEDY,
}
mc_colors = ["#3a86ff","#06d6a0","#8338ec","#ffbe0b","#ff4d6d","#ff9f1c"]
bars3 = ax3.bar(range(len(models_compare)), list(models_compare.values()),
                color=mc_colors, alpha=0.88)
for i, (bar, val) in enumerate(zip(bars3, models_compare.values())):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom", color=P["text"], fontsize=8, fontweight="bold")
ax3.axhline(0.45, color=P["gold"], linewidth=1.5, linestyle="-.", label="Target 0.45")
ax3.set_xticks(range(len(models_compare)))
ax3.set_xticklabels(list(models_compare.keys()), rotation=35, ha="right", color=P["text"], fontsize=7)
ax3.set_ylabel("R²", color=P["text"])
ax3.set_title("R² v10 vs v9", color=P["text"], fontweight="bold")
ax3.tick_params(colors=P["text"])
ax3.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=7)
ax3.yaxis.grid(True, color=P["grid"], alpha=0.5, zorder=0)

# ── Panel 4 : Speech Rate × Viralité ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(P["ax_bg"])
for sp in ax4.spines.values(): sp.set_color(P["grid"])

df_sp = df[df["has_speech"] == 1].copy()
df_sp["sr_bin"] = pd.cut(df_sp["speech_rate"].clip(0, 8), bins=8)
sr_agg = df_sp.groupby("sr_bin", observed=True)[TARGET].agg(["mean", "count"]).dropna()

x_sr = range(len(sr_agg))
ax4.bar(x_sr, sr_agg["mean"].values, color=P["lsa"], alpha=0.8)
ax4.axhline(df_sp[TARGET].mean(), color=P["gold"], linewidth=1.5, linestyle="--", label="μ (spoken)")
ax4.set_xticks(x_sr)
ax4.set_xticklabels([str(b)[:7] for b in sr_agg.index], rotation=45, ha="right",
                     color=P["text"], fontsize=7)
ax4.set_xlabel("Speech Rate (mots/s)", color=P["text"])
ax4.set_ylabel("target_log moyen", color=P["text"])
ax4.set_title("Speech Rate vs Virality", color=P["text"], fontweight="bold")
ax4.tick_params(colors=P["text"])
ax4.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=8)
ax4.yaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 5 : Hook Score × Viralité ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(P["ax_bg"])
for sp in ax5.spines.values(): sp.set_color(P["grid"])

hook_grp = df_sp.groupby("hook_score", observed=True)[TARGET].agg(["mean", "count"])
hook_grp = hook_grp[hook_grp["count"] >= 5]

scatter = ax5.scatter(hook_grp.index, hook_grp["mean"],
                      s=hook_grp["count"].clip(10, 200),
                      c=hook_grp["mean"], cmap="RdYlGn", alpha=0.9, zorder=3,
                      vmin=df_sp[TARGET].quantile(0.1), vmax=df_sp[TARGET].quantile(0.9))
if len(hook_grp) > 2:
    z = np.polyfit(hook_grp.index.astype(float), hook_grp["mean"].values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(hook_grp.index.min(), hook_grp.index.max(), 50)
    ax5.plot(x_line, p(x_line), color=P["gold"], linewidth=2, linestyle="--", zorder=4)
plt.colorbar(scatter, ax=ax5, label="target_log").ax.yaxis.label.set_color(P["text"])
ax5.set_xlabel("Hook Score", color=P["text"])
ax5.set_ylabel("target_log moyen", color=P["text"])
ax5.set_title("Hook Score vs Virality (size = n)", color=P["text"], fontweight="bold")
ax5.tick_params(colors=P["text"])
ax5.yaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 6 : Feature Importance (top 20, coloré par type) ──────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(P["ax_bg"])
for sp in ax6.spines.values(): sp.set_color(P["grid"])

fi_top20  = fi_series.head(20)
fi_colors = []
for feat in fi_top20.index:
    if feat in SPEECH_FEATURES:    fi_colors.append(P["speech"])
    elif feat in BERT_COLS:         fi_colors.append(P["lsa"])
    elif feat in MOMENTUM_COLS:    fi_colors.append("#3a86ff")
    elif feat in CROSS_COLS_V8:    fi_colors.append("#f39c12")
    elif feat in CROSS_SPEECH_COLS:fi_colors.append("#27ae60")
    else:                          fi_colors.append("#888888")

ax6.barh(range(len(fi_top20)), fi_top20.values[::-1], color=fi_colors[::-1], alpha=0.85)
ax6.set_yticks(range(len(fi_top20)))
ax6.set_yticklabels(fi_top20.index[::-1], color=P["text"], fontsize=8)
ax6.set_xlabel("Importance (CatBoost)", color=P["text"])
ax6.set_title("Top 20 Features\n Green=Speech Purple=BERT Blue=Momentum Orange=Cross", color=P["text"], fontweight="bold", fontsize=9)
ax6.tick_params(colors=P["text"])
ax6.xaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 7 : Hook type × target_log (boxplot) ──────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
ax7.set_facecolor(P["ax_bg"])
for sp in ax7.spines.values(): sp.set_color(P["grid"])

hook_types = {
    "Question": df_sp[df_sp["is_question"] == 1][TARGET],
    "Liste":    df_sp[df_sp["is_list"] == 1][TARGET],
    "Urgence":  df_sp[df_sp["is_urgency"] == 1][TARGET],
    "POI":      df_sp[df_sp["is_poi"] == 1][TARGET],
    "Personnel":df_sp[df_sp["is_personal"] == 1][TARGET],
    "Neutre":   df_sp[(df_sp["hook_score"] == 0)][TARGET],
}
hook_types = {k: v for k, v in hook_types.items() if len(v) > 5}
bp_data    = [v.values for v in hook_types.values()]
bp_colors  = ["#3a86ff","#06d6a0","#ffbe0b","#8338ec","#ef476f","#888888"][:len(hook_types)]

bp = ax7.boxplot(bp_data, patch_artist=True, notch=True,
                 medianprops=dict(color="white", linewidth=2),
                 whiskerprops=dict(color=P["text"]), capprops=dict(color=P["text"]),
                 flierprops=dict(marker=".", color=P["grid"], alpha=0.3))
for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax7.axhline(df[TARGET].mean(), color=P["gold"], linewidth=1.5, linestyle="--", label="μ global")
ax7.set_xticks(range(1, len(hook_types) + 1))
ax7.set_xticklabels(
    [f"{k}\n(n={len(v):,})" for k, v in hook_types.items()],
    color=P["text"], fontsize=9
)
ax7.set_ylabel("target_log", color=P["text"])
ax7.set_title("Virality by Hook Type (Hook Analysis)", color=P["text"], fontweight="bold")
ax7.tick_params(colors=P["text"])
ax7.yaxis.grid(True, color=P["grid"], alpha=0.5)
ax7.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"])

# ── Panel 8 : Progression R² v5 → v10 ───────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(P["ax_bg"])
for sp in ax8.spines.values(): sp.set_color(P["grid"])

best_v10   = max(r["r2"] for r in global_results_v10.values())
versions   = ["v5\n(base)", "v6\n(momentum)", "v7\n(NLP+CV)", "v8\n(niches)", "v9\n(speech)", "v10\n(bert)"]
r2_history = [0.18, 0.24, 0.3332, 0.3310, 0.3176, best_v10]
bar_colors = ["#666","#888","#aaa","#ff4d6d","#888888",
              P["gold"] if best_v10 >= 0.45 else P["speech"]]

bars8 = ax8.bar(range(len(versions)), r2_history, color=bar_colors, alpha=0.85)
for bar, val in zip(bars8, r2_history):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom", color=P["text"], fontsize=9, fontweight="bold")
ax8.axhline(0.45, color=P["gold"], linewidth=1.5, linestyle="-.", label="Target 0.45")
ax8.axhline(0.40, color=P["target"], linewidth=1.0, linestyle=":", label="Target 0.40")
ax8.set_xticks(range(len(versions)))
ax8.set_xticklabels(versions, color=P["text"], fontsize=9)
ax8.set_ylabel("Global R²", color=P["text"])
ax8.set_title("R² progression across versions", color=P["text"], fontweight="bold")
ax8.tick_params(colors=P["text"])
ax8.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=7)
ax8.yaxis.grid(True, color=P["grid"], alpha=0.5)

plt.savefig("stacking_v10_bert.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("   Dashboard saved -> stacking_v10_bert.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 14. CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {"version": "v10", "model": k, "r2": v["r2"], "mae": v["mae"],
     "delta_v9": v["r2"] - RECORD_R2_GLOBAL,
     "beats_record": v["r2"] > RECORD_R2_GLOBAL,
     "reaches_040": v["r2"] >= 0.40,
     "reaches_045": v["r2"] >= 0.45}
    for k, v in global_results_v10.items()
] + [
    {"version": "v9_record", "model": "global", "r2": RECORD_R2_GLOBAL, "mae": RECORD_MAE,
     "delta_v9": 0, "beats_record": False, "reaches_040": False, "reaches_045": False},
    {"version": "v8_record", "model": "Comedy",  "r2": RECORD_R2_COMEDY, "mae": np.nan,
     "delta_v9": RECORD_R2_COMEDY - RECORD_R2_GLOBAL, "beats_record": True,
     "reaches_040": True, "reaches_045": False}
])
results_df.to_csv("results_v10.csv", index=False)
print("   Results exported -> results_v10.csv")

fi_series.to_frame("importance").reset_index().rename(columns={"index": "feature"}).to_csv(
    "feature_importance_v10.csv", index=False
)
print("   Feature importance -> feature_importance_v10.csv")

export_cols = ["creator_id_int", "video_rank", "niche", "has_speech", "speech_rate",
               "hook_score", "is_question", "is_list", "is_urgency", "hook_text",
               "sentiment_ratio", TARGET]
df[export_cols].to_csv("df_v10_enriched.csv", index=False)
print("   Enriched dataset -> df_v10_enriched.csv")

print("\n" + "=" * 72)
print("   STACKING v10 COMPLETE")
print("=" * 72)

best_model  = max(global_results_v10, key=lambda k: global_results_v10[k]["r2"])
best_r2     = global_results_v10[best_model]["r2"]
print(f"\n   Best model: {best_model}  R²={best_r2:.4f}")
print(f"   Delta vs v9     : {best_r2 - RECORD_R2_GLOBAL:+.4f}")
print(f"  💬 A/B speech gain : {delta_ab:+.4f} (on spoken segment, base->speech+BERT)")
if best_r2 >= 0.45:
    print(f"\n   TARGET R² >= 0.45 REACHED!")
elif best_r2 >= 0.40:
    print(f"\n   Target R² >= 0.40 crossed!")
else:
    print(f"\n   Plateau: {best_r2:.4f}")
    print(f"     Next direction: Comedy niche model with BERT")

