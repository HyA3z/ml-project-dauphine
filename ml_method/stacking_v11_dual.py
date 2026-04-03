"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TIKTOK VIRALITY — STACKING v11 : DUAL PIPELINE                ║
║           Silent Pipeline (tabular) + Spoken Pipeline (BERT) — Separated       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v10 diagnosis:                                                          ║
║  • BERT strongly improves the spoken segment in isolation (R²=0.38)        ║
║  • But in the global model, BERT is 0 for 65% of videos (silent) ║
║  • This 65%/35% discontinuity disrupts trees -> global regression   ║
║                                                                            ║
║  v11 Architecture — SEPARATE Dual Pipeline from the start:                ║
║  • PIPELINE A (65.4%) — SILENT Videos                                    ║
║      Features: tabular only (momentum, hist, cross, niche)      ║
║      Stacking: XGBoost + LightGBM + CatBoost + Ridge L2 OOF              ║
║  • PIPELINE B (34.6%) — SPOKEN Videos                                    ║
║      Features: tabular + speech + BERT SVD 20D                        ║
║      Stacking: XGBoost + LightGBM + CatBoost + Ridge L2 OOF              ║
║  • FINAL ASSEMBLY: concatenation of predictions by has_speech        ║
║      No additional meta-learner — each pipeline is self-contained    ║
║                                                                            ║
║  All v9/v10 fixes preserved:                               ║
║        - creator_id extracted from webVideoUrl, tri (creator_id, video_rank)        ║
║        - Causal expanding window target encoding                            ║
║        - BERT cache bert_embeddings_cache.npy                               ║
║        - early_stopping_rounds removed from cross_val_predict                  ║
║  Record to beat: R² = 0.3310 (global) | R² = 0.4036 (Comedy)           ║
║  Goal v11: R² >= 0.35 global | R² >= 0.45 spoken segment               ║
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
TOP_K_FEATURES_A  = 40            # Pipeline A — silent (features tabulaires)
TOP_K_FEATURES_B  = 45            # Pipeline B — spoken (tabulaire + BERT)
N_BERT_SVD        = 20
BERT_MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
BERT_CACHE_PATH   = "bert_embeddings_cache.npy"
BERT_BATCH_SIZE   = 64
MIN_NICHE_SAMPLES = 80
SMOOTH_K          = 5

print("=" * 72)
print("  STACKING v11 — DUAL PIPELINE (Silent | Spoken)")
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
# 6. FEATURE SETS v11 — SEPARATED BY SEGMENT
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

# Pipeline A — silent: tabular features only (no BERT, no speech)
ALL_FEATURES_A = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS_V8 +
    NLP_COLS + NICHE_COLS + ["has_speech"]   # has_speech=0 for all -> null but stable signal
)

# Pipeline B — spoken: tabular + speech + BERT
ALL_FEATURES_B = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS_V8 +
    NLP_COLS + NICHE_COLS + SPEECH_FEATURES + BERT_COLS
)

print(f"\n   Pipeline A (silent) : {len(ALL_FEATURES_A)} features")
print(f"   Pipeline B (spoken) : {len(ALL_FEATURES_B)} features "
      f"(+{len(SPEECH_FEATURES) + len(BERT_COLS)} speech+BERT)")

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
# 8. SLIDING RANK TARGET ENCODING (anti-leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("  [LEVER A] Sliding rank target encoding (causal, anti-leakage)")
print("─" * 72)

global_mean = df.loc[train_mask, TARGET].mean()

df = df.sort_values(["creator_id_int", "video_rank"]).reset_index(drop=True)

def expanding_te(group, smooth_k, global_mean):
    target_vals = group[TARGET].values
    te_vals     = np.full(len(target_vals), global_mean)
    cumsum, cumcount = 0.0, 0
    for i in range(len(target_vals)):
        te_vals[i] = (cumsum + smooth_k * global_mean) / (cumcount + smooth_k)
        cumsum   += target_vals[i]
        cumcount += 1
    return pd.Series(te_vals, index=group.index)

df["creator_target_mean"] = (
    df.groupby("creator_id_int", group_keys=False)
      .apply(lambda g: expanding_te(g, SMOOTH_K, global_mean))
)

train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

te_corr = np.corrcoef(df.loc[train_mask, "creator_target_mean"],
                      df.loc[train_mask, TARGET])[0, 1]
assert te_corr < 0.999, f"❌ Leakage detected (corr={te_corr:.4f})"
print(f"   creator_target_mean (expanding OOF, k={SMOOTH_K}) | corr={te_corr:.4f}")

ALL_FEATURES_A_FINAL = ALL_FEATURES_A + ["creator_target_mean"]
ALL_FEATURES_B_FINAL = ALL_FEATURES_B + ["creator_target_mean"]

# Masques segment
mask_silent = df["has_speech"] == 0
mask_speech = df["has_speech"] == 1

# ─────────────────────────────────────────────────────────────────────────────
# 9. QUICK A/B TEST (preserved for comparison)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [A/B TEST] IMPACT OF SPEECH ON PREDICTION")
print("=" * 72)

ab_r2_results  = {}
ab_mae_results = {}
delta_ab       = 0.0

BASE_FEATURES_AB = [f for f in ALL_FEATURES_A_FINAL
                    if f not in SPEECH_FEATURES and f not in BERT_COLS]

def ab_quick_model():
    return LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         num_leaves=40, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbose=-1)

for grp_name, seg_mask, feats, label in [
    ("A", mask_silent, BASE_FEATURES_AB, " Without speech (base)"),
    ("B", mask_speech, BASE_FEATURES_AB, " With speech (base)"),
]:
    df_seg   = df[seg_mask]
    X_tr = df_seg.loc[df_seg.index.isin(df.index[train_mask]), feats].astype(float)
    y_tr = df_seg.loc[df_seg.index.isin(df.index[train_mask]), TARGET]
    X_vl = df_seg.loc[df_seg.index.isin(df.index[val_mask]),   feats].astype(float)
    y_vl = df_seg.loc[df_seg.index.isin(df.index[val_mask]),   TARGET]
    X_te = df_seg.loc[df_seg.index.isin(df.index[test_mask]),  feats].astype(float)
    y_te = df_seg.loc[df_seg.index.isin(df.index[test_mask]),  TARGET]
    if len(X_tr) < 50 or len(X_te) < 10:
        continue
    m = ab_quick_model()
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    r2  = r2_score(y_te, m.predict(X_te))
    mae = mean_absolute_error(y_te, m.predict(X_te))
    ab_r2_results[label]  = r2
    ab_mae_results[label] = mae
    seg_label = "silent" if grp_name == "A" else "spoken"
    print(f"\n  {label}  (train={len(X_tr):,} | test={len(X_te):,})")
    print(f"     R² = {r2:.4f} | MAE = {mae:.4f}")

# Groupe C : spoken + speech + BERT
FULL_FEATURES_C = [f for f in ALL_FEATURES_B_FINAL
                   if f in BASE_FEATURES_AB or f in SPEECH_FEATURES or f in BERT_COLS]
df_sp   = df[mask_speech]
X_tr_c  = df[mask_speech].loc[df[mask_speech].index.isin(df.index[train_mask]), FULL_FEATURES_C].astype(float)
y_tr_c  = df[mask_speech].loc[df[mask_speech].index.isin(df.index[train_mask]), TARGET]
X_vl_c  = df[mask_speech].loc[df[mask_speech].index.isin(df.index[val_mask]),   FULL_FEATURES_C].astype(float)
y_vl_c  = df[mask_speech].loc[df[mask_speech].index.isin(df.index[val_mask]),   TARGET]
X_te_c  = df[mask_speech].loc[df[mask_speech].index.isin(df.index[test_mask]),  FULL_FEATURES_C].astype(float)
y_te_c  = df[mask_speech].loc[df[mask_speech].index.isin(df.index[test_mask]),  TARGET]
if len(X_tr_c) > 30 and len(X_te_c) > 5:
    m_c = LGBMRegressor(n_estimators=800, max_depth=6, learning_rate=0.05,
                        num_leaves=50, subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbose=-1)
    m_c.fit(X_tr_c, y_tr_c, eval_set=[(X_vl_c, y_vl_c)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    r2_c  = r2_score(y_te_c, m_c.predict(X_te_c))
    mae_c = mean_absolute_error(y_te_c, m_c.predict(X_te_c))
    ab_r2_results[" Avec parole (+speech+BERT)"]  = r2_c
    ab_mae_results[" Avec parole (+speech+BERT)"] = mae_c
    r2_base_sp = ab_r2_results.get(" With speech (base)", 0)
    delta_ab   = r2_c - r2_base_sp
    print(f"\n   GROUP C — With speech, BASE + SPEECH + BERT "
          f"(train={len(X_tr_c):,} | test={len(X_te_c):,})")
    print(f"     R² = {r2_c:.4f} | MAE = {mae_c:.4f}")
    print(f"     BERT+Speech gain vs base : {delta_ab:+.4f}")
    print(f"     {' R²≥0.45 !' if r2_c >= 0.45 else f' {r2_c:.4f} (objectif: 0.45)'}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE SELECTION PER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("  [LEVER B] Feature Selection — Pipeline A (silent) & B (spoken)")
print("─" * 72)

def select_features(all_feats, top_k, X_tr, y_tr, X_vl, y_vl, label):
    selector = CatBoostRegressor(
        iterations=3000, learning_rate=0.05, depth=6,
        early_stopping_rounds=100, random_state=42, verbose=0
    )
    selector.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    fi = pd.Series(selector.get_feature_importance(),
                   index=all_feats).sort_values(ascending=False)
    selected = fi.head(top_k).index.tolist()
    print(f"\n  Top {top_k} features [{label}]:")
    for rk, (feat, imp) in enumerate(fi.head(top_k).items(), 1):
        tag = ""
        if feat == "creator_target_mean": tag = " ←  TE"
        elif feat in BERT_COLS:           tag = " ←  BERT"
        elif feat in SPEECH_FEATURES:     tag = " ←  SPEECH"
        elif feat in MOMENTUM_COLS:       tag = " ←  MOM"
        elif feat in CROSS_COLS_V8:       tag = " ←  CROSS"
        elif feat in NICHE_COLS:          tag = " ←   NICHE"
        elif feat in NLP_COLS:            tag = " ←  NLP"
        print(f"    #{rk:>2}  {feat:<35} {imp:>6.2f}%{tag}")
    return selected, fi

# ── Pipeline A — silent ────────────────────────────────────────────────────────
Xa_tr_full = df.loc[train_mask, ALL_FEATURES_A_FINAL].astype(float)
ya_tr      = df.loc[train_mask, TARGET]
Xa_vl_full = df.loc[val_mask,   ALL_FEATURES_A_FINAL].astype(float)
ya_vl      = df.loc[val_mask,   TARGET]
Xa_te_full = df.loc[test_mask,  ALL_FEATURES_A_FINAL].astype(float)
ya_te      = df.loc[test_mask,  TARGET]

SELECTED_A, fi_A = select_features(
    ALL_FEATURES_A_FINAL, TOP_K_FEATURES_A,
    Xa_tr_full, ya_tr, Xa_vl_full, ya_vl, "Pipeline A — silent"
)

# ── Pipeline B — spoken ───────────────────────────────────────────────────────
# Feature selection uniquement sur le segment spoken
df_speech   = df[mask_speech]
Xb_tr_full  = df_speech.loc[df_speech.index.isin(df.index[train_mask]), ALL_FEATURES_B_FINAL].astype(float)
yb_tr       = df_speech.loc[df_speech.index.isin(df.index[train_mask]), TARGET]
Xb_vl_full  = df_speech.loc[df_speech.index.isin(df.index[val_mask]),   ALL_FEATURES_B_FINAL].astype(float)
yb_vl       = df_speech.loc[df_speech.index.isin(df.index[val_mask]),   TARGET]
Xb_te_full  = df_speech.loc[df_speech.index.isin(df.index[test_mask]),  ALL_FEATURES_B_FINAL].astype(float)
yb_te       = df_speech.loc[df_speech.index.isin(df.index[test_mask]),  TARGET]

SELECTED_B, fi_B = select_features(
    ALL_FEATURES_B_FINAL, TOP_K_FEATURES_B,
    Xb_tr_full, yb_tr, Xb_vl_full, yb_vl, "Pipeline B — spoken"
)

# Matrices finales
Xa_tr = df.loc[train_mask, SELECTED_A].astype(float)
Xa_vl = df.loc[val_mask,   SELECTED_A].astype(float)
Xa_te = df.loc[test_mask,  SELECTED_A].astype(float)

Xb_tr = df_speech.loc[df_speech.index.isin(df.index[train_mask]), SELECTED_B].astype(float)
yb_tr = df_speech.loc[df_speech.index.isin(df.index[train_mask]), TARGET]
Xb_vl = df_speech.loc[df_speech.index.isin(df.index[val_mask]),   SELECTED_B].astype(float)
yb_vl = df_speech.loc[df_speech.index.isin(df.index[val_mask]),   TARGET]
Xb_te = df_speech.loc[df_speech.index.isin(df.index[test_mask]),  SELECTED_B].astype(float)
yb_te = df_speech.loc[df_speech.index.isin(df.index[test_mask]),  TARGET]

# ─────────────────────────────────────────────────────────────────────────────
# 11. DUAL PIPELINE — SEPARATE STACKING
# ─────────────────────────────────────────────────────────────────────────────
global_results_v11 = {}

def run_stacking_pipeline(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                           label, n_trials=N_OPTUNA_TRIALS):
    """Stacking XGB + LGBM + CatBoost + Ridge L2 OOF — generic."""
    results = {}
    t_start = time.time()

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print(f"\n    XGBoost [{label}] ({n_trials} trials)...")
    def xgb_obj(trial):
        p = dict(
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
        m = XGBRegressor(**p, tree_method="hist", early_stopping_rounds=200,
                         eval_metric="mae", verbosity=0, random_state=42)
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        return -r2_score(y_vl, m.predict(X_vl))

    xgb_study = optuna.create_study(direction="minimize",
                                     sampler=TPESampler(seed=42),
                                     pruner=HyperbandPruner())
    xgb_study.optimize(xgb_obj, n_trials=n_trials, show_progress_bar=False)
    best_xgb = xgb_study.best_params
    xgb_m = XGBRegressor(**best_xgb, tree_method="hist", verbosity=0, random_state=42)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    r2x = r2_score(y_te, xgb_m.predict(X_te))
    maex = mean_absolute_error(y_te, xgb_m.predict(X_te))
    results["XGBoost"] = {"r2": r2x, "mae": maex}
    print(f"  R²={r2x:.4f} | MAE={maex:.4f}  {'' if r2x > RECORD_R2_GLOBAL else ''}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print(f"\n    LightGBM [{label}] ({n_trials} trials)...")
    def lgbm_obj(trial):
        p = dict(
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
        m = LGBMRegressor(**p, random_state=42, verbose=-1)
        cb = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
        m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=cb)
        return -r2_score(y_vl, m.predict(X_vl))

    lgbm_study = optuna.create_study(direction="minimize",
                                      sampler=TPESampler(seed=42),
                                      pruner=HyperbandPruner())
    lgbm_study.optimize(lgbm_obj, n_trials=n_trials, show_progress_bar=False)
    best_lgbm = lgbm_study.best_params
    lgbm_m = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
    lgbm_m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
    r2l = r2_score(y_te, lgbm_m.predict(X_te))
    mael = mean_absolute_error(y_te, lgbm_m.predict(X_te))
    results["LightGBM"] = {"r2": r2l, "mae": mael}
    print(f"  R²={r2l:.4f} | MAE={mael:.4f}  {'' if r2l > RECORD_R2_GLOBAL else ''}")

    # ── CatBoost ─────────────────────────────────────────────────────────────
    print(f"\n    CatBoost [{label}] ({n_trials} trials)...")
    def cat_obj(trial):
        p = dict(
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
        m = CatBoostRegressor(**p, early_stopping_rounds=200,
                               random_state=42, verbose=0)
        m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
        return -r2_score(y_vl, m.predict(X_vl))

    cat_study = optuna.create_study(direction="minimize",
                                     sampler=TPESampler(seed=42),
                                     pruner=HyperbandPruner())
    cat_study.optimize(cat_obj, n_trials=n_trials, show_progress_bar=False)
    best_cat = cat_study.best_params
    cat_m = CatBoostRegressor(**best_cat, early_stopping_rounds=200,
                               random_state=42, verbose=0)
    cat_m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    r2c = r2_score(y_te, cat_m.predict(X_te))
    maec = mean_absolute_error(y_te, cat_m.predict(X_te))
    results["CatBoost"] = {"r2": r2c, "mae": maec}
    print(f"  R²={r2c:.4f} | MAE={maec:.4f}  {'' if r2c > RECORD_R2_GLOBAL else ''}")

    # ── Stacking L2 OOF ──────────────────────────────────────────────────────
    print(f"\n    Stacking L2 OOF [{label}]...")
    kf = KFold(n_splits=5, shuffle=False)
    xgb_s  = XGBRegressor(
        **{k: v for k, v in best_xgb.items() if k != "early_stopping_rounds"},
        tree_method="hist", verbosity=0, random_state=42)
    lgbm_s = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
    cat_s  = CatBoostRegressor(
        **{k: v for k, v in best_cat.items() if k != "early_stopping_rounds"},
        random_state=42, verbose=0)

    oof_xgb  = cross_val_predict(xgb_s,  X_tr, y_tr, cv=kf)
    oof_lgbm = cross_val_predict(lgbm_s, X_tr, y_tr, cv=kf)
    oof_cat  = cross_val_predict(cat_s,  X_tr, y_tr, cv=kf)

    xgb_s.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    lgbm_s.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
               callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
    cat_s.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))

    def build_meta(p1, p2, p3):
        return np.column_stack([p1, p2, p3,
                                (p1+p2+p3)/3,
                                np.abs(p1-p3), np.abs(p2-p3), np.abs(p1-p2)])

    meta_tr = build_meta(oof_xgb, oof_lgbm, oof_cat)
    meta_vl = build_meta(xgb_s.predict(X_vl), lgbm_s.predict(X_vl), cat_s.predict(X_vl))
    meta_te = build_meta(xgb_s.predict(X_te), lgbm_s.predict(X_te), cat_s.predict(X_te))

    def meta_obj(trial):
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
        m.fit(meta_tr, y_tr, eval_set=[(meta_vl, y_vl)], verbose=False)
        return -r2_score(y_vl, m.predict(meta_vl))

    meta_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=77))
    meta_study.optimize(meta_obj, n_trials=20, show_progress_bar=False)

    xgb_l2 = XGBRegressor(**meta_study.best_params, tree_method="hist",
                            early_stopping_rounds=100, eval_metric="mae",
                            verbosity=0, random_state=77)
    xgb_l2.fit(meta_tr, y_tr, eval_set=[(meta_vl, y_vl)], verbose=False)

    cat_l2 = CatBoostRegressor(depth=4, learning_rate=0.01, iterations=3000,
                                l2_leaf_reg=5, subsample=0.8,
                                early_stopping_rounds=100, random_state=77, verbose=0)
    cat_l2.fit(meta_tr, y_tr, eval_set=(meta_vl, y_vl))

    meta_l2_vl = np.column_stack([xgb_l2.predict(meta_vl), cat_l2.predict(meta_vl)])
    meta_l2_te = np.column_stack([xgb_l2.predict(meta_te), cat_l2.predict(meta_te)])

    ridge_l2 = Ridge(alpha=1.0)
    ridge_l2.fit(meta_l2_vl, y_vl)
    y_pred = ridge_l2.predict(meta_l2_te)

    r2_stack  = r2_score(y_te, y_pred)
    mae_stack = mean_absolute_error(y_te, y_pred)
    delta     = r2_stack - RECORD_R2_GLOBAL
    print(f"  R²={r2_stack:.4f} | MAE={mae_stack:.4f}  "
          f"{'' if r2_stack > RECORD_R2_GLOBAL else ''}  "
          f"(Δ vs v8 : {delta:+.4f}) | {time.time()-t_start:.0f}s")
    results["Stacking-L2"] = {"r2": r2_stack, "mae": mae_stack}
    return results, y_pred

# ── Lancement Pipeline A — silent ───────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 5A] PIPELINE A — SILENT VIDEOS (65.4%)")
print("=" * 72)

# For pipeline A we use the full dataset (complete train/val/test)
# Spoken videos are included in train to calibrate
# tabular features, but test score is global
y_train_a = df.loc[train_mask, TARGET]
y_val_a   = df.loc[val_mask,   TARGET]
y_test_a  = df.loc[test_mask,  TARGET]

results_A, y_pred_A_full = run_stacking_pipeline(
    Xa_tr, y_train_a, Xa_vl, y_val_a, Xa_te, y_test_a,
    label="silent-global", n_trials=N_OPTUNA_TRIALS
)

# Pipeline A score on the silent segment of the test only
mask_silent_test = test_mask & mask_silent
idx_silent_test  = [list(df.index[test_mask]).index(i)
                    for i in df.index[mask_silent_test]]
r2_A_silent = r2_score(y_test_a.values[idx_silent_test],
                        y_pred_A_full[idx_silent_test])
print(f"\n   Pipeline A — silent segment only: R²={r2_A_silent:.4f}")
global_results_v11["Pipeline-A (silent, global)"] = results_A["Stacking-L2"]

# ── Lancement Pipeline B — spoken ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 5B] PIPELINE B — SPOKEN VIDEOS (34.6%)")
print("=" * 72)

results_B, y_pred_B_speech = run_stacking_pipeline(
    Xb_tr, yb_tr, Xb_vl, yb_vl, Xb_te, yb_te,
    label="spoken-BERT", n_trials=N_OPTUNA_TRIALS
)
global_results_v11["Pipeline-B (spoken+BERT)"] = results_B["Stacking-L2"]

# ── Assemblage final — score global reconstruction ───────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 6] FINAL ASSEMBLY — Reconstructed global score")
print("=" * 72)

# y_pred_A_full: predictions on all test data (tabular only)
# y_pred_B_speech: predictions on the spoken test segment (tabular+BERT)
# In A we replace spoken segment predictions with those from B
y_pred_dual = y_pred_A_full.copy()
mask_speech_test = test_mask & mask_speech
idx_speech_test  = [list(df.index[test_mask]).index(i)
                    for i in df.index[mask_speech_test]]
for rank_pos, global_idx in enumerate(idx_speech_test):
    y_pred_dual[global_idx] = y_pred_B_speech[rank_pos]

r2_dual  = r2_score(y_test_a, y_pred_dual)
mae_dual = mean_absolute_error(y_test_a, y_pred_dual)
delta_dual = r2_dual - RECORD_R2_GLOBAL

print(f"\n   DUAL PIPELINE SCORE (A silent + B spoken):")
print(f"     R²={r2_dual:.4f} | MAE={mae_dual:.4f} "
      f"(Δ vs record v8 : {delta_dual:+.4f}) "
      f"{'' if r2_dual > RECORD_R2_GLOBAL else ''}")

# Scores segmentés
r2_dual_silent = r2_score(y_test_a.values[idx_silent_test],
                           y_pred_dual[idx_silent_test])
r2_dual_speech = r2_score(yb_te, y_pred_B_speech)
print(f"\n   Segmented R² (dual pipeline) :")
print(f"     Silent (has_speech=0): R²={r2_dual_silent:.4f}")
print(f"     Spoken (has_speech=1): R²={r2_dual_speech:.4f}")

global_results_v11["Dual-Pipeline"] = {"r2": r2_dual, "mae": mae_dual}
ab_r2_results[" Spoken (Pipeline-B, Stacking-L2)"]  = r2_dual_speech
ab_r2_results[" Silent (Pipeline-A, Stacking-L2)"]  = r2_dual_silent

# Variable alias pour compatibilité avec le dashboard existant
y_pred_l2          = y_pred_dual
global_results_v11 = global_results_v11   # alias pour le dashboard


print("\n" + "=" * 72)
print("  FINAL SUMMARY — v11 vs v8")
print("=" * 72)

print(f"\n  {'Model':<30} {'R²':>8} {'Delta v8':>9} {'MAE':>8}  Status")
print(f"  {'─'*65}")
for name, res in global_results_v11.items():
    dr2  = res["r2"] - RECORD_R2_GLOBAL
    icon = "" if res["r2"] == max(r["r2"] for r in global_results_v11.values()) else "  "
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
print("\n   Generating v11 dashboard...")

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#0d0d0d")
fig.suptitle(
    "Stacking v11 — Dual Pipeline | Silent (tabular) x Spoken (BERT) x TikTok Virality",
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

for has_s, color, label in [(0, P["nospeech"], f'Sans parole (n={mask_silent.sum():,})'),
                              (1, P["speech"],   f'Avec parole (n={mask_speech.sum():,})')]:
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

# ── Panel 3 : R² v11 vs records ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(P["ax_bg"])
for sp in ax3.spines.values(): sp.set_color(P["grid"])

r2_A  = global_results_v11.get("Pipeline-A (silent, global)",  {}).get("r2", 0)
r2_B  = global_results_v11.get("Pipeline-B (spoken+BERT)",    {}).get("r2", 0)
r2_D  = global_results_v11.get("Dual-Pipeline",              {}).get("r2", 0)

models_compare = {
    "Pipeline-A\n(silent)":   r2_A,
    "Pipeline-B\n(spoken)":  r2_B,
    "Dual\n(v11)":          r2_D,
    "Record v8\n(global)":  RECORD_R2_GLOBAL,
    "Comedy v8\n(local)":   RECORD_R2_COMEDY,
}
mc_colors = ["#3a86ff","#2ecc71","#ffbe0b","#ff4d6d","#ff9f1c"]
bars3 = ax3.bar(range(len(models_compare)), list(models_compare.values()),
                color=mc_colors, alpha=0.88)
for bar, val in zip(bars3, models_compare.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom", color=P["text"], fontsize=8, fontweight="bold")
ax3.axhline(0.45, color=P["gold"], linewidth=1.5, linestyle="-.", label="Target 0.45")
ax3.set_xticks(range(len(models_compare)))
ax3.set_xticklabels(list(models_compare.keys()), color=P["text"], fontsize=7)
ax3.set_ylabel("R²", color=P["text"])
ax3.set_title("R² v11 — Dual Pipeline vs records", color=P["text"], fontweight="bold")
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

fi_top20  = fi_B.head(20)   # importance Pipeline B (spoken+BERT)
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
ax6.set_title("Top 20 Features — Pipeline B (spoken)\n Green=Speech Purple=BERT Blue=Momentum Orange=Cross", color=P["text"], fontweight="bold", fontsize=9)
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

best_v11   = max(r["r2"] for r in global_results_v11.values())
versions   = ["v5\n(base)", "v6\n(momentum)", "v7\n(NLP+CV)", "v8\n(niches)", "v9\n(speech)", "v10\n(bert)"]
r2_history = [0.18, 0.24, 0.3332, 0.3310, 0.3176, best_v11]
bar_colors = ["#666","#888","#aaa","#ff4d6d","#888888",
              P["gold"] if best_v11 >= 0.45 else P["speech"]]

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

plt.savefig("stacking_v11_dual.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("   Dashboard saved -> stacking_v11_dual.png")
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
    for k, v in global_results_v11.items()
] + [
    {"version": "v9_record", "model": "global", "r2": RECORD_R2_GLOBAL, "mae": RECORD_MAE,
     "delta_v9": 0, "beats_record": False, "reaches_040": False, "reaches_045": False},
    {"version": "v8_record", "model": "Comedy",  "r2": RECORD_R2_COMEDY, "mae": np.nan,
     "delta_v9": RECORD_R2_COMEDY - RECORD_R2_GLOBAL, "beats_record": True,
     "reaches_040": True, "reaches_045": False}
])
results_df.to_csv("results_v11.csv", index=False)
print("   Results exported -> results_v11.csv")

fi_B.to_frame("importance").reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
    "feature_importance_v11.csv", index=False
)
print("   Feature importance -> feature_importance_v11.csv")

export_cols = ["creator_id_int", "video_rank", "niche", "has_speech", "speech_rate",
               "hook_score", "is_question", "is_list", "is_urgency", "hook_text",
               "sentiment_ratio", TARGET]
df[export_cols].to_csv("df_v11_enriched.csv", index=False)
print("   Enriched dataset -> df_v11_enriched.csv")

print("\n" + "=" * 72)
print("   STACKING v11 COMPLETE")
print("=" * 72)

best_model  = max(global_results_v11, key=lambda k: global_results_v11[k]["r2"])
best_r2     = global_results_v11[best_model]["r2"]
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

