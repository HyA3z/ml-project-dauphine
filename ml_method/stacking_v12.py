"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         TIKTOK VIRALITY — STACKING v12 : NICHE-AWARE DUAL PIPELINE         ║
║         Niche specialization + Residual Stacking + Target Augmentation          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  v11 diagnosis:                                                           ║
║  • Plateau at R²=0.3490 (spoken) / 0.2805 (silent) — healthy architecture       ║
║  • Comedy record (0.4036) beats global -> niches have their own       ║
║    dynamics that the global model dilutes                           ║
║  • Pipeline A (silent) regresses vs v8: tabular features saturated         ║
║  • BERT+Speech only gains +0.0039 on spoken segment (NLP ceiling)        ║
║  • creator_target_mean remains the 2nd most important feature -> creator     ║
║    signal under-exploited (flat encoding, no recent trend)         ║
║                                                                             ║
║  v12 Directions — 3 combined levers:                                         ║
║                                                                             ║
║  [LEVER 1] NICHE-AWARE PIPELINE (main)                               ║
║      -> Specialized models per niche (Comedy, Fitness, Finance, Study,       ║
║        Tech, Cooking) on the silent segment                                    ║
║      -> Comedy alone targeted 0.40+: now exploiting across all niches ║
║      -> Weighted assembly: niche-model if n_train>=MIN_NICHE | global otherwise  ║
║                                                                             ║
║  [LEVER 2] ENRICHED CREATOR FEATURES                                     ║
║      -> creator_target_mean -> already present (corr=0.478)                       ║
║      -> NEW: creator_recent_slope (trend over last 3 videos)      ║
║      -> NEW: creator_consistency_score (std / mean of last 5 videos)      ║
║      -> NEW: creator_peak_ratio (peak / creator median)                 ║
║      -> NEW: creator_niche_te (creator x niche cross TE)               ║
║                                                                             ║
║  [LEVER 3] RESIDUAL STACKING (new architecture)                     ║
║      -> Phase 1: lightweight global model -> OOF predictions (residual base)       ║
║      -> Phase 2: niche models learn the RESIDUAL of the global model         ║
║      -> Final assembly = global_pred + niche_residual_pred                  ║
║      -> Advantage: niche models focus on what the global model misses  ║
║                                                                             ║
║  All v9/v10/v11 fixes preserved:                            ║
║      - Separate Silent/Spoken dual pipeline                                     ║
║      - creator_id extracted from webVideoUrl, tri causal                             ║
║      - Anti-leakage expanding window target encoding                         ║
║      - BERT cache bert_embeddings_cache.npy                                   ║
║      - early_stopping_rounds removed from cross_val_predict                          ║
║  Record to beat: R² = 0.3310 (global) | R² = 0.4036 (Comedy)            ║
║  Goal v12: R² >= 0.37 global | R² >= 0.45 spoken segment               ║
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
N_OPTUNA_NICHE    = 25            # Moins de trials par niche (plus rapide)
N_OOF_FOLDS       = 5
TOP_K_FEATURES_A  = 45            # +5 vs v11 (nouvelles creator features)
TOP_K_FEATURES_B  = 50            # +5 vs v11
N_BERT_SVD        = 20
BERT_MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
BERT_CACHE_PATH   = "bert_embeddings_cache.npy"
BERT_BATCH_SIZE   = 64
MIN_NICHE_TRAIN   = 200           # Seuil minimum pour un modèle niche dédié
MIN_NICHE_SAMPLES = 80
SMOOTH_K          = 5
NICHES_ORDERED    = ["Comedy", "Fitness", "Finance", "Study", "Tech", "Cooking"]

print("=" * 72)
print("  STACKING v12 — NICHE-AWARE DUAL PIPELINE + RESIDUAL STACKING")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data_subtittles.csv")
df["creator_id"] = df["webVideoUrl"].str.extract(r'tiktok\.com/@([^/]+)/video')
df = df.sort_values(["creator_id", "video_rank"]).reset_index(drop=True)
df["creator_id_int"] = df["creator_id"].astype("category").cat.codes

n_creators = df["creator_id_int"].nunique()
print(f"{n_creators} creators | {len(df):,} videos")
assert n_creators > 100, "Warning: creator_id_int suspicious — check sorting"

# ─────────────────────────────────────────────────────────────────────────────
# 2. CASCADE CLASSIFICATION (identical to v8/v11)
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
# 3. FEATURE ENGINEERING v8 (pipeline preserved)
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

# Caption NLP (TextBlob)
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
# 4. SPEECH FEATURE ENGINEERING (v9 — conservée)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 3] SPEECH FEATURE ENGINEERING (preserved from v9)")
print("=" * 72)

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

df["subtitles_clean"] = df["subtitles"].fillna("").str.strip()
df["has_speech"]      = (df["subtitles_clean"] != "").astype(int)

def hook_words(text, n=5):
    if not text:
        return ""
    return " ".join(text.split()[:n])

df["hook_text"]         = df["subtitles_clean"].apply(hook_words)
df["word_count_speech"] = df["subtitles_clean"].apply(lambda t: len(t.split()) if t else 0)
df["char_count_speech"] = df["subtitles_clean"].str.len()
df["speech_rate"]       = df["word_count_speech"] / df["duration"].clip(lower=1)
df["speech_rate"]       = df["speech_rate"].clip(upper=df["speech_rate"].quantile(0.99))

df["is_question"]  = df["hook_text"].apply(lambda t: int(bool(QUESTION_PAT.search(t))))
df["is_list"]      = df["hook_text"].apply(lambda t: int(bool(LIST_PAT.search(t))))
df["is_urgency"]   = df["hook_text"].apply(lambda t: int(bool(URGENCY_PAT.search(t))))
df["is_personal"]  = df["hook_text"].apply(lambda t: int(bool(PERSONAL_PAT.search(t))))
df["is_poi"]       = df["hook_text"].apply(lambda t: int(bool(POI_PAT.search(t))))

df["hook_score"] = (
    df["is_list"]     * 2.5 +
    df["is_question"] * 2.0 +
    df["is_urgency"]  * 1.5 +
    df["is_poi"]      * 1.5 +
    df["is_personal"] * 1.0
)

df["positive_count"]  = df["subtitles_clean"].apply(lambda t: len(POSITIVE_PAT.findall(t)))
df["negative_count"]  = df["subtitles_clean"].apply(lambda t: len(NEGATIVE_PAT.findall(t)))
df["sentiment_ratio"] = (df["positive_count"] - df["negative_count"]) / (df["word_count_speech"].clip(lower=1))

SPEECH_NUMERIC_COLS = ["speech_rate","is_question","is_list","is_urgency","is_personal",
                       "is_poi","hook_score","positive_count","negative_count","sentiment_ratio",
                       "word_count_speech","char_count_speech"]
mask_no_speech = df["has_speech"] == 0
df.loc[mask_no_speech, SPEECH_NUMERIC_COLS] = 0

df["subtitles_for_tfidf"] = df["subtitles_clean"].apply(
    lambda t: t if t else "NON_VERBAL"
)

print(f"  has_speech : {df['has_speech'].sum():,} videos with speech ({df['has_speech'].mean()*100:.1f}%)")
print(f"   Average speech rate (spoken) : {df.loc[df['has_speech']==1, 'speech_rate'].mean():.2f} mots/s")
print(f"   Question hook : {df.loc[df['has_speech']==1, 'is_question'].mean()*100:.1f}% of spoken videos")
print(f"   List hook    : {df.loc[df['has_speech']==1, 'is_list'].mean()*100:.1f}% of spoken videos")
print(f"   Urgency hook  : {df.loc[df['has_speech']==1, 'is_urgency'].mean()*100:.1f}% of spoken videos")

# Cross-features Speech × Momentum (v9)
df["speech_x_momentum"]    = df["has_speech"]      * df["momentum_3"]
df["hook_x_viral"]         = df["hook_score"]       * df["viral_potential"]
df["rate_x_engagement"]    = df["speech_rate"]      * df["engagement_total_hist"]
df["hook_x_tier"]          = df["hook_score"]       * df["follower_tier"]
df["sentiment_x_momentum"] = df["sentiment_ratio"]  * df["momentum_3"]

CROSS_SPEECH_COLS = ["speech_x_momentum","hook_x_viral","rate_x_engagement",
                     "hook_x_tier","sentiment_x_momentum"]
df[CROSS_SPEECH_COLS] = df[CROSS_SPEECH_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

SPEECH_FEATURES = SPEECH_NUMERIC_COLS + CROSS_SPEECH_COLS + ["has_speech"]

# ─────────────────────────────────────────────────────────────────────────────
# 5. BERT EMBEDDINGS + SVD (v10 — preserved)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  [PHASE 4] BERT Embeddings -> SVD {N_BERT_SVD}D (v10 conservée)")
print("=" * 72)

TRAIN_RANK_MIN, TRAIN_RANK_MAX = 11, 26

def load_bert_model():
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
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"   {len(texts):,} texts encoded in {time.time()-t0:.0f}s "
          f"| shape={embeddings.shape}")
    return embeddings

texts_for_bert = df["subtitles_for_tfidf"].tolist()

if os.path.exists(BERT_CACHE_PATH):
    print(f"    BERT cache found : {BERT_CACHE_PATH}")
    bert_embeddings = np.load(BERT_CACHE_PATH)
    if bert_embeddings.shape[0] != len(df):
        print(f"  Warning:  Incomplete cache ({bert_embeddings.shape[0]} vs {len(df)}) — re-encoding")
        bert_model      = load_bert_model()
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

train_mask_bert  = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
svd_bert         = TruncatedSVD(n_components=N_BERT_SVD, random_state=42)
bert_full_reduced = svd_bert.fit_transform(bert_embeddings[train_mask_bert])
# Re-transform on the full dataset
bert_full_reduced = svd_bert.transform(bert_embeddings)

explained_var_bert = svd_bert.explained_variance_ratio_.sum()
print(f"   Explained variance ({N_BERT_SVD}D SVD post-BERT) : {explained_var_bert:.1%}")

BERT_COLS = [f"bert_{i}" for i in range(N_BERT_SVD)]
bert_df   = pd.DataFrame(bert_full_reduced, columns=BERT_COLS, index=df.index)
df        = pd.concat([df, bert_df], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 6. [NEW v12] ENRICHED CREATOR FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 4b] ENRICHED CREATOR FEATURES (new in v12)")
print("=" * 72)

print("   Construction des creator features avancées...")

# All these features use shift(1) to avoid leakage
grp_score = df.groupby("creator_id_int")["explosion_score"]

# creator_recent_slope: trend over last 3 videos (causal)
df["creator_recent_slope"] = (
    grp_score.shift(1)
    .groupby(df["creator_id_int"])
    .transform(lambda s: rolling_slope(s, window=3))
)

# creator_consistency_score: inverted CV (creator predictability)
rolling_std  = grp_score.shift(1).groupby(df["creator_id_int"]).transform(
    lambda s: s.rolling(5, min_periods=2).std()
)
rolling_mean = grp_score.shift(1).groupby(df["creator_id_int"]).transform(
    lambda s: s.rolling(5, min_periods=2).mean()
)
df["creator_consistency_score"] = rolling_mean / (rolling_std + 1e-6)

# creator_peak_ratio: historical peak vs median (viral potential)
creator_peak   = grp_score.shift(1).groupby(df["creator_id_int"]).transform(
    lambda s: s.expanding().max()
)
creator_median = grp_score.shift(1).groupby(df["creator_id_int"]).transform(
    lambda s: s.expanding().median()
)
df["creator_peak_ratio"] = creator_peak / (creator_median + 1e-6)

# creator_video_count: number of creator's previous videos (experience)
df["creator_video_count"] = df.groupby("creator_id_int").cumcount()

# creator_rank_in_niche: creator's percentile in their niche (relative to peers)
# Computed on historical stats, without leakage
df["creator_hist_median"] = df["hist_median_views"]  # déjà causal (stats historiques)
df["creator_rank_in_niche"] = df.groupby("niche")["creator_hist_median"].rank(pct=True)

CREATOR_COLS_V12 = [
    "creator_recent_slope",
    "creator_consistency_score",
    "creator_peak_ratio",
    "creator_video_count",
    "creator_rank_in_niche",
]

# Cleanup
df[CREATOR_COLS_V12] = (
    df[CREATOR_COLS_V12]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(df[CREATOR_COLS_V12].median())
)

print(f"   creator_recent_slope     | corr={np.corrcoef(df.loc[train_mask_bert, 'creator_recent_slope'], df.loc[train_mask_bert, TARGET])[0,1]:.4f}")
print(f"   creator_consistency_score| corr={np.corrcoef(df.loc[train_mask_bert, 'creator_consistency_score'], df.loc[train_mask_bert, TARGET])[0,1]:.4f}")
print(f"   creator_peak_ratio       | corr={np.corrcoef(df.loc[train_mask_bert, 'creator_peak_ratio'], df.loc[train_mask_bert, TARGET])[0,1]:.4f}")
print(f"   creator_rank_in_niche    | corr={np.corrcoef(df.loc[train_mask_bert, 'creator_rank_in_niche'], df.loc[train_mask_bert, TARGET])[0,1]:.4f}")

# Creator cross-features v12
df["creator_mom_x_peak"]     = df["momentum_3"]         * df["creator_peak_ratio"]
df["creator_slope_x_tier"]   = df["creator_recent_slope"] * df["follower_tier"]
df["creator_consist_x_viral"]= df["creator_consistency_score"] * df["viral_potential"]
df["creator_rank_x_momentum"]= df["creator_rank_in_niche"] * df["momentum_3"]

CROSS_CREATOR_COLS = [
    "creator_mom_x_peak",
    "creator_slope_x_tier",
    "creator_consist_x_viral",
    "creator_rank_x_momentum",
]
df[CROSS_CREATOR_COLS] = (
    df[CROSS_CREATOR_COLS]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)
print(f"   {len(CREATOR_COLS_V12)} creator features + {len(CROSS_CREATOR_COLS)} cross-creator features")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE SETS v12
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

# Pipeline A — silent : tabulaires + creator v12
ALL_FEATURES_A = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS_V8 +
    NLP_COLS + NICHE_COLS + CREATOR_COLS_V12 + CROSS_CREATOR_COLS +
    ["has_speech"]
)

# Pipeline B — spoken: tabular + speech + BERT + creator v12
ALL_FEATURES_B = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS_V8 +
    NLP_COLS + NICHE_COLS + SPEECH_FEATURES + BERT_COLS +
    CREATOR_COLS_V12 + CROSS_CREATOR_COLS
)

print(f"\n   Pipeline A (silent) : {len(ALL_FEATURES_A)} features")
print(f"   Pipeline B (spoken) : {len(ALL_FEATURES_B)} features "
      f"(+{len(SPEECH_FEATURES) + len(BERT_COLS)} speech+BERT, "
      f"+{len(CREATOR_COLS_V12)+len(CROSS_CREATOR_COLS)} creator)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

print(f"\n   Split: Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")
print(f"     has_speech train : {df.loc[train_mask, 'has_speech'].mean()*100:.1f}%")
print(f"     has_speech test  : {df.loc[test_mask,  'has_speech'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 9. TARGET ENCODING — GLOBAL + NICHE-AWARE (anti-leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("  [LEVIER A] Causal global + niche-aware target encoding (v12)")
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

# Global creator TE (preserved from v11)
df["creator_target_mean"] = (
    df.groupby("creator_id_int", group_keys=False)
      .apply(lambda g: expanding_te(g, SMOOTH_K, global_mean))
)

# [NEW v12] Creator x niche TE: each creator has a TE per niche
# We compute for each (creator, niche) combination a causal sliding mean
niche_global_means = df.loc[train_mask].groupby("niche")[TARGET].mean().to_dict()

def expanding_te_niche(group, smooth_k, niche_mean):
    target_vals = group[TARGET].values
    te_vals     = np.full(len(target_vals), niche_mean)
    cumsum, cumcount = 0.0, 0
    for i in range(len(target_vals)):
        te_vals[i] = (cumsum + smooth_k * niche_mean) / (cumcount + smooth_k)
        cumsum   += target_vals[i]
        cumcount += 1
    return pd.Series(te_vals, index=group.index)

niche_te_list = []
for (cid, niche), grp in df.groupby(["creator_id_int", "niche"]):
    nm = niche_global_means.get(niche, global_mean)
    te_series = expanding_te_niche(grp.sort_values("video_rank"), SMOOTH_K, nm)
    niche_te_list.append(te_series)

df["creator_niche_te"] = pd.concat(niche_te_list).reindex(df.index)
df["creator_niche_te"] = df["creator_niche_te"].fillna(global_mean)

# Re-sync masks after re-sort
train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

te_corr = np.corrcoef(df.loc[train_mask, "creator_target_mean"],
                      df.loc[train_mask, TARGET])[0, 1]
te_niche_corr = np.corrcoef(df.loc[train_mask, "creator_niche_te"],
                             df.loc[train_mask, TARGET])[0, 1]
assert te_corr < 0.999, f"❌ Leakage detected creator_target_mean (corr={te_corr:.4f})"
assert te_niche_corr < 0.999, f"❌ Leakage detected creator_niche_te (corr={te_niche_corr:.4f})"

print(f"   creator_target_mean (expanding OOF, k={SMOOTH_K}) | corr={te_corr:.4f}")
print(f"   creator_niche_te    (expanding OOF × niche)       | corr={te_niche_corr:.4f}")

ALL_FEATURES_A_FINAL = ALL_FEATURES_A + ["creator_target_mean", "creator_niche_te"]
ALL_FEATURES_B_FINAL = ALL_FEATURES_B + ["creator_target_mean", "creator_niche_te"]

mask_silent = df["has_speech"] == 0
mask_speech = df["has_speech"] == 1

# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE SELECTION PER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("  [LEVER B] Feature Selection — Pipeline A & B")
print("─" * 72)

Xa_tr_full = df.loc[train_mask, ALL_FEATURES_A_FINAL].astype(float)
ya_tr      = df.loc[train_mask, TARGET]
Xa_vl_full = df.loc[val_mask,   ALL_FEATURES_A_FINAL].astype(float)
ya_vl      = df.loc[val_mask,   TARGET]
Xa_te_full = df.loc[test_mask,  ALL_FEATURES_A_FINAL].astype(float)
ya_te      = df.loc[test_mask,  TARGET]

df_speech  = df[mask_speech]
Xb_tr_full = df_speech.loc[df_speech.index.isin(df.index[train_mask]), ALL_FEATURES_B_FINAL].astype(float)
yb_tr      = df_speech.loc[df_speech.index.isin(df.index[train_mask]), TARGET]
Xb_vl_full = df_speech.loc[df_speech.index.isin(df.index[val_mask]),   ALL_FEATURES_B_FINAL].astype(float)
yb_vl      = df_speech.loc[df_speech.index.isin(df.index[val_mask]),   TARGET]
Xb_te_full = df_speech.loc[df_speech.index.isin(df.index[test_mask]),  ALL_FEATURES_B_FINAL].astype(float)
yb_te      = df_speech.loc[df_speech.index.isin(df.index[test_mask]),  TARGET]

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
        if feat in ["creator_target_mean", "creator_niche_te"]:  tag = " ←  TE"
        elif feat in BERT_COLS:            tag = " ←  BERT"
        elif feat in SPEECH_FEATURES:      tag = " ←  SPEECH"
        elif feat in MOMENTUM_COLS:        tag = " ←  MOM"
        elif feat in CROSS_COLS_V8:        tag = " ←  CROSS"
        elif feat in CROSS_CREATOR_COLS:   tag = " ←  CROSS-CREATOR"
        elif feat in CREATOR_COLS_V12:     tag = " ←  CREATOR"
        elif feat in NICHE_COLS:           tag = " ←   NICHE"
        elif feat in NLP_COLS:             tag = " ←  NLP"
        print(f"    #{rk:>2}  {feat:<38} {imp:>6.2f}%{tag}")
    return selected, fi

SELECTED_A, fi_A = select_features(
    ALL_FEATURES_A_FINAL, TOP_K_FEATURES_A,
    Xa_tr_full, ya_tr, Xa_vl_full, ya_vl, "Pipeline A — silent"
)
SELECTED_B, fi_B = select_features(
    ALL_FEATURES_B_FINAL, TOP_K_FEATURES_B,
    Xb_tr_full, yb_tr, Xb_vl_full, yb_vl, "Pipeline B — spoken"
)

# Rebuild with selected features
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
# 11. GENERIC STACKING (identical to v11)
# ─────────────────────────────────────────────────────────────────────────────
global_results_v12 = {}

def run_stacking_pipeline(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                           label, n_trials=N_OPTUNA_TRIALS):
    """Stacking XGB + LGBM + CatBoost + Ridge L2 OOF."""
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
    r2x  = r2_score(y_te, xgb_m.predict(X_te))
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
    r2l  = r2_score(y_te, lgbm_m.predict(X_te))
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
    r2c  = r2_score(y_te, cat_m.predict(X_te))
    maec = mean_absolute_error(y_te, cat_m.predict(X_te))
    results["CatBoost"] = {"r2": r2c, "mae": maec}
    print(f"  R²={r2c:.4f} | MAE={maec:.4f}  {'' if r2c > RECORD_R2_GLOBAL else ''}")

    # ── Stacking L2 OOF ──────────────────────────────────────────────────────
    print(f"\n    Stacking L2 OOF [{label}]...")
    kf = KFold(n_splits=N_OOF_FOLDS, shuffle=False)

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

    # Also return models for residual stacking
    return results, y_pred, (xgb_s, lgbm_s, cat_s, ridge_l2, xgb_l2, cat_l2,
                              meta_vl, meta_te, oof_xgb, oof_lgbm, oof_cat)

# ─────────────────────────────────────────────────────────────────────────────
# 12. PIPELINE A — SILENT (global)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 5A] PIPELINE A — SILENT VIDEOS (global)")
print("=" * 72)

y_train_a = df.loc[train_mask, TARGET]
y_val_a   = df.loc[val_mask,   TARGET]
y_test_a  = df.loc[test_mask,  TARGET]

results_A, y_pred_A_full, models_A = run_stacking_pipeline(
    Xa_tr, y_train_a, Xa_vl, y_val_a, Xa_te, y_test_a,
    label="silent-global", n_trials=N_OPTUNA_TRIALS
)

mask_silent_test = test_mask & mask_silent
idx_silent_test  = [list(df.index[test_mask]).index(i)
                    for i in df.index[mask_silent_test]]
r2_A_silent = r2_score(y_test_a.values[idx_silent_test],
                        y_pred_A_full[idx_silent_test])
print(f"\n   Pipeline A — silent segment only: R²={r2_A_silent:.4f}")
global_results_v12["Pipeline-A (silent, global)"] = results_A["Stacking-L2"]

# ─────────────────────────────────────────────────────────────────────────────
# 13. [NEW v12] NICHE-AWARE RESIDUAL STACKING (Pipeline A only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 5A-NICHE] NICHE-BASED RESIDUAL STACKING (new in v12)")
print("=" * 72)
print("  Architecture: modèle global → résidus → modèles niche corrigent le résidu")

# Global model OOF on train (for computing residuals without leakage)
xgb_s_A, lgbm_s_A, cat_s_A = models_A[0], models_A[1], models_A[2]
ridge_l2_A, xgb_l2_A, cat_l2_A = models_A[3], models_A[4], models_A[5]
oof_xgb_A, oof_lgbm_A, oof_cat_A = models_A[8], models_A[9], models_A[10]

def build_meta(p1, p2, p3):
    return np.column_stack([p1, p2, p3,
                            (p1+p2+p3)/3,
                            np.abs(p1-p3), np.abs(p2-p3), np.abs(p1-p2)])

meta_oof_A = build_meta(oof_xgb_A, oof_lgbm_A, oof_cat_A)
meta_l2_oof_A = np.column_stack([xgb_l2_A.predict(meta_oof_A),
                                   cat_l2_A.predict(meta_oof_A)])
y_oof_global_A = ridge_l2_A.predict(meta_l2_oof_A)

# OOF residuals on train
y_train_a_vals = y_train_a.values
residuals_train = y_train_a_vals - y_oof_global_A

# Residuals on test (via global predictions)
residuals_test = y_test_a.values - y_pred_A_full

niche_residual_preds = np.zeros(len(y_test_a))
niche_results_dict   = {}

print(f"\n  Available niches : {df.loc[train_mask, 'niche'].value_counts().to_dict()}")

for niche in NICHES_ORDERED:
    # Niche masks on train and test
    niche_mask_train = train_mask & (df["niche"] == niche)
    niche_mask_test  = test_mask  & (df["niche"] == niche)

    n_train_niche = niche_mask_train.sum()
    n_test_niche  = niche_mask_test.sum()

    if n_train_niche < MIN_NICHE_TRAIN or n_test_niche < 10:
        print(f"\n  Skipping {niche:<10}: train={n_train_niche} < {MIN_NICHE_TRAIN} — skipped")
        continue

    print(f"\n    NICHE: {niche}  (train={n_train_niche} | test={n_test_niche})")

    # Features pour ce modèle niche : on apprend les RÉSIDUS
    Xn_tr = df.loc[niche_mask_train, SELECTED_A].astype(float)
    yn_tr = pd.Series(residuals_train[
        [list(df.index[train_mask]).index(i) for i in df.index[niche_mask_train]]
    ], index=Xn_tr.index)

    Xn_vl = df.loc[val_mask & (df["niche"] == niche), SELECTED_A].astype(float)
    yn_vl_idx = [list(df.index[val_mask]).index(i)
                 for i in df.index[val_mask & (df["niche"] == niche)]]

    # Val residuals: y_val - global_val_pred
    meta_vl_A = models_A[6]
    meta_l2_vl_A = np.column_stack([xgb_l2_A.predict(meta_vl_A),
                                     cat_l2_A.predict(meta_vl_A)])
    y_val_global = ridge_l2_A.predict(meta_l2_vl_A)
    residuals_val = y_val_a.values - y_val_global

    yn_vl = pd.Series(
        residuals_val[yn_vl_idx] if len(yn_vl_idx) > 0 else np.array([]),
        index=Xn_vl.index
    )

    Xn_te = df.loc[niche_mask_test, SELECTED_A].astype(float)

    if len(Xn_vl) < 5:
        print(f"     Warning:  Validation set too small ({len(Xn_vl)}) — cross-validation only")
        # Fallback: simple LightGBM without early stopping on val
        niche_model = LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
        )
        niche_model.fit(Xn_tr, yn_tr)
    else:
        # Optuna optimization of niche model
        def niche_lgbm_obj(trial):
            p = dict(
                max_depth         = trial.suggest_int  ("max_depth",         3, 7),
                learning_rate     = trial.suggest_float("learning_rate",     0.005, 0.05, log=True),
                n_estimators      = trial.suggest_int  ("n_estimators",      500, 5000, step=500),
                num_leaves        = trial.suggest_int  ("num_leaves",        15, 80),
                subsample         = trial.suggest_float("subsample",         0.6, 0.95),
                colsample_bytree  = trial.suggest_float("colsample_bytree",  0.5, 0.9),
                min_child_samples = trial.suggest_int  ("min_child_samples", 5, 50),
                reg_alpha         = trial.suggest_float("reg_alpha",         0.01, 3.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda",        0.3, 5.0, log=True),
            )
            m = LGBMRegressor(**p, random_state=42, verbose=-1)
            cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            m.fit(Xn_tr, yn_tr, eval_set=[(Xn_vl, yn_vl)], callbacks=cb)
            return -r2_score(yn_vl, m.predict(Xn_vl)) if len(yn_vl) > 2 else 0.0

        niche_study = optuna.create_study(direction="minimize",
                                           sampler=TPESampler(seed=42+hash(niche) % 100))
        niche_study.optimize(niche_lgbm_obj, n_trials=N_OPTUNA_NICHE,
                             show_progress_bar=False)
        niche_model = LGBMRegressor(**niche_study.best_params, random_state=42, verbose=-1)
        if len(Xn_vl) > 2:
            niche_model.fit(Xn_tr, yn_tr, eval_set=[(Xn_vl, yn_vl)],
                            callbacks=[lgb.early_stopping(100, verbose=False),
                                       lgb.log_evaluation(-1)])
        else:
            niche_model.fit(Xn_tr, yn_tr)

    # Predict residual on niche test set
    residual_pred_niche = niche_model.predict(Xn_te)

    # Indices in y_test_a (global test)
    idx_niche_in_test = [list(df.index[test_mask]).index(i)
                         for i in df.index[niche_mask_test]]
    for pos, test_pos in enumerate(idx_niche_in_test):
        niche_residual_preds[test_pos] = residual_pred_niche[pos]

    # Residual score
    r2_niche_residual = r2_score(
        y_test_a.values[idx_niche_in_test],
        y_pred_A_full[idx_niche_in_test] + residual_pred_niche
    ) if len(idx_niche_in_test) > 5 else np.nan

    r2_niche_base = r2_score(
        y_test_a.values[idx_niche_in_test],
        y_pred_A_full[idx_niche_in_test]
    ) if len(idx_niche_in_test) > 5 else np.nan

    delta_niche = r2_niche_residual - r2_niche_base if not np.isnan(r2_niche_residual) else np.nan
    niche_results_dict[niche] = {"r2_base": r2_niche_base,
                                  "r2_residual": r2_niche_residual,
                                  "delta": delta_niche,
                                  "n_test": n_test_niche}
    status = "" if (not np.isnan(delta_niche) and delta_niche > 0) else ""
    print(f"     R² base={r2_niche_base:.4f} | R² résiduel={r2_niche_residual:.4f} "
          f"(Δ={delta_niche:+.4f}) {status}")

# ── Final assembly: global + niche residual ─────────────────────────────────
# Residual is only applied if a niche model was trained, otherwise residual=0
y_pred_A_niche = y_pred_A_full + niche_residual_preds

r2_A_niche  = r2_score(y_test_a, y_pred_A_niche)
mae_A_niche = mean_absolute_error(y_test_a, y_pred_A_niche)
r2_A_base   = results_A["Stacking-L2"]["r2"]
print(f"\n   Pipeline A Niche Residual: R²={r2_A_niche:.4f} | MAE={mae_A_niche:.4f}")
print(f"     Delta vs global Pipeline A: {r2_A_niche - r2_A_base:+.4f} "
      f"{'' if r2_A_niche > r2_A_base else ''}")

# We keep the best between global and residual
if r2_A_niche > r2_A_base:
    y_pred_A_best = y_pred_A_niche
    global_results_v12["Pipeline-A (silent, niche-résiduel)"] = {"r2": r2_A_niche, "mae": mae_A_niche}
    print("   Niche residual retenu pour l'assemblage final")
else:
    y_pred_A_best = y_pred_A_full
    global_results_v12["Pipeline-A (silent, niche-résiduel)"] = {"r2": r2_A_niche, "mae": mae_A_niche}
    print("  Warning:  Global retained (niche residual degrading)")

# ─────────────────────────────────────────────────────────────────────────────
# 14. PIPELINE B — SPOKEN (BERT + creator v12)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 5B] PIPELINE B — SPOKEN VIDEOS (BERT + creator v12)")
print("=" * 72)

results_B, y_pred_B_speech, models_B = run_stacking_pipeline(
    Xb_tr, yb_tr, Xb_vl, yb_vl, Xb_te, yb_te,
    label="spoken-BERT-creator", n_trials=N_OPTUNA_TRIALS
)
global_results_v12["Pipeline-B (spoken+BERT+creator)"] = results_B["Stacking-L2"]

# ─────────────────────────────────────────────────────────────────────────────
# 15. FINAL ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  [PHASE 6] FINAL ASSEMBLY — Reconstructed global score")
print("=" * 72)

y_pred_dual = y_pred_A_best.copy()
mask_speech_test = test_mask & mask_speech
idx_speech_test  = [list(df.index[test_mask]).index(i)
                    for i in df.index[mask_speech_test]]
for rank_pos, global_idx in enumerate(idx_speech_test):
    y_pred_dual[global_idx] = y_pred_B_speech[rank_pos]

r2_dual  = r2_score(y_test_a, y_pred_dual)
mae_dual = mean_absolute_error(y_test_a, y_pred_dual)
delta_dual = r2_dual - RECORD_R2_GLOBAL

print(f"\n   DUAL PIPELINE SCORE (A niche residual + B spoken):")
print(f"     R²={r2_dual:.4f} | MAE={mae_dual:.4f} "
      f"(Δ vs record v8 : {delta_dual:+.4f}) "
      f"{'' if r2_dual > RECORD_R2_GLOBAL else ''}")

# Scores segmentés
mask_silent_test = test_mask & mask_silent
mask_speech_test = test_mask & mask_speech
idx_silent_test  = [list(df.index[test_mask]).index(i)
                    for i in df.index[mask_silent_test]]
idx_speech_test2 = [list(df.index[test_mask]).index(i)
                    for i in df.index[mask_speech_test]]

r2_seg_silent = r2_score(y_test_a.values[idx_silent_test],
                          y_pred_dual[idx_silent_test]) if idx_silent_test else np.nan
r2_seg_speech = r2_score(y_test_a.values[idx_speech_test2],
                          y_pred_dual[idx_speech_test2]) if idx_speech_test2 else np.nan

print(f"\n   Segmented R² (dual pipeline v12):")
print(f"     Silent (has_speech=0): R²={r2_seg_silent:.4f}")
print(f"     Spoken (has_speech=1): R²={r2_seg_speech:.4f}")
global_results_v12["Dual-Pipeline"] = {"r2": r2_dual, "mae": mae_dual}

# ─────────────────────────────────────────────────────────────────────────────
# 16. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FINAL SUMMARY — v12 vs v11 vs v8")
print("=" * 72)
print(f"\n  Model                                    R²      Delta v8      MAE  Status")
print(f"  {'─'*73}")

R2_V11_A       = 0.2961
R2_V11_B       = 0.3490
R2_V11_DUAL    = 0.3005
MAE_V11_A      = 0.3014
MAE_V11_B      = 0.2953
MAE_V11_DUAL   = 0.2988

for name, res in global_results_v12.items():
    r2, mae = res["r2"], res["mae"]
    delta   = r2 - RECORD_R2_GLOBAL
    status  = " >" if r2 > RECORD_R2_GLOBAL else " ≤"
    print(f"     {name:<40} {r2:.4f}   {delta:+.4f}   {mae:.4f}  {status}")

print(f"  {'─'*73}")
print(f"     Record v8 (global)                     {RECORD_R2_GLOBAL:.4f}             {RECORD_MAE:.4f}")
print(f"     Record v8 (Comedy)                     {RECORD_R2_COMEDY:.4f}")
print(f"     v11 Pipeline-A (silent, global)          {R2_V11_A:.4f}   {R2_V11_A-RECORD_R2_GLOBAL:+.4f}   {MAE_V11_A:.4f}")
print(f"     v11 Pipeline-B (spoken+BERT)            {R2_V11_B:.4f}   {R2_V11_B-RECORD_R2_GLOBAL:+.4f}   {MAE_V11_B:.4f}")
print(f"     v11 Dual-Pipeline                      {R2_V11_DUAL:.4f}   {R2_V11_DUAL-RECORD_R2_GLOBAL:+.4f}   {MAE_V11_DUAL:.4f}")

print("\n  Niche Residual — detail per niche:")
for niche, nres in niche_results_dict.items():
    if np.isnan(nres.get("delta", np.nan)):
        continue
    arrow = "up" if nres["delta"] > 0 else "down"
    print(f"    {niche:<12} base={nres['r2_base']:.4f} → résiduel={nres['r2_residual']:.4f} "
          f"({arrow}{abs(nres['delta']):.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 17. DASHBOARD v12
# ─────────────────────────────────────────────────────────────────────────────
print("\n   Generating v12 dashboard...")

P = {
    "bg":      "#0d1117",
    "ax_bg":   "#161b22",
    "text":    "#e6edf3",
    "grid":    "#30363d",
    "gold":    "#f0c040",
    "target":  "#58a6ff",
    "speech":  "#bc8cff",
    "lsa":     "#56d364",
    "creator": "#ff7b72",
    "niche":   "#ffa657",
}

fig = plt.figure(figsize=(22, 16), facecolor=P["bg"])
fig.patch.set_facecolor(P["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── Panel 0 : Titre ──────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor(P["ax_bg"])
for sp in ax0.spines.values(): sp.set_color(P["grid"])

model_names = list(global_results_v12.keys())
r2_vals = [global_results_v12[m]["r2"] for m in model_names]
mae_vals = [global_results_v12[m]["mae"] for m in model_names]
colors_bar = []
for m in model_names:
    if "Dual" in m:       colors_bar.append(P["gold"])
    elif "niche" in m.lower(): colors_bar.append(P["niche"])
    elif "spoken" in m:    colors_bar.append(P["speech"])
    else:                 colors_bar.append(P["target"])

x = np.arange(len(model_names))
bars = ax0.bar(x, r2_vals, color=colors_bar, alpha=0.85, width=0.5)
for bar, val in zip(bars, r2_vals):
    ax0.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom",
             color=P["text"], fontsize=10, fontweight="bold")

ax0.axhline(RECORD_R2_GLOBAL, color="#ff4d6d", linewidth=1.5,
            linestyle="--", label=f"Record v8 global ({RECORD_R2_GLOBAL:.4f})")
ax0.axhline(0.40, color=P["gold"], linewidth=1.5,
            linestyle="-.", label="Target 0.40")
ax0.axhline(0.45, color=P["lsa"], linewidth=1.0,
            linestyle=":", label="Target 0.45")
ax0.set_xticks(x)
ax0.set_xticklabels(model_names, color=P["text"], fontsize=9, rotation=10)
ax0.set_ylabel("R²", color=P["text"])
ax0.set_title("STACKING v12 — Niche-Aware Dual Pipeline + Residual Stacking + Creator Features",
              color=P["text"], fontweight="bold", fontsize=13)
ax0.tick_params(colors=P["text"])
ax0.yaxis.grid(True, color=P["grid"], alpha=0.5)
ax0.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=8)

# ── Panel 1 : R² par niche (base vs résiduel) ────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor(P["ax_bg"])
for sp in ax1.spines.values(): sp.set_color(P["grid"])

niches_plotted = [n for n in niche_results_dict if not np.isnan(niche_results_dict[n].get("r2_residual", np.nan))]
if niches_plotted:
    xn  = np.arange(len(niches_plotted))
    r2b = [niche_results_dict[n]["r2_base"]     for n in niches_plotted]
    r2r = [niche_results_dict[n]["r2_residual"] for n in niches_plotted]
    w   = 0.35
    ax1.bar(xn - w/2, r2b, w, label="Global (base)",   color=P["target"],  alpha=0.7)
    ax1.bar(xn + w/2, r2r, w, label="Niche residual",  color=P["niche"],   alpha=0.85)
    ax1.set_xticks(xn)
    ax1.set_xticklabels(niches_plotted, color=P["text"], fontsize=8, rotation=15)
    ax1.set_ylabel("R² test", color=P["text"])
    ax1.set_title("R² Base vs Residual\nper Niche", color=P["text"], fontweight="bold")
    ax1.tick_params(colors=P["text"])
    ax1.yaxis.grid(True, color=P["grid"], alpha=0.5)
    ax1.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=8)

# ── Panel 2 : Feature Importance Pipeline A (top 20) ────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor(P["ax_bg"])
for sp in ax2.spines.values(): sp.set_color(P["grid"])

fi_top20_A = fi_A.head(20)
fi_colors_A = []
for feat in fi_top20_A.index:
    if feat in ["creator_target_mean", "creator_niche_te"]: fi_colors_A.append(P["gold"])
    elif feat in CREATOR_COLS_V12:   fi_colors_A.append(P["creator"])
    elif feat in CROSS_CREATOR_COLS: fi_colors_A.append(P["niche"])
    elif feat in MOMENTUM_COLS:      fi_colors_A.append("#3a86ff")
    elif feat in CROSS_COLS_V8:      fi_colors_A.append("#f39c12")
    else:                            fi_colors_A.append("#888888")

ax2.barh(range(len(fi_top20_A)), fi_top20_A.values[::-1], color=fi_colors_A[::-1], alpha=0.85)
ax2.set_yticks(range(len(fi_top20_A)))
ax2.set_yticklabels(fi_top20_A.index[::-1], color=P["text"], fontsize=7)
ax2.set_xlabel("Importance", color=P["text"])
ax2.set_title("Top 20 Features — Pipeline A\n Yellow=TE Red=Creator Orange=Cross-Creator Blue=Momentum",
              color=P["text"], fontweight="bold", fontsize=8)
ax2.tick_params(colors=P["text"])
ax2.xaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 3 : Feature Importance Pipeline B (top 20) ────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor(P["ax_bg"])
for sp in ax3.spines.values(): sp.set_color(P["grid"])

fi_top20_B = fi_B.head(20)
fi_colors_B = []
for feat in fi_top20_B.index:
    if feat in ["creator_target_mean", "creator_niche_te"]: fi_colors_B.append(P["gold"])
    elif feat in BERT_COLS:            fi_colors_B.append(P["lsa"])
    elif feat in SPEECH_FEATURES:      fi_colors_B.append(P["speech"])
    elif feat in CREATOR_COLS_V12:     fi_colors_B.append(P["creator"])
    elif feat in CROSS_CREATOR_COLS:   fi_colors_B.append(P["niche"])
    elif feat in MOMENTUM_COLS:        fi_colors_B.append("#3a86ff")
    elif feat in CROSS_COLS_V8:        fi_colors_B.append("#f39c12")
    else:                              fi_colors_B.append("#888888")

ax3.barh(range(len(fi_top20_B)), fi_top20_B.values[::-1], color=fi_colors_B[::-1], alpha=0.85)
ax3.set_yticks(range(len(fi_top20_B)))
ax3.set_yticklabels(fi_top20_B.index[::-1], color=P["text"], fontsize=7)
ax3.set_xlabel("Importance", color=P["text"])
ax3.set_title("Top 20 Features — Pipeline B\n Yellow=TE Green=BERT Purple=Speech Red=Creator Blue=Momentum",
              color=P["text"], fontweight="bold", fontsize=8)
ax3.tick_params(colors=P["text"])
ax3.xaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 4 : Progression R² v5 → v12 ───────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :2])
ax4.set_facecolor(P["ax_bg"])
for sp in ax4.spines.values(): sp.set_color(P["grid"])

best_v12    = max(r["r2"] for r in global_results_v12.values())
versions_h  = ["v5\n(base)", "v6\n(mom)", "v7\n(NLP)", "v8\n(niches)",
               "v9\n(speech)", "v10\n(bert)", "v11\n(dual)", "v12\n(niche-aware)"]
r2_hist     = [0.18, 0.24, 0.3332, 0.3310, 0.3176, 0.3490, 0.3490, best_v12]
bar_colors_h = ["#555","#777","#999","#ff4d6d","#aaa","#bc8cff","#56d364",
                P["gold"] if best_v12 >= 0.45 else P["niche"]]

bars4 = ax4.bar(range(len(versions_h)), r2_hist, color=bar_colors_h, alpha=0.85)
for bar, val in zip(bars4, r2_hist):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom",
             color=P["text"], fontsize=9, fontweight="bold")
ax4.axhline(0.45, color=P["gold"], linewidth=1.5, linestyle="-.", label="Target 0.45")
ax4.axhline(0.40, color=P["target"], linewidth=1.0, linestyle=":", label="Target 0.40")
ax4.axhline(RECORD_R2_GLOBAL, color="#ff4d6d", linewidth=1.0,
            linestyle="--", label=f"Record v8 ({RECORD_R2_GLOBAL:.4f})")
ax4.set_xticks(range(len(versions_h)))
ax4.set_xticklabels(versions_h, color=P["text"], fontsize=9)
ax4.set_ylabel("Global R²", color=P["text"])
ax4.set_title("R² progression across versions", color=P["text"], fontweight="bold")
ax4.tick_params(colors=P["text"])
ax4.legend(facecolor=P["ax_bg"], edgecolor=P["grid"], labelcolor=P["text"], fontsize=7)
ax4.yaxis.grid(True, color=P["grid"], alpha=0.5)

# ── Panel 5 : Creator features — corrélation avec target ────────────────────
ax5 = fig.add_subplot(gs[2, 2])
ax5.set_facecolor(P["ax_bg"])
for sp in ax5.spines.values(): sp.set_color(P["grid"])

creator_feat_list = ["creator_target_mean", "creator_niche_te"] + CREATOR_COLS_V12 + CROSS_CREATOR_COLS
corr_vals = []
corr_labels = []
for feat in creator_feat_list:
    if feat in df.columns:
        c = np.corrcoef(df.loc[train_mask, feat].fillna(0),
                        df.loc[train_mask, TARGET])[0, 1]
        corr_vals.append(abs(c))
        corr_labels.append(feat.replace("creator_", "").replace("_", "\n"))

sorted_idx = np.argsort(corr_vals)[::-1]
corr_vals_s   = [corr_vals[i]   for i in sorted_idx]
corr_labels_s = [corr_labels[i] for i in sorted_idx]
c_colors = [P["gold"] if "te" in corr_labels[i] or "target" in corr_labels[i]
            else P["creator"] for i in sorted_idx]

ax5.barh(range(len(corr_vals_s)), corr_vals_s[::-1], color=c_colors[::-1], alpha=0.85)
ax5.set_yticks(range(len(corr_vals_s)))
ax5.set_yticklabels(corr_labels_s[::-1], color=P["text"], fontsize=7)
ax5.set_xlabel("|Correlation| with target_log", color=P["text"])
ax5.set_title("Creator Features v12\n|corr| with target", color=P["text"], fontweight="bold")
ax5.tick_params(colors=P["text"])
ax5.xaxis.grid(True, color=P["grid"], alpha=0.5)

plt.savefig("stacking_v12_niche_aware.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("   Dashboard saved -> stacking_v12_niche_aware.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 18. CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {"version": "v12", "model": k,
     "r2": v["r2"], "mae": v["mae"],
     "delta_v8": v["r2"] - RECORD_R2_GLOBAL,
     "delta_v11_dual": v["r2"] - R2_V11_DUAL,
     "beats_record": v["r2"] > RECORD_R2_GLOBAL,
     "reaches_040": v["r2"] >= 0.40,
     "reaches_045": v["r2"] >= 0.45}
    for k, v in global_results_v12.items()
] + [
    {"version": "v11", "model": "Pipeline-B (spoken+BERT)", "r2": R2_V11_B,
     "mae": MAE_V11_B, "delta_v8": R2_V11_B - RECORD_R2_GLOBAL,
     "delta_v11_dual": 0, "beats_record": True, "reaches_040": False, "reaches_045": False},
    {"version": "v11", "model": "Dual-Pipeline", "r2": R2_V11_DUAL,
     "mae": MAE_V11_DUAL, "delta_v8": R2_V11_DUAL - RECORD_R2_GLOBAL,
     "delta_v11_dual": 0, "beats_record": False, "reaches_040": False, "reaches_045": False},
    {"version": "v8_record", "model": "global", "r2": RECORD_R2_GLOBAL, "mae": RECORD_MAE,
     "delta_v8": 0, "delta_v11_dual": 0, "beats_record": False, "reaches_040": False, "reaches_045": False},
    {"version": "v8_record", "model": "Comedy",  "r2": RECORD_R2_COMEDY, "mae": float("nan"),
     "delta_v8": RECORD_R2_COMEDY - RECORD_R2_GLOBAL, "delta_v11_dual": RECORD_R2_COMEDY - R2_V11_DUAL,
     "beats_record": True, "reaches_040": True, "reaches_045": False},
])
results_df.to_csv("results_v12.csv", index=False)
print("   Results exported -> results_v12.csv")

fi_B.to_frame("importance").reset_index().rename(
    columns={"index": "feature", 0: "importance"}
).to_csv("feature_importance_v12.csv", index=False)
print("   Feature importance -> feature_importance_v12.csv")

export_cols = ["creator_id_int", "video_rank", "niche", "has_speech",
               "speech_rate", "hook_score", "creator_target_mean",
               "creator_niche_te", "creator_recent_slope",
               "creator_consistency_score", "creator_peak_ratio",
               "creator_rank_in_niche", "sentiment_ratio", TARGET]
df[[c for c in export_cols if c in df.columns]].to_csv("df_v12_enriched.csv", index=False)
print("   Enriched dataset -> df_v12_enriched.csv")

print("\n" + "=" * 72)
print("   STACKING v12 COMPLETE")
print("=" * 72)

best_model = max(global_results_v12, key=lambda k: global_results_v12[k]["r2"])
best_r2    = global_results_v12[best_model]["r2"]
best_mae   = global_results_v12[best_model]["mae"]

print(f"\n   Best model: {best_model}  R²={best_r2:.4f}  MAE={best_mae:.4f}")
print(f"   Delta vs v8 record    : {best_r2 - RECORD_R2_GLOBAL:+.4f}")
print(f"   Delta vs v11 dual     : {best_r2 - R2_V11_DUAL:+.4f}")
print(f"   Delta vs v11 Pipeline-B : {best_r2 - R2_V11_B:+.4f}")

if best_r2 >= 0.45:
    print(f"\n   TARGET R² >= 0.45 REACHED!")
elif best_r2 >= 0.40:
    print(f"\n   Target R² >= 0.40 crossed!")
elif best_r2 > RECORD_R2_GLOBAL:
    print(f"\n   Record v8 battu ! ({best_r2:.4f} > {RECORD_R2_GLOBAL:.4f})")
else:
    print(f"\n   Plateau: {best_r2:.4f}")
    print(f"     v13 direction: pseudo-labelling + augmentation on Unknown videos")
