"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       TIKTOK VIRALITY — STACKING v13 : MIXTURE OF EXPERTS (MoE)              ║
║       Architecture : Router → 3 Experts → Stacking → Global Assembly         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MoE Architecture :                                                          ║
║    STEP 1 — ROUTING : Splits the dataset into 3 exclusive subgroups :        ║
║       • Comedy Expert  : niche == 'Comedy'                                   ║
║       • Spoken Expert  : niche != 'Comedy' AND has_speech == 1               ║
║       • Silent Expert  : niche != 'Comedy' AND has_speech == 0               ║
║                                                                              ║
║    STEP 2 — EXPERT PIPELINE (generic) :                                      ║
║       A. Local Feature Selection (Chronological KFold, optimal K)            ║
║       B. 5 level 1 models : XGB, LGBM, CatBoost, RF, KNN                     ║
║       C. Ridge Meta-learner on OOF predictions                               ║
║                                                                              ║
║    STEP 3 — SPECIFIC EXECUTION PER EXPERT :                                  ║
║       • Comedy  : static + momentum + creator profiling (no BERT)            ║
║       • Spoken  : ALL features incl. BERT-SVD + NLP                          ║
║       • Silent  : static + momentum + CV PIL (no BERT)                       ║
║                                                                              ║
║    STEP 4 — GLOBAL ASSEMBLY :                                                ║
║       Concatenation, reordering, global R² and MAE                           ║
║                                                                              ║
║  Hyperparameters : reused from v8 / v12                                      ║
║  Feature Engineering : Momentum, Target Encoding, NLP, CV — v8/v12           ║
║  Record to beat : R² = 0.3490 (spoken) | R² = 0.4036 (Comedy)                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import ast
import re
import io
import os
import time
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from collections import Counter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
TARGET            = "target_log"
RECORD_R2_GLOBAL  = 0.3310
RECORD_MAE        = 0.2907
N_OOF_FOLDS       = 5
K_CANDIDATES = list(range(10, 61))  # Tests each value : 10, 11, 12, ..., 60  # K values to test
N_BERT_SVD        = 20
BERT_MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
BERT_CACHE_PATH   = "bert_embeddings_cache.npy"
CV_CACHE_PATH     = "cv_features_cache.csv"
SAVE_EVERY        = 100
SMOOTH_K          = 5

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — LOADING AND PREPARING THE FULL DATASET
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  STACKING v13 — MIXTURE OF EXPERTS (MoE) ROUTING ARCHITECTURE")
print("=" * 72)

print("\n Loading the dataset...")
df = pd.read_csv("cleaned_data_subtittles.csv")

# Extraction creator_id from TikTok URL
df["creator_id"] = df["webVideoUrl"].str.extract(r'tiktok\.com/@([^/]+)/video')
df["creator_id_int"] = df["creator_id"].astype("category").cat.codes

# Causal sort : essential for momentum + target encoding anti-leakage
df = df.sort_values(["creator_id_int", "video_rank"]).reset_index(drop=True)

print(f" {df['creator_id_int'].nunique()} creators | {len(df):,} videos")

# ═══════════════════════════════════════════════════════════════════════════════
# [PHASE 1] CASCADE CLASSIFICATION — niche (reused v8/v12)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [PHASE 1] CASCADE CLASSIFICATION — 6 Niches")
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

# Pass 1 : P1+P2 without heritage
print("   Pass 1 — P1+P2 Classification...")
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

# ═══════════════════════════════════════════════════════════════════════════════
# [PHASE 2] FULL FEATURE ENGINEERING (v8/v12)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  [PHASE 2] FULL FEATURE ENGINEERING (v8/v12)")
print("=" * 72)

# ── Base features ─────────────────────────────────────────────────────────────
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

# ── Momentum (v8/v12 kept) ────────────────────────────────────────────────────
print("   Momentum...")

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

MOMENTUM_COLS = [
    "momentum_3","trend_slope","consistency","momentum_ratio",
    "trend_direction","volatility_tier","momentum_7","accel",
    "peak_score","recovery","streak_up","momentum_norm",
]
df[MOMENTUM_COLS] = df[MOMENTUM_COLS].fillna(df[MOMENTUM_COLS].median())

# ── NLP Caption (TextBlob) ────────────────────────────────────────────────────
print("   NLP Caption (TextBlob)...")
try:
    from textblob import TextBlob
    def get_sentiment(text):
        try:
            b = TextBlob(str(text))
            return b.sentiment.polarity, b.sentiment.subjectivity
        except Exception:
            return 0.0, 0.0
    sents = df["caption"].fillna("").apply(get_sentiment)
    df["sentiment_polarity"]     = sents.apply(lambda x: x[0])
    df["sentiment_subjectivity"] = sents.apply(lambda x: x[1])
    df["emotional_intensity"]    = df["sentiment_polarity"].abs()
    df["word_count_caption"]     = df["caption"].fillna("").apply(lambda s: len(s.split()))
    print("      TextBlob OK")
except ImportError:
    print("      TextBlob missing — NLP features set to 0")
    for col in ["sentiment_polarity","sentiment_subjectivity","emotional_intensity","word_count_caption"]:
        df[col] = 0.0

NLP_COLS = ["sentiment_polarity","sentiment_subjectivity","emotional_intensity","word_count_caption"]

# ── Cross-features v8 ─────────────────────────────────────────────────────────
print("   Cross-features v8...")
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

CROSS_COLS = [
    "mom3_x_tier","accel_x_viral","recovery_x_hist","streak_x_engage",
    "peak_x_p90","mom7_x_consist","trend_x_duration","norm_x_tier",
    "intensity_x_momentum","polarity_x_viral",
]
df[CROSS_COLS] = df[CROSS_COLS].replace([np.inf, -np.inf], np.nan).fillna(df[CROSS_COLS].median())

# ── Computer Vision PIL (cache) ───────────────────────────────────────────────
print("\n   Computer Vision Features (PIL cache)...")
t_cv = time.time()

def extract_image_features(url, timeout=5):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("L")
        img_small  = img.resize((64, 64))
        arr        = np.array(img_small, dtype=np.float32)
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
    print(f"      CV cache found : {len(done_idx):,} images | {n - len(done_idx)} remaining")
else:
    cache_df = pd.DataFrame(columns=["img_brightness","img_contrast","img_complexity"])
    cache_df.index.name = "row_idx"
    done_idx = set()
    print(f"      CV from scratch ({n:,} images)")

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
            columns=["img_brightness","img_contrast","img_complexity"]
        )
        batch_df.index.name = "row_idx"
        cache_df = pd.concat([cache_df, batch_df])
        cache_df.to_csv(CV_CACHE_PATH)
        batch = {}
        print(f"      [{len(done_idx)+count+1:>5}/{n}] errors={errors} | {time.time()-t_cv:.0f}s")

cache_df = pd.read_csv(CV_CACHE_PATH, index_col="row_idx").sort_index()
for col in ["img_brightness","img_contrast","img_complexity"]:
    cache_df[col] = cache_df[col].fillna(cache_df[col].median())
df["img_brightness"] = cache_df["img_brightness"].values
df["img_contrast"]   = cache_df["img_contrast"].values
df["img_complexity"] = cache_df["img_complexity"].values
CV_COLS = ["img_brightness","img_contrast","img_complexity"]
print(f"   CV features ready in {time.time()-t_cv:.0f}s")

# Cross CV
df["contrast_x_momentum"] = df["img_contrast"]   * df["momentum_3"]
df["bright_x_viral"]      = df["img_brightness"] * df["viral_potential"]
df["complex_x_tier"]      = df["img_complexity"] * df["follower_tier"]
CV_CROSS_COLS = ["contrast_x_momentum","bright_x_viral","complex_x_tier"]
df[CV_CROSS_COLS] = df[CV_CROSS_COLS].replace([np.inf, -np.inf], np.nan).fillna(df[CV_CROSS_COLS].median())

# ── Speech Features (v12) ─────────────────────────────────────────────────────
print("\n   Speech Features (v12)...")

QUESTION_PAT = re.compile(
    r'\b(why|how|what|when|who|which|where|would|could|should|do you|are you|'
    r'is it|can you|did you|have you|will you)\b|\?', re.IGNORECASE
)
LIST_PAT = re.compile(
    r'\b(top|best|\d+\s*(?:ways?|tips?|steps?|reasons?|things?|mistakes?|hacks?|'
    r'facts?|secrets?|signs?|rules?|habits?|tricks?))\b', re.IGNORECASE
)
URGENCY_PAT  = re.compile(r'\b(stop|wait|listen|attention|alert|breaking|now|today|immediately|never|always|everybody|everyone|nobody)\b', re.IGNORECASE)
PERSONAL_PAT = re.compile(r'\b(i |my |me |i\'m|i\'ve|i\'ll|i\'d|we |our |us )\b', re.IGNORECASE)
POI_PAT      = re.compile(r'\b(pov|tuto|tutorial|recipe|hack|trick|tip|secret|reveal|review|react|explain|show|teach|learn|watch)\b', re.IGNORECASE)
POSITIVE_PAT = re.compile(r'\b(amazing|incredible|perfect|great|excellent|best|love|awesome|beautiful|fantastic|brilliant|outstanding|wonderful|superb)\b', re.IGNORECASE)
NEGATIVE_PAT = re.compile(r'\b(horrible|worst|bad|terrible|awful|dangerous|warning|problem|fail|wrong|never|stupid|mistake|error)\b', re.IGNORECASE)

df["subtitles_clean"] = df["subtitles"].fillna("").str.strip()
df["has_speech"]      = (df["subtitles_clean"] != "").astype(int)

def hook_words(text, n=5):
    return " ".join(text.split()[:n]) if text else ""

df["hook_text"]         = df["subtitles_clean"].apply(hook_words)
df["word_count_speech"] = df["subtitles_clean"].apply(lambda t: len(t.split()) if t else 0)
df["char_count_speech"] = df["subtitles_clean"].str.len()
df["speech_rate"]       = df["word_count_speech"] / df["duration"].clip(lower=1)
df["speech_rate"]       = df["speech_rate"].clip(upper=df["speech_rate"].quantile(0.99))

df["is_question"] = df["hook_text"].apply(lambda t: int(bool(QUESTION_PAT.search(t))))
df["is_list"]     = df["hook_text"].apply(lambda t: int(bool(LIST_PAT.search(t))))
df["is_urgency"]  = df["hook_text"].apply(lambda t: int(bool(URGENCY_PAT.search(t))))
df["is_personal"] = df["hook_text"].apply(lambda t: int(bool(PERSONAL_PAT.search(t))))
df["is_poi"]      = df["hook_text"].apply(lambda t: int(bool(POI_PAT.search(t))))

df["hook_score"]      = (df["is_list"]*2.5 + df["is_question"]*2.0 + df["is_urgency"]*1.5
                         + df["is_poi"]*1.5 + df["is_personal"]*1.0)
df["positive_count"]  = df["subtitles_clean"].apply(lambda t: len(POSITIVE_PAT.findall(t)))
df["negative_count"]  = df["subtitles_clean"].apply(lambda t: len(NEGATIVE_PAT.findall(t)))
df["sentiment_ratio"] = (df["positive_count"] - df["negative_count"]) / (df["word_count_speech"].clip(lower=1))

SPEECH_NUMERIC_COLS = [
    "speech_rate","is_question","is_list","is_urgency","is_personal",
    "is_poi","hook_score","positive_count","negative_count","sentiment_ratio",
    "word_count_speech","char_count_speech"
]
df.loc[df["has_speech"] == 0, SPEECH_NUMERIC_COLS] = 0

# Cross-features Speech × Momentum
df["speech_x_momentum"]    = df["has_speech"]      * df["momentum_3"]
df["hook_x_viral"]         = df["hook_score"]       * df["viral_potential"]
df["rate_x_engagement"]    = df["speech_rate"]      * df["engagement_total_hist"]
df["hook_x_tier"]          = df["hook_score"]       * df["follower_tier"]
df["sentiment_x_momentum"] = df["sentiment_ratio"]  * df["momentum_3"]

CROSS_SPEECH_COLS = ["speech_x_momentum","hook_x_viral","rate_x_engagement","hook_x_tier","sentiment_x_momentum"]
df[CROSS_SPEECH_COLS] = df[CROSS_SPEECH_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

SPEECH_FEATURES = SPEECH_NUMERIC_COLS + CROSS_SPEECH_COLS + ["has_speech"]
print(f"    has_speech : {df['has_speech'].sum():,} videos ({df['has_speech'].mean()*100:.1f}%)")

# ── BERT Embeddings + SVD (v12) ───────────────────────────────────────────────
print("\n   BERT Embeddings + SVD...")
df["subtitles_for_bert"] = df["subtitles_clean"].apply(lambda t: t if t else "NON_VERBAL")
texts_for_bert = df["subtitles_for_bert"].tolist()

# Train mask to calibrate SVD
_train_mask_bert = df["video_rank"].between(11, 26)

if os.path.exists(BERT_CACHE_PATH):
    print(f"      BERT cache found : {BERT_CACHE_PATH}")
    bert_embeddings = np.load(BERT_CACHE_PATH)
    if bert_embeddings.shape[0] != len(df):
        print(f"      Incomplete cache ({bert_embeddings.shape[0]} vs {len(df)}) — re-encoding")
        from sentence_transformers import SentenceTransformer
        bert_model      = SentenceTransformer(BERT_MODEL_NAME)
        bert_embeddings = bert_model.encode(texts_for_bert, batch_size=64,
                                             show_progress_bar=True,
                                             convert_to_numpy=True,
                                             normalize_embeddings=True)
        np.save(BERT_CACHE_PATH, bert_embeddings)
    else:
        print(f"      Cache loaded : {bert_embeddings.shape}")
else:
    try:
        from sentence_transformers import SentenceTransformer
        print(f"      Loading BERT : {BERT_MODEL_NAME}...")
        bert_model      = SentenceTransformer(BERT_MODEL_NAME)
        bert_embeddings = bert_model.encode(texts_for_bert, batch_size=64,
                                             show_progress_bar=True,
                                             convert_to_numpy=True,
                                             normalize_embeddings=True)
        np.save(BERT_CACHE_PATH, bert_embeddings)
        print(f"      BERT cache saved → {BERT_CACHE_PATH}")
    except ImportError:
        print("      sentence-transformers missing — BERT replaced by zeros")
        bert_embeddings = np.zeros((len(df), 384))

svd_bert = TruncatedSVD(n_components=N_BERT_SVD, random_state=42)
svd_bert.fit(bert_embeddings[_train_mask_bert])
bert_reduced = svd_bert.transform(bert_embeddings)

explained_var_bert = svd_bert.explained_variance_ratio_.sum()
print(f"    BERT explained variance ({N_BERT_SVD}D SVD) : {explained_var_bert:.1%}")

BERT_COLS = [f"bert_{i}" for i in range(N_BERT_SVD)]
bert_df   = pd.DataFrame(bert_reduced, columns=BERT_COLS, index=df.index)
df        = pd.concat([df, bert_df], axis=1)

# ── Enriched Creator Features (v12) ──────────────────────────────────────────
print("\n   Enriched Creator Features (v12)...")

grp_score = df.groupby("creator_id_int")["explosion_score"]

df["creator_recent_slope"] = (
    grp_score.shift(1).groupby(df["creator_id_int"])
    .transform(lambda s: rolling_slope(s, window=3))
)
rolling_std  = grp_score.shift(1).groupby(df["creator_id_int"]).transform(lambda s: s.rolling(5, min_periods=2).std())
rolling_mean = grp_score.shift(1).groupby(df["creator_id_int"]).transform(lambda s: s.rolling(5, min_periods=2).mean())
df["creator_consistency_score"] = rolling_mean / (rolling_std + 1e-6)

creator_peak   = grp_score.shift(1).groupby(df["creator_id_int"]).transform(lambda s: s.expanding().max())
creator_median = grp_score.shift(1).groupby(df["creator_id_int"]).transform(lambda s: s.expanding().median())
df["creator_peak_ratio"]   = creator_peak / (creator_median + 1e-6)
df["creator_video_count"]  = df.groupby("creator_id_int").cumcount()
df["creator_hist_median"]  = df["hist_median_views"]
df["creator_rank_in_niche"]= df.groupby("niche")["creator_hist_median"].rank(pct=True)

CREATOR_COLS = [
    "creator_recent_slope","creator_consistency_score",
    "creator_peak_ratio","creator_video_count","creator_rank_in_niche",
]
df[CREATOR_COLS] = df[CREATOR_COLS].replace([np.inf, -np.inf], np.nan).fillna(df[CREATOR_COLS].median())

# Cross-features creator
df["creator_mom_x_peak"]      = df["momentum_3"]                * df["creator_peak_ratio"]
df["creator_slope_x_tier"]    = df["creator_recent_slope"]      * df["follower_tier"]
df["creator_consist_x_viral"] = df["creator_consistency_score"] * df["viral_potential"]
df["creator_rank_x_momentum"] = df["creator_rank_in_niche"]     * df["momentum_3"]

CROSS_CREATOR_COLS = [
    "creator_mom_x_peak","creator_slope_x_tier",
    "creator_consist_x_viral","creator_rank_x_momentum",
]
df[CROSS_CREATOR_COLS] = df[CROSS_CREATOR_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"    {len(CREATOR_COLS)} creator features + {len(CROSS_CREATOR_COLS)} cross-creator features")

# ── Chronological Split ───────────────────────────────────────────────────────
TRAIN_RANK_MIN, TRAIN_RANK_MAX = 11, 26
train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

print(f"\n   Split : Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")

# ── Target Encoding OOF Anti-Leakage (v12) ────────────────────────────────────
print("\n   Causal Target Encoding (expanding window, anti-leakage)...")

global_mean = df.loc[train_mask, TARGET].mean()
df = df.sort_values(["creator_id_int", "video_rank"]).reset_index(drop=True)

def expanding_te(group, smooth_k, gm):
    vals = group[TARGET].values
    te   = np.full(len(vals), gm)
    cs, cc = 0.0, 0
    for i in range(len(vals)):
        te[i] = (cs + smooth_k * gm) / (cc + smooth_k)
        cs += vals[i]; cc += 1
    return pd.Series(te, index=group.index)

df["creator_target_mean"] = (
    df.groupby("creator_id_int", group_keys=False)
      .apply(lambda g: expanding_te(g, SMOOTH_K, global_mean))
)

niche_global_means = df.loc[train_mask].groupby("niche")[TARGET].mean().to_dict()

def expanding_te_niche(group, smooth_k, niche_mean):
    vals = group[TARGET].values
    te   = np.full(len(vals), niche_mean)
    cs, cc = 0.0, 0
    for i in range(len(vals)):
        te[i] = (cs + smooth_k * niche_mean) / (cc + smooth_k)
        cs += vals[i]; cc += 1
    return pd.Series(te, index=group.index)

niche_te_list = []
for (cid, niche), grp in df.groupby(["creator_id_int", "niche"]):
    nm = niche_global_means.get(niche, global_mean)
    te_series = expanding_te_niche(grp.sort_values("video_rank"), SMOOTH_K, nm)
    niche_te_list.append(te_series)
df["creator_niche_te"] = pd.concat(niche_te_list).reindex(df.index).fillna(global_mean)

# Re-sync masks after sort
train_mask = df["video_rank"].between(TRAIN_RANK_MIN, TRAIN_RANK_MAX)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

te_corr = np.corrcoef(df.loc[train_mask, "creator_target_mean"], df.loc[train_mask, TARGET])[0, 1]
assert te_corr < 0.999, f" Leakage detected creator_target_mean (corr={te_corr:.4f})"
print(f"   creator_target_mean | corr={te_corr:.4f}")

TE_COLS = ["creator_target_mean", "creator_niche_te"]

# ─────────────────────────────────────────────────────────────────────────────
# DEFINITION OF FEATURE POOLS PER EXPERT
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

# Comedy Expert : static + momentum + creator profiling (NO BERT)
FEATURES_COMEDY = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS +
    NLP_COLS + NICHE_COLS + CREATOR_COLS + CROSS_CREATOR_COLS + TE_COLS
)

# Spoken Expert : ALL features incl. BERT-SVD + speech NLP
FEATURES_SPOKEN = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS +
    NLP_COLS + NICHE_COLS + SPEECH_FEATURES + BERT_COLS +
    CREATOR_COLS + CROSS_CREATOR_COLS + TE_COLS
)

# Silent Expert : static + momentum + CV PIL (NO BERT)
FEATURES_SILENT = (
    STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS +
    NLP_COLS + NICHE_COLS + CV_COLS + CV_CROSS_COLS +
    CREATOR_COLS + CROSS_CREATOR_COLS + TE_COLS
)

print(f"\n   Feature pool — Comedy={len(FEATURES_COMEDY)} | Spoken={len(FEATURES_SPOKEN)} | Silent={len(FEATURES_SILENT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ROUTING : Separation into 3 Experts
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  STEP 1 — ROUTING : Separation into 3 Experts")
print("=" * 72)

# Routing masks are defined from the full dataset
# then intersected with train_mask / test_mask
mask_comedy = df["niche"] == "Comedy"
mask_spoken = (df["niche"] != "Comedy") & (df["has_speech"] == 1)
mask_silent = (df["niche"] != "Comedy") & (df["has_speech"] == 0)

# Exclusivity check
assert (mask_comedy & mask_spoken).sum() == 0, "Overlap Comedy/Spoken"
assert (mask_comedy & mask_silent).sum() == 0, "Overlap Comedy/Silent"
assert (mask_spoken & mask_silent).sum() == 0, "Overlap Spoken/Silent"

for name, mask in [("Comedy", mask_comedy), ("Spoken", mask_spoken), ("Silent", mask_silent)]:
    n_tr = (mask & train_mask).sum()
    n_te = (mask & test_mask).sum()
    pct  = mask.sum() / len(df) * 100
    print(f"  Expert {name:<8} : {mask.sum():>6,} videos ({pct:4.1f}%) | train={n_tr:>4,} | test={n_te:>4,}")

# Global y_test in original order (for final assembly)
y_test_global = df.loc[test_mask, TARGET].copy()
print(f"\n   Routing OK — global y_test : {len(y_test_global)} observations")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXPERT PIPELINE (Generic Function)
# ═══════════════════════════════════════════════════════════════════════════════

def train_expert_pipeline(
    df_full: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    expert_mask: pd.Series,
    max_features_pool: list,
    expert_name: str,
    val_mask: pd.Series,
) -> dict:
    """
    MoE Pipeline for a given expert.

    Parameters
    ----------
    df_full          : Full DataFrame (all videos)
    train_mask       : Boolean mask — chronological train split
    test_mask        : Boolean mask — chronological test split
    expert_mask      : Boolean mask — membership to this expert (routing)
    max_features_pool: List of column names constituting the max pool
    expert_name      : Expert name (for logs)
    val_mask         : Boolean mask — validation split (for CatBoost selector)

    Returns
    --------
    dict with :
        - "test_index"   : pandas index of the test videos of this expert
        - "predictions"  : numpy array of final predictions
        - "k_opt"        : optimal K found
        - "r2_oof_curve" : {k: r2_oof} for plotting
    """
    print(f"\n{'='*72}")
    print(f"   EXPERT : {expert_name.upper()}")
    print(f"{'='*72}")

    # ── Subsets for this expert ────────────────────────────────────────
    train_idx = df_full.index[train_mask & expert_mask]
    val_idx   = df_full.index[val_mask   & expert_mask]
    test_idx  = df_full.index[test_mask  & expert_mask]

    print(f"   Train={len(train_idx):,} | Val={len(val_idx):,} | Test={len(test_idx):,}")
    print(f"   Feature pool : {len(max_features_pool)}")

    if len(train_idx) < 30:
        raise ValueError(f"Expert {expert_name} : too few training data ({len(train_idx)})")
    if len(test_idx) == 0:
        raise ValueError(f"Expert {expert_name} : no test data")

    # Ensure all pool features exist in the df
    available = [f for f in max_features_pool if f in df_full.columns]
    if len(available) < len(max_features_pool):
        missing = set(max_features_pool) - set(available)
        print(f"      {len(missing)} features missing from the pool, ignored : {list(missing)[:5]}...")
    max_features_pool = available

    X_train_full = df_full.loc[train_idx, max_features_pool].astype(float)
    y_train      = df_full.loc[train_idx, TARGET]
    X_val_full   = df_full.loc[val_idx,   max_features_pool].astype(float) if len(val_idx) > 0 else X_train_full.iloc[:5]
    y_val        = df_full.loc[val_idx,   TARGET]                           if len(val_idx) > 0 else y_train.iloc[:5]
    X_test_full  = df_full.loc[test_idx,  max_features_pool].astype(float)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2A — LOCAL FEATURE SELECTION : optimal K via chronological KFold
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n   STEP 2A — Local Feature Selection (Fast CatBoost + Chrono KFold)")

    # 1) Fast CatBoost to get feature ranking
    cat_selector = CatBoostRegressor(
        iterations=2000, learning_rate=0.05, depth=6,
        early_stopping_rounds=80, random_state=42, verbose=0
    )
    cat_selector.fit(X_train_full, y_train, eval_set=(X_val_full, y_val))

    fi_series = pd.Series(
        cat_selector.get_feature_importance(),
        index=max_features_pool
    ).sort_values(ascending=False)

    print(f"   Top 10 features according to fast CatBoost :")
    for rk, (feat, imp) in enumerate(fi_series.head(10).items(), 1):
        print(f"    #{rk:>2}  {feat:<40} {imp:>6.2f}%")

    # 2) Chronological KFold : test each candidate K
    # KFold without shuffle = chronological order (data already sorted by video_rank)
    kf_select = KFold(n_splits=N_OOF_FOLDS, shuffle=False)

    # Lightweight baseline model for validation loop
    ref_model = LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
    )

    r2_oof_curve = {}
    print(f"\n   Searching for optimal K (K ∈ {K_CANDIDATES}) :")
    print(f"  {'K':>4}  {'R²_OOF':>10}  {'Status'}")
    print(f"  {'─'*35}")

    for k in K_CANDIDATES:
        top_k_feats  = fi_series.head(k).index.tolist()
        X_k          = X_train_full[top_k_feats].values
        oof_preds    = np.zeros(len(y_train))

        for fold, (fold_tr_idx, fold_vl_idx) in enumerate(kf_select.split(X_k)):
            ref_model.fit(
                X_k[fold_tr_idx], y_train.values[fold_tr_idx],
            )
            oof_preds[fold_vl_idx] = ref_model.predict(X_k[fold_vl_idx])

        r2_oof = r2_score(y_train.values, oof_preds)
        r2_oof_curve[k] = r2_oof

        marker = "←  best so far" if r2_oof == max(r2_oof_curve.values()) else ""
        print(f"  K={k:>3}  R²_OOF={r2_oof:>8.4f}  {marker}")

    k_opt = max(r2_oof_curve, key=r2_oof_curve.get)
    print(f"\n   K_opt = {k_opt}  (R²_OOF = {r2_oof_curve[k_opt]:.4f})")

    # 3) Permanently reduce datasets to Top K_opt
    selected_features = fi_series.head(k_opt).index.tolist()
    X_train = X_train_full[selected_features]
    X_val   = X_val_full[selected_features]
    X_test  = X_test_full[selected_features]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2B — TRAINING THE 5 BASE MODELS WITH OPTUNA (Level 1)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n    STEP 2B — Optuna Optimization & 5 models (K_opt={k_opt} features)")

    kf_oof = KFold(n_splits=N_OOF_FOLDS, shuffle=False)
    
    # 1. XGBOOST OPTIMIZATION
    def objective_xgb(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0),
            "tree_method": "hist", "verbosity": 0, "random_state": 42
        }
        scores = []
        for tr_idx, vl_idx in kf_oof.split(X_train):
            model = XGBRegressor(**params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[vl_idx])
            scores.append(mean_absolute_error(y_train.iloc[vl_idx], preds))
        return np.mean(scores)

    print(f"     [1/5] Optuna Tuning: XGBoost...")
    study_xgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_xgb.optimize(objective_xgb, n_trials=30)
    xgb_params = study_xgb.best_params
    xgb_params.update({"tree_method": "hist", "verbosity": 0, "random_state": 42})

    # 2. LIGHTGBM OPTIMIZATION
    def objective_lgbm(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "verbose": -1, "random_state": 42
        }
        scores = []
        for tr_idx, vl_idx in kf_oof.split(X_train):
            model = LGBMRegressor(**params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            scores.append(mean_absolute_error(y_train.iloc[vl_idx], model.predict(X_train.iloc[vl_idx])))
        return np.mean(scores)

    print(f"     [2/5] Optuna Tuning: LightGBM...")
    study_lgbm = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_lgbm.optimize(objective_lgbm, n_trials=30)
    lgbm_params = study_lgbm.best_params
    lgbm_params.update({"verbose": -1, "random_state": 42})

    # 3. CATBOOST OPTIMIZATION
    def objective_cat(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "verbose": 0, "random_state": 42
        }
        scores = []
        for tr_idx, vl_idx in kf_oof.split(X_train):
            model = CatBoostRegressor(**params)
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            scores.append(mean_absolute_error(y_train.iloc[vl_idx], model.predict(X_train.iloc[vl_idx])))
        return np.mean(scores)

    print(f"     [3/5] Optuna Tuning: CatBoost...")
    study_cat = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study_cat.optimize(objective_cat, n_trials=30)
    cat_params = study_cat.best_params
    cat_params.update({"verbose": 0, "random_state": 42})

    # 4. RANDOM FOREST & KNN (Robust default params to save time)
    rf_params = dict(n_estimators=500, max_depth=10, min_samples_leaf=4, max_features=0.6, n_jobs=-1, random_state=42)
    
    # ── GENERATING OOF AND TEST PREDICTIONS ──
    oof_xgb, oof_lgbm, oof_cat = np.zeros(len(y_train)), np.zeros(len(y_train)), np.zeros(len(y_train))
    
    print(f"   Generating OOF and Final Fit...")
    # XGBoost
    oof_xgb = cross_val_predict(XGBRegressor(**xgb_params), X_train, y_train, cv=kf_oof)
    xgb_final = XGBRegressor(**xgb_params).fit(X_train, y_train)
    pred_xgb = xgb_final.predict(X_test)
    
    # LightGBM
    oof_lgbm = cross_val_predict(LGBMRegressor(**lgbm_params), X_train, y_train, cv=kf_oof)
    lgbm_final = LGBMRegressor(**lgbm_params).fit(X_train, y_train)
    pred_lgbm = lgbm_final.predict(X_test)
    
    # CatBoost
    oof_cat = cross_val_predict(CatBoostRegressor(**cat_params), X_train, y_train, cv=kf_oof)
    cat_final = CatBoostRegressor(**cat_params).fit(X_train, y_train)
    pred_cat = cat_final.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(**rf_params)
    oof_rf = cross_val_predict(rf_model, X_train, y_train, cv=kf_oof)
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)

    # KNN
    knn_pipeline = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1))
    ])
    oof_knn = cross_val_predict(knn_pipeline, X_train, y_train, cv=kf_oof)
    knn_pipeline.fit(X_train, y_train)
    pred_knn = knn_pipeline.predict(X_test)

    print(f"     XGBoost R²_OOF = {r2_score(y_train, oof_xgb):.4f}")
    print(f"     LightGBM R²_OOF = {r2_score(y_train, oof_lgbm):.4f}")
    print(f"     CatBoost R²_OOF = {r2_score(y_train, oof_cat):.4f}")
    print(f"     RandomForest R²_OOF = {r2_score(y_train, oof_rf):.4f}")
    print(f"     KNN R²_OOF = {r2_score(y_train, oof_knn):.4f}")
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2C — META-LEARNER (Ridge Stacking)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n   STEP 2C — Meta-Learner (Ridge Stacking)")

    # Pure OOF matrix (5 columns = 5 base models)
    meta_train = np.column_stack([oof_xgb, oof_lgbm, oof_cat, oof_rf, oof_knn])
    # Pure test matrix (5 columns = predictions of the 5 models on the test set)
    meta_test  = np.column_stack([pred_xgb, pred_lgbm, pred_cat, pred_rf, pred_knn])

    # Ridge meta-learner trained DIRECTLY on pure OOF (without statistics)
    ridge_meta = Ridge(alpha=10.0)
    ridge_meta.fit(meta_train, y_train)

    final_predictions = ridge_meta.predict(meta_test)

    # Log updated meta-learner coefficients (only 5 models)
    model_names = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "KNN"]
    print(f"   Ridge meta-learner — coefficients :")
    for name_m, coef in zip(model_names, ridge_meta.coef_):
        print(f"    {name_m:<15} : {coef:>+8.4f}")

    # Calculate meta-learner OOF R2 (essential for logs)
    r2_meta_oof = r2_score(y_train, ridge_meta.predict(meta_train))

    print(f"\n   Expert {expert_name} completed")
    print(f"      Meta R²_OOF  = {r2_meta_oof:.4f}")
    print(f"      Test set size = {len(test_idx)}")

    return {
        "test_index":   test_idx,
        "predictions":  final_predictions,
        "k_opt":        k_opt,
        "r2_oof_curve": r2_oof_curve,
        "r2_oof_meta":  r2_meta_oof,
        "expert_name":  expert_name,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SPECIFIC EXECUTION OF THE 3 EXPERTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  STEP 3 — EXECUTION OF THE 3 EXPERTS")
print("=" * 72)

expert_results = {}

# ── Comedy Expert ─────────────────────────────────────────────────────────────
# Features : static + momentum + creator profiling   (NO BERT)
print("\n   Launching COMEDY Expert...")
expert_results["Comedy"] = train_expert_pipeline(
    df_full          = df,
    train_mask       = train_mask,
    test_mask        = test_mask,
    expert_mask      = mask_comedy,
    max_features_pool= FEATURES_COMEDY,
    expert_name      = "Comedy",
    val_mask         = val_mask,
)

# ── Spoken Expert ─────────────────────────────────────────────────────────────
# Features : ALL, including BERT-SVD + speech NLP
print("\n   Launching SPOKEN Expert...")
expert_results["Spoken"] = train_expert_pipeline(
    df_full          = df,
    train_mask       = train_mask,
    test_mask        = test_mask,
    expert_mask      = mask_spoken,
    max_features_pool= FEATURES_SPOKEN,
    expert_name      = "Spoken",
    val_mask         = val_mask,
)

# ── Silent Expert ─────────────────────────────────────────────────────────────
# Features : static + momentum + CV PIL   (NO BERT)
print("\n   Launching SILENT Expert...")
expert_results["Silent"] = train_expert_pipeline(
    df_full          = df,
    train_mask       = train_mask,
    test_mask        = test_mask,
    expert_mask      = mask_silent,
    max_features_pool= FEATURES_SILENT,
    expert_name      = "Silent",
    val_mask         = val_mask,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GLOBAL ASSEMBLY AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("  STEP 4 — GLOBAL ASSEMBLY AND EVALUATION")
print("=" * 72)

# 1) Concatenate the predictions of the 3 experts into an index→pred dictionary
all_preds_dict = {}
for expert_name, res in expert_results.items():
    for idx, pred in zip(res["test_index"], res["predictions"]):
        all_preds_dict[idx] = pred

# 2) Reorder predictions according to the exact order of y_test_global
#    (which preserves the order of the DataFrame sorted by [creator_id_int, video_rank])
test_indices_ordered = df.index[test_mask].tolist()

y_pred_final = np.array([all_preds_dict[i] for i in test_indices_ordered])
y_true_final = y_test_global.values  # same order as test_indices_ordered

# Full coverage check
n_covered = len(all_preds_dict)
n_expected = test_mask.sum()
if n_covered != n_expected:
    print(f"      Incomplete coverage : {n_covered}/{n_expected} predicted videos")
    # Missing videos → fallback to global mean
    missing_idxs = [i for i in test_indices_ordered if i not in all_preds_dict]
    print(f"      {len(missing_idxs)} videos without assigned expert → using global mean")
    fallback_mean = df.loc[train_mask, TARGET].mean()
    for i in missing_idxs:
        all_preds_dict[i] = fallback_mean
    y_pred_final = np.array([all_preds_dict[i] for i in test_indices_ordered])

# 3) Global metrics
r2_global  = r2_score(y_true_final, y_pred_final)
mae_global = mean_absolute_error(y_true_final, y_pred_final)

print(f"\n  ╔══════════════════════════════════════════╗")
print(f"  ║   GLOBAL RESULTS — MoE v13           ║")
print(f"  ╠══════════════════════════════════════════╣")
print(f"  ║   Global R²  : {r2_global:>8.4f}                ║")
print(f"  ║   Global MAE : {mae_global:>8.4f}                ║")
print(f"  ╠══════════════════════════════════════════╣")
print(f"  ║   Record     : R²={RECORD_R2_GLOBAL:>6.4f} | MAE={RECORD_MAE:>6.4f}  ║")
delta_r2  = r2_global - RECORD_R2_GLOBAL
delta_mae = mae_global - RECORD_MAE
icon_r2   = "  +IMPROVEMENT" if r2_global > RECORD_R2_GLOBAL else "   regression"
print(f"  ║   ΔR²  vs record : {delta_r2:>+8.4f}  {icon_r2}    ║")
print(f"  ║   ΔMAE vs record : {delta_mae:>+8.4f}                ║")
print(f"  ╚══════════════════════════════════════════╝")

# 4) Results per expert
print(f"\n  ── Details per Expert ──")
print(f"  {'Expert':<10} {'K_opt':>6} {'Meta R²_OOF':>14} {'N_test':>8}")
print(f"  {'─'*45}")
for expert_name, res in expert_results.items():
    print(f"  {expert_name:<10} {res['k_opt']:>6}   {res['r2_oof_meta']:>10.4f}   {len(res['test_index']):>8,}")

# R² per expert on the actual test set
print(f"\n  ── Test R² per Expert (final evaluation) ──")
for expert_name, res in expert_results.items():
    y_true_expert = df.loc[res["test_index"], TARGET].values
    y_pred_expert = res["predictions"]
    r2_e  = r2_score(y_true_expert, y_pred_expert)
    mae_e = mean_absolute_error(y_true_expert, y_pred_expert)
    flag  = "TARGET" if r2_e >= 0.40 else ("PASS" if r2_e > RECORD_R2_GLOBAL else "FAIL")
    print(f"  {flag} {expert_name:<10}  R²={r2_e:.4f} | MAE={mae_e:.4f}  (n_test={len(res['test_index'])})")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Optimal K curves per expert
# ═══════════════════════════════════════════════════════════════════════════════
print("\n Generating visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0d0d0d")
fig.suptitle("MoE v13 — Optimal K Search per Expert (Chronological KFold)",
             color="white", fontsize=14, fontweight="bold")

COLORS = {"Comedy": "#ffe66d", "Spoken": "#4ecdc4", "Silent": "#a8e6cf"}

for ax, (expert_name, res) in zip(axes, expert_results.items()):
    ax.set_facecolor("#1a1a1a")
    ks     = list(res["r2_oof_curve"].keys())
    r2s    = list(res["r2_oof_curve"].values())
    k_opt  = res["k_opt"]
    color  = COLORS[expert_name]

    ax.plot(ks, r2s, marker="o", color=color, linewidth=2, markersize=7, alpha=0.9)
    ax.axvline(k_opt, color="#ff4d6d", linewidth=1.5, linestyle="--",
               label=f"K_opt = {k_opt}")
    ax.scatter([k_opt], [res["r2_oof_curve"][k_opt]], color="#ff4d6d",
               s=120, zorder=5)
    ax.axhline(RECORD_R2_GLOBAL, color="#888", linewidth=1, linestyle=":",
               label=f"Record R²={RECORD_R2_GLOBAL}")
    ax.set_title(f"Expert {expert_name}", color=color, fontweight="bold", fontsize=12)
    ax.set_xlabel("K (number of features)", color="white")
    ax.set_ylabel("OOF R²", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d2d2d")
    ax.yaxis.grid(True, color="#2d2d2d", alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")

plt.tight_layout()
plt.savefig("moe_v13_k_curves.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("   K-curves chart → moe_v13_k_curves.png")
plt.close()

# ─── Final predictions CSV Export ───────────────────────────────────────
results_df = pd.DataFrame({
    "original_index": test_indices_ordered,
    "y_true":         y_true_final,
    "y_pred_moe":     y_pred_final,
    "residual":       y_true_final - y_pred_final,
    "expert":         [
        "Comedy" if mask_comedy[i] else ("Spoken" if mask_spoken[i] else "Silent")
        for i in test_indices_ordered
    ],
})
results_df.to_csv("moe_v13_predictions.csv", index=False)
print("   Predictions exported → moe_v13_predictions.csv")

print("\n Stacking v13 MoE completed.\n")
print(f"   Final Global R² : {r2_global:.4f}  (Δ vs record : {delta_r2:+.4f})")
print(f"   Final Global MAE: {mae_global:.4f}  (Δ vs record : {delta_mae:+.4f})")