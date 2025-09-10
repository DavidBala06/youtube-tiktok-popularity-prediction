import pandas as pd
import numpy as np
import re
from datetime import datetime


_EMOJI_REGEX = re.compile(r"[\U0001F300-\U0001FAD6\U0001FAE0-\U0001FAFF]")


def _count_emojis(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(_EMOJI_REGEX.findall(text))


def _uppercase_ratio(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    upp = sum(1 for c in text if c.isupper())
    return upp / len(text)


def _count_upper_words(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return sum(1 for w in text.split() if w.isupper() and len(w) > 1)


def _has_numbers(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return int(bool(re.search(r"\d", text)))


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den_safe = np.where(den == 0, 1.0, den)
    out = num / den_safe
    out = np.where(np.isfinite(out), out, 0.0)
    return out



def preprocess_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
   Transforms the raw dataframe into a matrix of NUMERIC features.
is_training: True if for training, False for prediction
    """
    df = df.copy()

    # ---- 1) Date Features (AVAILABLE AT PREDICTION) ----
    if "Published At" in df.columns:
        dt = pd.to_datetime(df["Published At"], errors="coerce")
        df["year"] = dt.dt.year.fillna(0).astype(int)
        df["month"] = dt.dt.month.fillna(0).astype(int)
        df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        # Video age - calculated differently for training vs prediction
        today = pd.Timestamp.today().normalize()
        if is_training:
            # In training, we use the publication date to calculate the age
            video_age = (today - dt.dt.normalize()).dt.days
        else:
            # When predicting, we assume that the video is new (age 0)
            video_age = pd.Series(0, index=df.index)

        df["video_age_days"] = video_age.fillna(0).clip(lower=0)
    else:
        df["year"] = 0
        df["month"] = 0
        df["dayofweek"] = 0
        df["is_weekend"] = 0
        df["video_age_days"] = 0

    # ---- 2) Title Features (AVAILABLE AT PREDICTION) ----
    if "Title" in df.columns:
        t = df["Title"].fillna("")
        df["title_len_chars"] = t.apply(len)
        df["title_len_words"] = t.apply(lambda x: len(x.split()))
        df["title_exclamations"] = t.apply(lambda x: x.count("!"))
        df["title_questions"] = t.apply(lambda x: x.count("?"))
        df["title_emojis"] = t.apply(_count_emojis)
        df["title_upper_words"] = t.apply(_count_upper_words)
        df["title_uppercase_ratio"] = t.apply(_uppercase_ratio)
        df["title_has_numbers"] = t.apply(_has_numbers)
    else:
        for col in [
            "title_len_chars", "title_len_words", "title_exclamations", "title_questions",
            "title_emojis", "title_upper_words", "title_uppercase_ratio", "title_has_numbers"
        ]:
            df[col] = 0

    # ---- 3) ONLY for training: reports and logs ----
    if is_training:
        # Normalize basic numeric columns
        for col in ["Likes", "Comments", "Dislikes", "Shares"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                df[col] = df[col].clip(lower=0)

        #Useful reports (for training ONLY)
        if "Likes" in df.columns and "Comments" in df.columns:
            df["comments_to_likes_ratio"] = _safe_ratio(df["Comments"].to_numpy(), (df["Likes"] + 1).to_numpy())
        else:
            df["comments_to_likes_ratio"] = 0.0

        # Log for stabilization
        if "Likes" in df.columns:
            df["log_Likes"] = np.log1p(df["Likes"])
        else:
            df["log_Likes"] = 0.0
        if "Comments" in df.columns:
            df["log_Comments"] = np.log1p(df["Comments"])
        else:
            df["log_Comments"] = 0.0
    else:
        # When predicting, we set these features to 0 (not available)
        for col in ["Likes", "Comments", "Dislikes", "Shares",
                    "comments_to_likes_ratio", "log_Likes", "log_Comments"]:
            if col not in df.columns:
                df[col] = 0

    # ---- 4) One-hot per Keyword (if any) ----
    if "Keyword" in df.columns:
        df = pd.get_dummies(df, columns=["Keyword"], prefix="kw")

    # ---- 5) Keep numeric ONLY + inf/NaN cleanup ----
    feats = df.select_dtypes(include=["number"]).copy()

    # Removes the target if it remains in the selectio
    if "Views" in feats.columns:
        feats = feats.drop(columns=["Views"])

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.fillna(0, inplace=True)
    feats = feats.astype("float32")

    return feats
