from typing import Dict, Tuple, List, Optional
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

# assume load_and_merge_with_participation and add_personnel_features are imported / defined above
PBP_PATH: str = "Data/pbp_2024_0.csv"
PART_PATH: str = "Data/pbp_participation_2024.csv"

def train_model_with_personnel(df_merged: pd.DataFrame) -> Tuple[RandomForestClassifier, List[str]]:
    # Base situation cols 
    base_cols: List[str] = [
        "down", "ydstogo", "yardline_100", "goal_to_go",
        "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
        "game_half", "score_differential", "wp", "vegas_wp", "ep",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "posteam", "defteam", "posteam_type", "div_game", "shotgun",
    ]
    # Different situation cols
    diff_cols: List[str] = ['down', 'ydstogo', 'yardline_100', 'goal_to_go', 'quarter_seconds_remaining', 
                                                      'half_seconds_remaining', 'game_seconds_remaining', 'score_differential', 'wp', 
                                                      'ep', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'posteam', 'defteam']
    # New numeric personnel features
    personnel_numeric: List[str] = [
        "off_rb", "off_te", "off_wr",
        "def_dl", "def_lb", "def_db",
        "off_empty", "off_heavy", "def_nickel", "def_dime",
    ]

    # Categorical personnel groups (one-hot)
    # Categorical personnel groups (one-hot, using simplified buckets)
    personnel_groups_cat: List[str] = ["off_group_bucket", "def_group_bucket"]

    # Keep rows with a valid label
    df_model: pd.DataFrame = df_merged.dropna(subset=["play_type"]).copy()
    df_model["play_type"] = df_model["play_type"].astype("string")

    # Get rid of faulty columns
    have_cols: List[str] = [c for c in diff_cols + personnel_numeric + personnel_groups_cat if c in df_model.columns]

    # X/y
    X: pd.DataFrame = df_model[have_cols].copy()
    y: pd.Series = df_model["play_type"]

    # Enforce numeric on numeric cols 
    numeric_cols: List[str] = [
    c for c in have_cols
    if c not in [
        "posteam", "defteam", "posteam_type", "game_half",
        "off_group", "def_group",       
        "off_group_bucket", "def_group_bucket",  # NEW: keep as categoricals
    ]
]

    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")

    
    X[numeric_cols] = X[numeric_cols].fillna(0)
    y = y.loc[X.index]


    # One-hot encode categoricals (add our new group cols)
    categorical_cols: List[str] = [
    c for c in [
        "posteam", "defteam", "posteam_type", "game_half",
        "off_group_bucket", "def_group_bucket"   # use buckets, not raw groups
    ]
    if c in X.columns
    ]
    X_enc: pd.DataFrame = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

   
    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y
    )
    if "old_game_id" in df_model.columns:
        groups: pd.Series = df_model["old_game_id"]
    else:
        groups: pd.Series = df_model["game_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_enc, y, groups))

    X_train: pd.DataFrame = X_enc.iloc[train_idx]
    X_test: pd.DataFrame = X_enc.iloc[test_idx]
    y_train: pd.Series = y.iloc[train_idx]
    y_test: pd.Series = y.iloc[test_idx]
    # Fit
    clf: RandomForestClassifier = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Eval
    y_pred = clf.predict(X_test)
    acc: float = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return clf, list(X_train.columns)

_POS_OFF: Tuple[str, ...] = ("RB", "TE", "WR")
_POS_DEF: Tuple[str, ...] = ("DL", "LB", "DB")

_OFF_REGEX: re.Pattern[str] = re.compile(r"(\d+)\s*(RB|TE|WR)", flags=re.IGNORECASE)
_DEF_REGEX: re.Pattern[str] = re.compile(r"(\d+)\s*(DL|LB|DB)", flags=re.IGNORECASE)

_TOKEN_RE: re.Pattern[str] = re.compile(r"(\d+)\s*([A-Z]+)")
_DEF_POS_MAP: Dict[str, str] = {
    # DL
    "DE": "DL", "DT": "DL", "NT": "DL", "EDGE": "DL",
    # LB
    "LB": "LB", "ILB": "LB", "MLB": "LB", "OLB": "LB",
    # DB
    "DB": "DB", "CB": "DB", "FS": "DB", "SS": "DB", "NB": "DB", "SAF": "DB",
}
_MISC_BAD = {"K", "P", "LS", "QB", "RB", "WR", "FB", "TE"}  # ignore ST/offense leakage

def _parse_counts(s: Optional[str], pat: re.Pattern[str], keys: Tuple[str, ...]) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in keys}
    if not isinstance(s, str) or not s:
        return counts
    for num_str, label in pat.findall(s.upper()):
        try:
            counts[label] += int(num_str)
        except ValueError:
            continue
    return counts

def parse_off_personnel(s: Optional[str]) -> Tuple[int, int, int, str]:
    counts: Dict[str, int] = _parse_counts(s, _OFF_REGEX, _POS_OFF)
    rb: int = counts["RB"]
    te: int = counts["TE"]
    wr: int = counts["WR"]
    grp: str = f"{rb}{te}" if (rb or te) else "UNK"
    return rb, te, wr, grp

def parse_def_personnel_simple(s: Optional[str]) -> Tuple[int, int, int, str]:
    counts: Dict[str, int] = _parse_counts(s, _DEF_REGEX, _POS_DEF)
    dl: int = counts["DL"]
    lb: int = counts["LB"]
    db: int = counts["DB"]
    grp: str = f"{dl}-{lb}-{db}" if (dl or lb or db) else "UNK"
    return dl, lb, db, grp

def normalize_def_personnel_to_buckets(s: Optional[str]) -> Optional[str]:
    """
    Convert verbose strings like '3 CB, 2 DE, 2 DT, 1 FS, 1 ILB, 1 MLB, 1 SS'
    into '4 DL, 2 LB, 5 DB'. Returns None if cannot parse.
    """
    if not isinstance(s, str) or not s:
        return None
    parts = _TOKEN_RE.findall(s.upper())
    if not parts:
        return None

    c: Counter[str] = Counter()
    for n_str, pos in parts:
        if pos in _MISC_BAD:
            continue
        bucket: Optional[str] = _DEF_POS_MAP.get(pos)
        if not bucket:
            continue
        try:
            c[bucket] += int(n_str)
        except ValueError:
            continue

    if sum(c.values()) == 0:
        return None

    dl: int = c.get("DL", 0)
    lb: int = c.get("LB", 0)
    db: int = c.get("DB", 0)
    return f"{dl} DL, {lb} LB, {db} DB"

def add_personnel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected (from participation + pbp merge):
      - personnel_off: e.g., '1 RB, 1 TE, 3 WR'
      - personnel_def: either 'X DL, Y LB, Z DB' or raw positions (CB/FS/DE/etc.) we will normalize
    Output adds:
      off_rb, off_te, off_wr, off_group, off_empty, off_heavy
      def_dl, def_lb, def_db, def_group, def_nickel, def_dime
    """
    out: pd.DataFrame = df.copy()

    if "personnel_off" not in out.columns:
        out["personnel_off"] = None
    if "personnel_def" not in out.columns:
        out["personnel_def"] = None

    # Normalize defense if verbose
    needs_norm = out["personnel_def"].astype("string").str.contains(
    r"(?:CB|FS|SS|DE|DT|NT|ILB|MLB|OLB|EDGE|NB|SAF)", na=False, regex=True
)

    out.loc[needs_norm, "personnel_def"] = out.loc[needs_norm, "personnel_def"].map(
        normalize_def_personnel_to_buckets
    )

    # Parse offense & defense buckets
    # Parse offense & defense buckets (EXPAND TUPLES -> COLUMNS)
    off_cols: pd.DataFrame = out["personnel_off"].apply(parse_off_personnel).apply(pd.Series)
    off_cols.columns = ["off_rb", "off_te", "off_wr", "off_group"]

    def_cols: pd.DataFrame = out["personnel_def"].apply(parse_def_personnel_simple).apply(pd.Series)
    def_cols.columns = ["def_dl", "def_lb", "def_db", "def_group"]


    out = pd.concat([out, off_cols, def_cols], axis=1)

    # Derived flags
    out["off_empty"] = (out["off_rb"] == 0).astype(int)
    out["off_heavy"] = ((out["off_te"] >= 2) | (out["off_rb"] >= 2) | (out["off_wr"] <= 2)).astype(int)
    out["def_nickel"] = (out["def_db"] == 5).astype(int)
    out["def_dime"] = (out["def_db"] >= 6).astype(int)

    out["off_group"] = out["off_group"].astype("string")
    out["def_group"] = out["def_group"].astype("string")
    #simplify groups
    out["off_group_bucket"] = out["off_group"].apply(simplify_off_group)
    out["def_group_bucket"] = out["def_group"].apply(simplify_def_group)

    return out
def simplify_off_group(g: str) -> str:
    if g in {"10", "11"}:
        return "spread"
    if g in {"12", "13", "22"}:
        return "heavy"
    if g in {"00"}:
        return "empty"
    return "other"

def simplify_def_group(g: str) -> str:
    if g in {"4-3-4", "3-4-4"}:
        return "base"
    if g in {"4-2-5", "3-3-5"}:
        return "nickel"
    if g in {"2-3-6", "3-2-6", "4-1-6"}:
        return "dime"
    return "other"

def load_and_merge_with_participation(pbp_path: str, part_path: str) -> pd.DataFrame:
    pbp: pd.DataFrame = pd.read_csv(pbp_path, low_memory=False)
    part: pd.DataFrame = pd.read_csv(part_path, low_memory=False)

    part = part.rename(columns={
        "offense_personnel": "personnel_off",
        "defense_personnel": "personnel_def"
    })

    key_game_col: str = "old_game_id" if "old_game_id" in pbp.columns and "old_game_id" in part.columns else "game_id"
    if key_game_col not in pbp.columns or key_game_col not in part.columns:
        raise KeyError("Need shared game key: old_game_id or game_id in both DataFrames.")
    if "play_id" not in pbp.columns or "play_id" not in part.columns:
        raise KeyError("Both DataFrames must contain 'play_id'.")
    merged: pd.DataFrame = pd.merge(
        part,
        pbp,
        on=[key_game_col, "play_id"],
        how="left",
        validate="m:1"
    )

    # Keep only run/pass plays
    merged = merged[merged["play_type"].isin(["run", "pass"])].copy()

    # Add engineered personnel features
    merged = add_personnel_features(merged)
    print(merged.columns)
    return merged

if __name__ == "__main__":
    df_merged: pd.DataFrame = load_and_merge_with_participation(PBP_PATH, PART_PATH)
    print("off buckets:\n", df_merged["off_group_bucket"].value_counts().head(), "\n")
    print("def buckets:\n", df_merged["def_group_bucket"].value_counts().head(), "\n")
    model, feature_columns = train_model_with_personnel(df_merged)

    # (Optional) quick peek at which features made it through
    print(f"\nNum features: {len(feature_columns)}")
    print(feature_columns[:12], "...")
    import numpy as np
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(25), importances[indices[:25]])
    plt.xticks(
        range(25),
        np.array(feature_columns)[indices[:25]],
        rotation=45,
        ha="right"
    )
    plt.title("Top Feature Importances (Run vs Pass)")
    plt.tight_layout()
    plt.show()

    #print all features
    top_features = sorted(
        zip(feature_columns, importances), key=lambda x: x[1], reverse=True
    )
    print("\nOrdered Features:")
    for feat, score in top_features:
        print(f"{feat:<30} {score:.4f}")
