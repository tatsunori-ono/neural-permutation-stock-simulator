
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utilities
# -----------------------------
def normalize_ticker(raw: str) -> str:
    """
    Accepts:
      - "6758" -> "6758.T"
      - "6758.T" -> "6758.T"
      - "NVDA" -> "NVDA"
    """
    t = (raw or "").strip().upper()
    if not t:
        return t
    if t.isdigit():
        return f"{t}.T"
    return t


def pretty_name(ticker: str) -> str:
    t = normalize_ticker(ticker)
    if t.endswith(".T") and t[:-2].isdigit():
        return f"{t[:-2]} (TSE)"
    return t


@st.cache_data(ttl=60 * 60)
def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV using yfinance.
    Returns dataframe with columns: Open, High, Low, Close, Volume.
    """
    t = normalize_ticker(ticker)
    df = yf.download(
        t,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # yfinance sometimes returns multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[cols].copy()
    df = df.dropna()
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    return df


def make_features(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """
    Build intraday/overnight log-returns and an EWMA volatility feature.

    r_ov[t] = log(Open_t / Close_{t-1})
    r_id[t] = log(Close_t / Open_t)
    r_cc[t] = r_ov[t] + r_id[t]
    var_ewm[t] = EWM_mean(r_cc^2) with span
    vol_ewm[t] = sqrt(var_ewm[t])
    """
    if df is None or len(df) < 40:
        return pd.DataFrame()

    out = df.copy()
    out["prev_close"] = out["Close"].shift(1)
    out["r_ov"] = np.log(out["Open"] / out["prev_close"])
    out["r_id"] = np.log(out["Close"] / out["Open"])
    out["r_cc"] = out["r_ov"] + out["r_id"]

    alpha = 2.0 / (span + 1.0)
    out["var_ewm"] = out["r_cc"].pow(2).ewm(alpha=alpha, adjust=False).mean()
    out["vol_ewm"] = np.sqrt(out["var_ewm"].clip(lower=1e-12))

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def make_supervised(
    feat_df: pd.DataFrame,
    lookback: int = 60,
    feat_cols: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create supervised sequences:
      X_i = features[t-lookback+1 ... t]  (length lookback)
      y_i = [r_ov[t+1], r_id[t+1]]

    Returns:
      X: (N, lookback, n_features)
      y: (N, 2)
      close_t: close price aligned with y at time t+1 (for convenience)
    """
    if feat_cols is None:
        feat_cols = ["r_ov", "r_id", "vol_ewm"]

    df = feat_df.copy()
    needed = set(feat_cols + ["r_ov", "r_id", "Close"])
    if any(c not in df.columns for c in needed):
        return np.zeros((0, lookback, len(feat_cols))), np.zeros((0, 2)), np.zeros((0,))

    feats = df[feat_cols].values.astype(np.float32)
    r_ov = df["r_ov"].values.astype(np.float32)
    r_id = df["r_id"].values.astype(np.float32)
    close = df["Close"].values.astype(np.float32)

    # We predict t+1 using information up to t.
    X_list = []
    y_list = []
    close_list = []

    # last index that can be used as t is len(df)-2 (because target at t+1)
    for t in range(lookback - 1, len(df) - 1):
        X = feats[t - lookback + 1 : t + 1]
        y = np.array([r_ov[t + 1], r_id[t + 1]], dtype=np.float32)
        X_list.append(X)
        y_list.append(y)
        close_list.append(close[t + 1])

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, lookback, len(feat_cols)), dtype=np.float32)
    y = np.stack(y_list, axis=0) if y_list else np.zeros((0, 2), dtype=np.float32)
    close_next = np.array(close_list, dtype=np.float32)
    return X, y, close_next


def split_train_test(X: np.ndarray, y: np.ndarray, test_size: int) -> Tuple:
    n = len(X)
    if n <= test_size + 10:
        return X, y, X[:0], y[:0]
    split = n - test_size
    return X[:split], y[:split], X[split:], y[split:]


# -----------------------------
# Neural model (bivariate Gaussian head)
# -----------------------------
class LSTMReturnModel(nn.Module):
    def __init__(self, n_features: int = 3, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),  # mu1, mu2, log_s1, log_s2, rho_raw
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        params = self.head(h)
        mu = params[:, 0:2]
        log_sigma = params[:, 2:4]
        rho_raw = params[:, 4]
        return mu, log_sigma, rho_raw


def bivar_gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, rho_raw: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of y under a predicted bivariate normal distribution.
    Uses a correlation parameterization so covariance stays positive definite.
    """
    eps = 1e-6
    s1 = torch.exp(log_sigma[:, 0]).clamp(min=eps)
    s2 = torch.exp(log_sigma[:, 1]).clamp(min=eps)
    rho = torch.tanh(rho_raw) * 0.99

    cov11 = s1 * s1
    cov22 = s2 * s2
    cov12 = rho * s1 * s2

    cov = torch.stack(
        [
            torch.stack([cov11, cov12], dim=-1),
            torch.stack([cov12, cov22], dim=-1),
        ],
        dim=-2,
    )
    # Add jitter for numerical stability
    jitter = 1e-6 * torch.eye(2, device=y.device).unsqueeze(0)
    cov = cov + jitter

    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
    return -dist.log_prob(y).mean()


@dataclass
class TrainConfig:
    lookback: int = 60
    hidden: int = 64
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> Tuple[LSTMReturnModel, Dict[str, List[float]]]:
    """
    Train with a simple holdout validation split (tail of the train set).
    """
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    n_features = X.shape[-1]
    model = LSTMReturnModel(n_features=n_features, hidden=cfg.hidden).to(device)

    # Split train/val within the provided (chronological) training set
    n = len(X)
    val_size = max(64, int(0.1 * n))
    val_size = min(val_size, n // 3) if n >= 3 else 0
    if val_size < 64:
        val_size = min(64, max(0, n - 64))
    if val_size <= 0:
        X_train, y_train, X_val, y_val = X, y, X[:0], y[:0]
    else:
        X_train, y_train = X[:-val_size], y[:-val_size]
        X_val, y_val = X[-val_size:], y[-val_size:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device) if len(X_val) else None
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device) if len(y_val) else None

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_nll": [], "val_nll": []}

    best_val = float("inf")
    best_state = None
    patience = 5
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        # Mini-batch shuffle (within train only)
        idx = torch.randperm(len(X_train_t), device=device)
        Xb = X_train_t[idx]
        yb = y_train_t[idx]

        batch_losses = []
        for i in range(0, len(Xb), cfg.batch_size):
            xb = Xb[i : i + cfg.batch_size]
            ybatch = yb[i : i + cfg.batch_size]

            optimizer.zero_grad(set_to_none=True)
            mu, log_sigma, rho_raw = model(xb)
            loss = bivar_gaussian_nll(mu, log_sigma, rho_raw, ybatch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.detach().item())

        train_nll = float(np.mean(batch_losses)) if batch_losses else float("nan")

        model.eval()
        with torch.no_grad():
            if X_val_t is not None and len(X_val_t) > 0:
                mu, log_sigma, rho_raw = model(X_val_t)
                val_nll = bivar_gaussian_nll(mu, log_sigma, rho_raw, y_val_t).item()
            else:
                val_nll = float("nan")

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        # Early stopping only when validation exists
        if not math.isnan(val_nll):
            if val_nll < best_val - 1e-4:
                best_val = val_nll
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def evaluate_nll(model: LSTMReturnModel, X: np.ndarray, y: np.ndarray, device: str) -> float:
    if len(X) == 0:
        return float("nan")
    dev = torch.device(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y, dtype=torch.float32, device=dev)
    mu, log_sigma, rho_raw = model(X_t)
    nll = bivar_gaussian_nll(mu, log_sigma, rho_raw, y_t).item()
    return float(nll)


@torch.no_grad()
def simulate_neural_paths(
    model: LSTMReturnModel,
    feat_df: pd.DataFrame,
    cfg: TrainConfig,
    horizon_days: int,
    n_sims: int,
    span: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Simulate future close prices by recursively sampling:
      (r_ov, r_id) ~ predicted bivariate normal

    Returns dict:
      - close_paths: (n_sims, horizon_days)
      - ret_cc: (n_sims, horizon_days)
    """
    assert horizon_days >= 1
    assert n_sims >= 1

    device = torch.device(cfg.device)
    model.eval()

    feat_cols = ["r_ov", "r_id", "vol_ewm"]
    hist_np = feat_df[feat_cols].values.astype(np.float32)
    if len(hist_np) < cfg.lookback:
        raise ValueError("Not enough history after feature engineering for the chosen lookback.")

    last_hist = hist_np[-cfg.lookback:]
    last_close = float(feat_df["Close"].iloc[-1])
    last_var = float(feat_df["var_ewm"].iloc[-1])

    # EWMA update coefficient consistent with feature builder
    alpha = 2.0 / (span + 1.0)

    set_seed(cfg.seed)

    hist = torch.tensor(last_hist, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_sims, 1, 1)
    close = torch.full((n_sims,), last_close, dtype=torch.float32, device=device)
    var = torch.full((n_sims,), last_var, dtype=torch.float32, device=device)

    close_paths = torch.zeros((n_sims, horizon_days), dtype=torch.float32, device=device)
    ret_cc = torch.zeros((n_sims, horizon_days), dtype=torch.float32, device=device)

    for k in range(horizon_days):
        mu, log_sigma, rho_raw = model(hist)

        s1 = torch.exp(log_sigma[:, 0]).clamp(min=1e-6)
        s2 = torch.exp(log_sigma[:, 1]).clamp(min=1e-6)
        rho = torch.tanh(rho_raw) * 0.99

        z1 = torch.randn(n_sims, device=device)
        z2 = torch.randn(n_sims, device=device)
        r_ov = mu[:, 0] + s1 * z1
        r_id = mu[:, 1] + s2 * (rho * z1 + torch.sqrt((1.0 - rho * rho).clamp(min=1e-8)) * z2)

        r_cc_k = r_ov + r_id
        close = close * torch.exp(r_cc_k)

        var = (1.0 - alpha) * var + alpha * (r_cc_k * r_cc_k)
        vol = torch.sqrt(var.clamp(min=1e-12))

        # Update history window
        new_row = torch.stack([r_ov, r_id, vol], dim=1).unsqueeze(1)
        hist = torch.cat([hist[:, 1:, :], new_row], dim=1)

        close_paths[:, k] = close
        ret_cc[:, k] = r_cc_k

    return {
        "close_paths": close_paths.detach().cpu().numpy(),
        "ret_cc": ret_cc.detach().cpu().numpy(),
        "last_close": np.array([last_close], dtype=np.float32),
    }


def simulate_permutation_paths(
    feat_df: pd.DataFrame,
    horizon_days: int,
    n_sims: int,
    seed: int = 7,
    independent_overnight_intraday: bool = True,
) -> Dict[str, np.ndarray]:
    """
    A pure 'permutation/bootstrapping' baseline.

    If independent_overnight_intraday=True:
      - sample r_ov from historical r_ov
      - sample r_id from historical r_id (independently)
      - r_cc = r_ov + r_id

    Else:
      - sample paired (r_ov, r_id) from the same day rows.

    Returns:
      - close_paths: (n_sims, horizon_days)
      - ret_cc: (n_sims, horizon_days)
    """
    rng = np.random.default_rng(seed)

    r_ov_hist = feat_df["r_ov"].values
    r_id_hist = feat_df["r_id"].values
    r_cc_hist = feat_df["r_cc"].values
    last_close = float(feat_df["Close"].iloc[-1])

    if len(r_cc_hist) < 100:
        raise ValueError("Not enough history for permutation baseline.")

    if independent_overnight_intraday:
        r_ov = rng.choice(r_ov_hist, size=(n_sims, horizon_days), replace=True)
        r_id = rng.choice(r_id_hist, size=(n_sims, horizon_days), replace=True)
        r_cc = r_ov + r_id
    else:
        idx = rng.choice(len(r_cc_hist), size=(n_sims, horizon_days), replace=True)
        r_cc = r_cc_hist[idx]

    close_paths = last_close * np.exp(np.cumsum(r_cc, axis=1))
    return {"close_paths": close_paths.astype(np.float32), "ret_cc": r_cc.astype(np.float32), "last_close": np.array([last_close], dtype=np.float32)}


def corr_from_history(closes: Dict[str, pd.Series], min_obs: int = 252) -> np.ndarray:
    """
    Compute empirical correlation matrix from aligned log returns.
    """
    df = pd.DataFrame({k: v for k, v in closes.items()}).dropna()
    if len(df) < min_obs:
        df = df.tail(min_obs)
    rets = np.log(df / df.shift(1)).dropna()
    if len(rets) < 30:
        raise ValueError("Not enough overlapping history to estimate correlation.")
    corr = rets.corr().values
    # Stabilize
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    # Make symmetric
    corr = 0.5 * (corr + corr.T)
    # Jitter diagonal
    corr = corr + 1e-6 * np.eye(corr.shape[0])
    return corr


def impose_rank_correlation(sim_returns: np.ndarray, corr: np.ndarray, seed: int = 7) -> np.ndarray:
    """
    Impose an (approximate) target correlation across assets using a rank re-ordering scheme.

    sim_returns: (n_sims, horizon_days, n_assets) independent samples per asset
    corr: (n_assets, n_assets) target correlation matrix

    Returns reordered returns with the same per-asset marginal distribution (per day)
    but approximately matching cross-asset correlation (per day).
    """
    rng = np.random.default_rng(seed)
    n_sims, horizon, n_assets = sim_returns.shape

    # Cholesky for correlated Gaussian ranks
    # If corr isn't PD, add jitter until it is.
    jitter = 1e-6
    for _ in range(10):
        try:
            L = np.linalg.cholesky(corr + jitter * np.eye(n_assets))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        # Fallback: identity
        L = np.eye(n_assets)

    out = np.empty_like(sim_returns)

    for k in range(horizon):
        eps = rng.standard_normal(size=(n_sims, n_assets))
        Z = eps @ L.T  # correlated gaussian
        # For each asset, reorder returns by ranks of Z
        for j in range(n_assets):
            order = np.argsort(Z[:, j])
            sorted_r = np.sort(sim_returns[:, k, j])
            tmp = np.empty(n_sims, dtype=sim_returns.dtype)
            tmp[order] = sorted_r
            out[:, k, j] = tmp

    return out


def future_business_days(last_date: pd.Timestamp, horizon_days: int) -> pd.DatetimeIndex:
    # TSE has holidays, but business-day index is a reasonable approximation for visualization.
    start = last_date + pd.Timedelta(days=1)
    return pd.bdate_range(start=start, periods=horizon_days)


def plot_fan_chart(
    hist_close: pd.Series,
    future_dates: pd.DatetimeIndex,
    sim_close_paths: np.ndarray,
    title: str,
    history_days: int = 252,
) -> plt.Figure:
    """
    Plot a fan chart using simulated close paths.
    """
    hist = hist_close.tail(history_days)

    pct = np.percentile(sim_close_paths, [5, 25, 50, 75, 95], axis=0)
    p5, p25, p50, p75, p95 = pct

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist.index, hist.values, linewidth=1.5)

    ax.plot(future_dates, p50, linewidth=1.5)
    ax.fill_between(future_dates, p25, p75, alpha=0.25)
    ax.fill_between(future_dates, p5, p95, alpha=0.10)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histogram(values: np.ndarray, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(values, bins=40)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def permutation_sanity_check(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
    n_perm: int = 30,
) -> Dict[str, float]:
    """
    'Is there learnable time structure?'
    We compare test NLL of:
      - model trained on real chronological data
      - models trained on a permuted training target (shuffled y) while keeping X intact
    If the real model is much better than permuted models, that's evidence it's not purely noise.
    """
    # Train real
    model_real, _ = train_model(X_train, y_train, cfg)
    real_nll = evaluate_nll(model_real, X_test, y_test, cfg.device)

    perm_nlls = []
    rng = np.random.default_rng(cfg.seed)
    for i in range(n_perm):
        y_perm = y_train.copy()
        rng.shuffle(y_perm, axis=0)
        model_p, _ = train_model(X_train, y_perm, cfg)
        nll_p = evaluate_nll(model_p, X_test, y_test, cfg.device)
        perm_nlls.append(nll_p)

    perm_nlls = np.array(perm_nlls, dtype=float)
    # Lower NLL is better -> p = fraction of permuted runs that are <= real (as good or better)
    p_value = float(np.mean(perm_nlls <= real_nll))

    return {
        "real_test_nll": float(real_nll),
        "perm_mean_test_nll": float(np.mean(perm_nlls)),
        "perm_std_test_nll": float(np.std(perm_nlls)),
        "p_value": p_value,
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Neural Permutation Stock Simulator", layout="wide")
st.title("Neural Permutation Stock Simulator (JP + US)")

st.caption(
    "This is a research dashboard that generates *simulated* future price paths. "
    "It cannot predict the future and is not financial advice."
)

DEFAULT_TICKERS_RAW = [
    "6758", "8306", "8729",  # your holdings (from screenshot)
    "NVDA",
    "5253", "9984", "5401", "9501", "4043", "5032",
    "5802", "9434", "7203", "8035", "8309", "8316",
    "6753", "8136", "8308", "6752", "8604", "7974",
    "6701", "8411", "7011", "6920", "7012", "7267",
    "6501", "6146", "8601", "6857", "9433", "4631",
    "9432", "9605", "9983",
]
DEFAULT_TICKERS = [normalize_ticker(t) for t in DEFAULT_TICKERS_RAW]

with st.sidebar:
    st.header("Data")
    start = st.date_input("Start date", value=pd.to_datetime("2014-01-01"))
    end = st.date_input("End date", value=pd.Timestamp.today().date())
    span = st.slider("EWMA volatility span", min_value=5, max_value=60, value=20, step=1)

    st.divider()
    st.header("Compute")
    device_opt = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.selectbox("Device", options=[device_opt, "cpu"], index=0 if device_opt == "cuda" else 0)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=7, step=1)

    st.divider()
    st.header("Neural model")
    lookback = st.slider("Lookback (days)", min_value=20, max_value=120, value=60, step=5)
    hidden = st.slider("Hidden size", min_value=16, max_value=256, value=64, step=16)
    epochs = st.slider("Training epochs", min_value=5, max_value=80, value=25, step=5)
    batch_size = st.selectbox("Batch size", options=[64, 128, 256, 512, 1024], index=2)
    lr = st.selectbox("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], index=2)
    weight_decay = st.selectbox("Weight decay", options=[0.0, 1e-5, 1e-4, 1e-3], index=2)

    st.divider()
    st.header("Simulation")
    horizon_days = st.slider("Horizon (days)", min_value=5, max_value=252, value=60, step=5)
    n_sims = st.slider("Scenarios", min_value=100, max_value=5000, value=1000, step=100)

cfg = TrainConfig(
    lookback=int(lookback),
    hidden=int(hidden),
    epochs=int(epochs),
    batch_size=int(batch_size),
    lr=float(lr),
    weight_decay=float(weight_decay),
    seed=int(seed),
    device=str(device),
)

tab1, tab2 = st.tabs(["Single ticker", "Portfolio (experimental)"])

# -----------------------------
# Single ticker tab
# -----------------------------
with tab1:
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.subheader("Ticker")
        ticker = st.selectbox("Select", options=DEFAULT_TICKERS, index=DEFAULT_TICKERS.index("6758.T") if "6758.T" in DEFAULT_TICKERS else 0)
        tkr = normalize_ticker(ticker)

        run_perm_check = st.checkbox("Run permutation sanity check (slow)", value=False)
        n_perm = st.slider("Permutation runs", min_value=10, max_value=100, value=30, step=10, disabled=not run_perm_check)

        st.divider()
        run = st.button("Train & simulate", type="primary")

    with colB:
        st.subheader("Baseline")
        baseline_mode = st.selectbox(
            "Permutation baseline",
            options=["Independent overnight/intraday", "Paired daily returns"],
            index=0,
        )
        st.write(
            "A pure permutation/bootstrapping baseline is shown for comparison "
            "(helps sanity-check whether your neural sampler is doing anything non-trivial)."
        )

    if run:
        with st.spinner("Downloading data..."):
            df = load_ohlcv(tkr, str(start), str(end))

        if df.empty:
            st.error(f"No data returned for {tkr}. It may be delisted or unavailable on Yahoo Finance.")
            st.stop()

        feat_df = make_features(df, span=int(span))
        if feat_df.empty or len(feat_df) < (cfg.lookback + 200):
            st.error("Not enough usable history after feature engineering. Try an earlier start date or smaller lookback.")
            st.stop()

        X, y, _ = make_supervised(feat_df, lookback=cfg.lookback, feat_cols=["r_ov", "r_id", "vol_ewm"])

        test_size = min(252, max(60, int(0.2 * len(X))))
        X_train, y_train, X_test, y_test = split_train_test(X, y, test_size=test_size)

        with st.spinner("Training neural model..."):
            model, hist = train_model(X_train, y_train, cfg)

        train_nll = evaluate_nll(model, X_train, y_train, cfg.device)
        test_nll = evaluate_nll(model, X_test, y_test, cfg.device)

        # Simulate neural scenarios
        with st.spinner("Simulating neural paths..."):
            sim = simulate_neural_paths(model, feat_df, cfg, horizon_days=int(horizon_days), n_sims=int(n_sims), span=int(span))
        close_paths = sim["close_paths"]

        # Permutation baseline
        with st.spinner("Simulating permutation baseline..."):
            indep = baseline_mode.startswith("Independent")
            sim_p = simulate_permutation_paths(
                feat_df,
                horizon_days=int(horizon_days),
                n_sims=int(n_sims),
                seed=int(seed),
                independent_overnight_intraday=indep,
            )
        close_paths_p = sim_p["close_paths"]

        # Compute forward dates
        last_date = feat_df.index[-1]
        future_dates = future_business_days(last_date, int(horizon_days))

        # Metrics
        last_close = float(feat_df["Close"].iloc[-1])
        final_prices = close_paths[:, -1]
        final_returns = (final_prices / last_close) - 1.0

        final_prices_p = close_paths_p[:, -1]
        final_returns_p = (final_prices_p / last_close) - 1.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last close", f"{last_close:,.2f}")
        m2.metric("Neural: median 60D return", f"{np.median(final_returns)*100:,.2f}%")
        m3.metric("Neural: P(gain)", f"{np.mean(final_returns>0)*100:,.1f}%")
        m4.metric("Neural: 5% VaR (60D)", f"{np.percentile(final_returns, 5)*100:,.2f}%")

        st.caption(f"Train NLL: {train_nll:.4f} | Test NLL: {test_nll:.4f} | Test window: {len(X_test)} samples")

        # Training curve
        fig_loss, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hist["train_nll"], label="train")
        if any(not math.isnan(v) for v in hist["val_nll"]):
            ax.plot(hist["val_nll"], label="val")
        ax.set_title("Training NLL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig_loss.tight_layout()

        # Fan charts
        fig_fan = plot_fan_chart(df["Close"], future_dates, close_paths, f"{pretty_name(tkr)} — Neural fan chart", history_days=252)
        fig_fan_p = plot_fan_chart(df["Close"], future_dates, close_paths_p, f"{pretty_name(tkr)} — Permutation baseline fan chart", history_days=252)

        # Histograms
        fig_hist = plot_histogram(final_returns * 100.0, "Neural: distribution of horizon returns", "Return (%)")
        fig_hist_p = plot_histogram(final_returns_p * 100.0, "Permutation baseline: distribution of horizon returns", "Return (%)")

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_fan, clear_figure=True)
            st.pyplot(fig_hist, clear_figure=True)
        with c2:
            st.pyplot(fig_fan_p, clear_figure=True)
            st.pyplot(fig_hist_p, clear_figure=True)

        st.pyplot(fig_loss, clear_figure=True)

        # Optional permutation sanity check
        if run_perm_check:
            with st.spinner("Running permutation sanity check (this can take a while)..."):
                res = permutation_sanity_check(X_train, y_train, X_test, y_test, cfg, n_perm=int(n_perm))

            st.subheader("Permutation sanity check")
            st.write(
                "We compare test performance of the real model against models trained on shuffled training targets. "
                "Lower NLL is better. A small p-value suggests the model is learning something beyond chance."
            )
            st.json(res)


# -----------------------------
# Portfolio tab (experimental)
# -----------------------------
with tab2:
    st.subheader("Portfolio scenario simulator (experimental)")

    st.write(
        "This mode trains one neural model per selected ticker, generates scenarios, then "
        "imposes a historical cross-asset correlation structure using rank re-ordering. "
        "It is a research prototype: treat outputs as simulations, not predictions."
    )

    tickers_sel = st.multiselect(
        "Select tickers",
        options=DEFAULT_TICKERS,
        default=[normalize_ticker(x) for x in ["6758", "8306", "NVDA"]],
    )

    if len(tickers_sel) == 0:
        st.info("Select at least 1 ticker.")
        st.stop()

    max_assets = 6
    if len(tickers_sel) > max_assets:
        st.warning(f"Selected {len(tickers_sel)} tickers. For speed, consider <= {max_assets} tickers.")

    # Default positions (based on your screenshot)
    default_pos = {
        "6758.T": 100,
        "8306.T": 30000,
        "8729.T": 100,
        "NVDA": 5000,
    }
    positions = []
    for t in tickers_sel:
        positions.append({"ticker": t, "shares": int(default_pos.get(t, 0))})

    pos_df = pd.DataFrame(positions)

    st.caption("Edit position sizes (shares). Leave 0 if you only want to include the ticker in correlation estimation.")
    pos_df = st.data_editor(pos_df, use_container_width=True, num_rows="fixed")

    convert_usd_to_jpy = st.checkbox("Convert US tickers to JPY using USDJPY (constant latest rate)", value=True)
    run_port = st.button("Run portfolio simulation", type="primary")

    if run_port:
        # Load all data
        with st.spinner("Downloading portfolio data..."):
            ohlcvs = {}
            feat_dfs = {}
            closes_native = {}
            for t in tickers_sel:
                df = load_ohlcv(t, str(start), str(end))
                if df.empty:
                    st.warning(f"No data for {t}. Skipping.")
                    continue
                feat = make_features(df, span=int(span))
                if feat.empty or len(feat) < (cfg.lookback + 200):
                    st.warning(f"Not enough usable history for {t}. Skipping.")
                    continue
                ohlcvs[t] = df
                feat_dfs[t] = feat
                closes_native[t] = df["Close"]

            if len(feat_dfs) == 0:
                st.error("No usable tickers after data checks.")
                st.stop()

        # FX conversion (constant) if requested
        usd_jpy = 1.0
        if convert_usd_to_jpy and any(not normalize_ticker(t).endswith(".T") for t in feat_dfs.keys()):
            with st.spinner("Downloading USDJPY..."):
                fx = load_ohlcv("JPY=X", str(start), str(end))
            if not fx.empty:
                usd_jpy = float(fx["Close"].iloc[-1])
            st.caption(f"Using USDJPY = {usd_jpy:,.3f} (constant)")

        # Train & simulate per asset
        asset_list = list(feat_dfs.keys())
        n_assets = len(asset_list)

        # Build correlation matrix from historical closes (converted to common currency for correlation estimation)
        closes_common = {}
        for t in asset_list:
            c = closes_native[t].copy()
            if convert_usd_to_jpy and not normalize_ticker(t).endswith(".T"):
                c = c * usd_jpy
            closes_common[t] = c

        try:
            corr = corr_from_history(closes_common, min_obs=252)
        except Exception as e:
            st.warning(f"Correlation estimation failed ({e}). Falling back to identity correlation.")
            corr = np.eye(n_assets, dtype=float)

        # Simulate independent returns per asset
        sim_returns = np.zeros((int(n_sims), int(horizon_days), n_assets), dtype=np.float32)
        last_prices_common = np.zeros((n_assets,), dtype=np.float32)

        with st.spinner("Training models & simulating scenarios per ticker..."):
            for j, t in enumerate(asset_list):
                feat = feat_dfs[t]
                X, y, _ = make_supervised(feat, lookback=cfg.lookback, feat_cols=["r_ov", "r_id", "vol_ewm"])
                test_size = min(252, max(60, int(0.2 * len(X))))
                X_train, y_train, X_test, y_test = split_train_test(X, y, test_size=test_size)

                model, _ = train_model(X_train, y_train, cfg)
                sim = simulate_neural_paths(model, feat, cfg, horizon_days=int(horizon_days), n_sims=int(n_sims), span=int(span))
                r_cc = sim["ret_cc"]  # (n_sims, horizon)
                sim_returns[:, :, j] = r_cc

                last_price = float(ohlcvs[t]["Close"].iloc[-1])
                if convert_usd_to_jpy and not normalize_ticker(t).endswith(".T"):
                    last_price *= usd_jpy
                last_prices_common[j] = last_price

        # Impose correlation (rank)
        with st.spinner("Imposing cross-asset correlation (rank re-ordering)..."):
            sim_returns_corr = impose_rank_correlation(sim_returns, corr=corr, seed=int(seed))

        # Build price paths
        # price_path = last_price * exp(cumsum(return))
        prices = last_prices_common.reshape(1, 1, n_assets) * np.exp(np.cumsum(sim_returns_corr, axis=1))

        # Portfolio value
        shares_map = {normalize_ticker(row["ticker"]): float(row["shares"]) for _, row in pos_df.iterrows()}
        shares = np.array([shares_map.get(normalize_ticker(t), 0.0) for t in asset_list], dtype=np.float32)
        port_values = (prices * shares.reshape(1, 1, n_assets)).sum(axis=2)  # (n_sims, horizon)

        # Dates
        any_feat = feat_dfs[asset_list[0]]
        last_date = any_feat.index[-1]
        future_dates = future_business_days(last_date, int(horizon_days))

        # Plot portfolio fan chart
        hist_port = None
        # Build historical portfolio series (approx, using available closes & constant fx)
        df_hist = pd.DataFrame({t: closes_common[t] for t in asset_list}).dropna()
        if len(df_hist) > 30:
            hist_port = (df_hist * shares.reshape(1, -1)).sum(axis=1)

        if hist_port is not None:
            fig_port = plot_fan_chart(hist_port, future_dates, port_values, "Portfolio value — Neural fan chart", history_days=252)
            st.pyplot(fig_port, clear_figure=True)
        else:
            st.info("Not enough overlapping history to plot historical portfolio value. Showing only forward distribution.")

        last_val = float((last_prices_common * shares).sum())
        final_val = port_values[:, -1]
        ret = (final_val / last_val) - 1.0 if last_val > 0 else np.full_like(final_val, np.nan)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio last value (approx)", f"{last_val:,.0f}")
        c2.metric("Median horizon return", f"{np.nanmedian(ret)*100:,.2f}%")
        c3.metric("P(gain)", f"{np.nanmean(ret>0)*100:,.1f}%")
        c4.metric("5% VaR", f"{np.nanpercentile(ret, 5)*100:,.2f}%")

        fig_v = plot_histogram(ret * 100.0, "Portfolio: distribution of horizon returns", "Return (%)")
        st.pyplot(fig_v, clear_figure=True)

        # Show correlation matrix
        corr_df = pd.DataFrame(corr, index=[pretty_name(t) for t in asset_list], columns=[pretty_name(t) for t in asset_list])
        st.subheader("Estimated historical return correlation (for copula ranks)")
        st.dataframe(corr_df, use_container_width=True)
