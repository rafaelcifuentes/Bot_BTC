Yo me habia quedado que este era el ultimo script fijado, verifica si es correcto, y de ahi adelante, con actitud positiva!
# swing_4h_ml_rf.py
# Etapa 3 – Swing Trading 4h con ML (RandomForest) y gestión de riesgos
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import requests
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")


# 1) CARGAR DATOS (4h) O DESCARGAR SI NO EXISTE
# =================================================
def load_data():
    DATA_FILE = "btc_4h.csv"
    if os.path.exists(DATA_FILE):
        try:
            print(f"Cargando datos desde archivo local: {DATA_FILE}")
            df = pd.read_csv(DATA_FILE, index_col=0, header=[0])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if df.empty: raise ValueError("El archivo local está vacío.")
            print("Archivo local cargado y procesado correctamente.")
            return df.dropna().sort_index()
        except Exception as e:
            print(f"Error al leer '{DATA_FILE}': {e}. Se eliminará y descargará de nuevo.")
            os.remove(DATA_FILE)

    print("Descargando 729 días de datos de BTC-USD (4h)...")
    df = yf.download("BTC-USD", interval="4h", period="729d", progress=False)

    # CORRECCIÓN: Aplanar las columnas si yfinance devuelve un MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not required_cols.issubset(df.columns):
        raise ValueError(
            f"Descarga inválida: faltan columnas en el DataFrame. Columnas recibidas: {df.columns.tolist()}")

    df.to_csv(DATA_FILE)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_index()


# 2) AÑADIR FEATURES
# =================================================
def add_features(df):
    print("Añadiendo features...")
    df["EMA12"] = ta.ema(df["Close"], length=12)
    df["EMA26"] = ta.ema(df["Close"], length=26)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]
    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BB_mid"] = bb["BBM_20_2.0"]
    df["BB_upper"] = bb["BBU_20_2.0"]
    df["BB_lower"] = bb["BBL_20_2.0"]
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Vol_SMA10"] = df["Volume"].rolling(10).mean()
    onchain = "onchain_flows_4h.csv"
    if os.path.exists(onchain):
        flows = pd.read_csv(onchain, index_col=0, parse_dates=True)
        flows.index = flows.index.tz_localize(None)
        df["net_flow"] = flows["net_flow"].resample("4H").ffill().reindex(df.index, method="ffill")
    else:
        df["net_flow"] = 0.0
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1500")
        ds = pd.DataFrame(resp.json()["data"])
        ds["timestamp"] = pd.to_datetime(ds["timestamp"].astype(int), unit="s")
        ds.set_index("timestamp", inplace=True)
        ds["value"] = pd.to_numeric(ds["value"], errors="coerce")
        ds.index = ds.index.tz_localize(None)
        df["fng"] = ds["value"].resample("4H").ffill().reindex(df.index,