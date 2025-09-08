# -*- coding: utf-8 -*-
"""
allocator_core.py — v0.2 (mínimo funcional)
Reglas por régimen con histeresis, penalización por correlación, volatility targeting,
rebalanceo consciente de costes y techo Kelly. Corazón actúa como filtro/veto.

Dependencias: numpy, pandas
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


# ----------------------------- Utilidades --------------------------------- #

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def normalize_slope(slope: float, scale: float = 80.0) -> float:
    """Lleva la pendiente (p. ej. de EMA200) a [0,1] vía sigmoide."""
    return _clip01(_sigmoid(slope * scale))

def normalize_adx(adx: float, cap: float = 50.0) -> float:
    return _clip01(adx / cap)

def normalize_fg(fg: float) -> float:
    """Fear&Greed: neutral=1, extremos=0."""
    return _clip01(1.0 - abs((fg - 50.0) / 50.0))

def normalize_funding(funding: float, f_max: float = 0.05) -> float:
    return _clip01(1.0 - np.clip(abs(funding) / f_max, 0.0, 1.0))

def rolling_annual_vol(returns: pd.Series, lookback: int = 30) -> float:
    if returns is None or returns.empty:
        return np.nan
    vol = returns.tail(lookback).std(ddof=0)
    if pd.isna(vol):
        return np.nan
    return float(vol * np.sqrt(252))

def safe_corr(df: pd.DataFrame) -> float:
    if df is None or df.shape[0] < 5:
        return 0.0
    c = df.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if c.shape[0] < 2:
        return 0.0
    # max correlación fuera de la diagonal
    np.fill_diagonal(c.values, 0.0)
    return float(np.abs(c.values).max())


# --------------------------- Detectores / Reglas --------------------------- #

@dataclass
class RegimeDetector:
    enter_trend: float = 0.65
    exit_trend: float = 0.55
    enter_range: float = 0.40
    exit_range: float = 0.50
    f_extremo: float = 0.05  # 5% funding extremo
    last_regime: str = "neutral"
    last_score: float = 0.50

    def composite_score(
        self,
        adx_daily: float,
        adx_4h: float,
        ema200_slope_4h: float,
        fear_greed: Optional[float],
        funding_rate: Optional[float],
    ) -> float:
        s_adx = 0.6 * normalize_adx(adx_daily) + 0.4 * normalize_adx(adx_4h, cap=40)
        s_slope = normalize_slope(ema200_slope_4h)
        s_sent = normalize_fg(fear_greed if fear_greed is not None else 50.0)
        s_fund = normalize_funding(funding_rate if funding_rate is not None else 0.0, self.f_extremo)
        R = 0.40 * s_adx + 0.30 * s_slope + 0.20 * s_sent + 0.10 * s_fund
        return _clip01(R)

    def classify_regime(
        self,
        R: float,
        atr_norm: Optional[float] = None,
        bb_width: Optional[float] = None,
        choppy_threshold: float = 0.035,   # ATR% “alto”
        squeeze_threshold: float = 0.02,   # BBWidth “bajo”
    ) -> str:
        # Histeresis trending/ranging
        if self.last_regime != "trending" and R >= self.enter_trend:
            return "trending"
        if self.last_regime == "trending" and R >= self.exit_trend:
            return "trending"

        if self.last_regime != "ranging" and R <= self.enter_range:
            return "ranging"
        if self.last_regime == "ranging" and R <= self.exit_range:
            return "ranging"

        # Choppy / Squeeze opcionales (si hay métricas)
        if atr_norm is not None and atr_norm >= choppy_threshold:
            return "choppy"
        if bb_width is not None and bb_width <= squeeze_threshold:
            return "squeeze"

        return "neutral"

    def update(
        self,
        adx_daily: float,
        adx_4h: float,
        ema200_slope_4h: float,
        fear_greed: Optional[float],
        funding_rate: Optional[float],
        atr_norm: Optional[float] = None,
        bb_width: Optional[float] = None,
    ) -> Tuple[str, float]:
        R = self.composite_score(adx_daily, adx_4h, ema200_slope_4h, fear_greed, funding_rate)
        regime = self.classify_regime(R, atr_norm, bb_width)
        self.last_regime, self.last_score = regime, R
        return regime, R


@dataclass
class BaseWeights:
    """Pesos base por régimen antes de ajustes."""
    w_trending: Tuple[float, float] = (0.60, 0.40)  # (perla, diamante)
    w_ranging:  Tuple[float, float] = (0.00, 0.50)
    w_choppy:   Tuple[float, float] = (0.00, 0.00)
    w_squeeze:  Tuple[float, float] = (0.20, 0.30)
    w_neutral:  Tuple[float, float] = (0.30, 0.30)

    def get(self, regime: str) -> Dict[str, float]:
        mapping = {
            "trending": self.w_trending,
            "ranging":  self.w_ranging,
            "choppy":   self.w_choppy,
            "squeeze":  self.w_squeeze,
            "neutral":  self.w_neutral,
        }
        perla, diamante = mapping.get(regime, self.w_neutral)
        return {"perla": float(perla), "diamante": float(diamante)}


@dataclass
class CorrelationGuard:
    window_bars: int = 6 * 30  # aprox 30 días en 4h
    corr_threshold: float = 0.70
    scale_low: float = 0.7   # escalado si leve
    scale_high: float = 0.5  # escalado si fuerte

    def adjust(self, weights: Dict[str, float], strategy_returns: pd.DataFrame) -> Dict[str, float]:
        tail = strategy_returns.tail(self.window_bars)
        max_corr = safe_corr(tail)
        if max_corr <= 0.5:
            return weights
        factor = self.scale_low if max_corr <= self.corr_threshold else self.scale_high
        return {k: v * factor for k, v in weights.items()}


@dataclass
class VolatilityTarget:
    target_vol: float = 0.15
    lookback_days: int = 30
    clip_min: float = 0.5
    clip_max: float = 2.0

    def scale(self, weights: Dict[str, float], asset_returns: pd.Series) -> Dict[str, float]:
        vol = rolling_annual_vol(asset_returns, lookback=self.lookback_days)
        if pd.isna(vol) or vol <= 0:
            return weights
        scalar = float(np.clip(self.target_vol / vol, self.clip_min, self.clip_max))
        return {k: v * scalar for k, v in weights.items()}


@dataclass
class CostAwareRebalancer:
    fee_bps: float = 0.0002   # 2 bps por nocional movido
    deadband: float = 0.05    # banda de tolerancia por estrategia
    benefit_mult: float = 2.0 # beneficio esperado debe ser > 2x coste

    def should_rebalance(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        portfolio_value: float,
        expected_benefit_hint: float = 0.0,
    ) -> bool:
        turnover = sum(abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in set(current) | set(target))
        tx_cost = turnover * portfolio_value * self.fee_bps
        # umbral mínimo para evitar micro-movimientos
        max_delta = max(abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in set(current) | set(target))
        if max_delta < self.deadband:
            return False
        return (expected_benefit_hint > self.benefit_mult * tx_cost)

    def apply(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        return target  # la decisión binaria se toma en should_rebalance()


@dataclass
class RiskManager:
    max_kelly: float = 0.25
    mdd_cap: float = 0.10  # 10% semanal, ejemplo
    total_risk_cap: float = 1.00

    @staticmethod
    def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_win <= 0 or avg_loss <= 0:
            return 0.0
        k = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0.0, k)

    def apply_kelly_cap(self, weights: Dict[str, float], kelly_by_strat: Dict[str, float]) -> Dict[str, float]:
        capped = {}
        for k, w in weights.items():
            kelly = kelly_by_strat.get(k, self.max_kelly)
            capped[k] = min(w, min(kelly, self.max_kelly))
        # normalizar si hace falta (mantener proporción)
        s = sum(capped.values())
        return {k: (v / s if s > 0 else 0.0) for k, v in capped.items()}

    def punish_drawdown(self, weights: Dict[str, float], weekly_mdd: Dict[str, float]) -> Dict[str, float]:
        out = dict(weights)
        for k, mdd in weekly_mdd.items():
            if mdd is not None and mdd > self.mdd_cap:
                out[k] *= 0.5
        s = sum(out.values())
        return {k: (v / s if s > 0 else 0.0) for k, v in out.items()}

    def cap_total_risk(self, weights: Dict[str, float]) -> Dict[str, float]:
        s = sum(weights.values())
        if s <= self.total_risk_cap:
            return weights
        return {k: v * (self.total_risk_cap / s) for k, v in weights.items()}


# ------------------------------- Núcleo ----------------------------------- #

@dataclass
class AllocatorCore:
    regime: RegimeDetector = field(default_factory=RegimeDetector)
    basew: BaseWeights = field(default_factory=BaseWeights)
    corr: CorrelationGuard = field(default_factory=CorrelationGuard)
    vol: VolatilityTarget = field(default_factory=VolatilityTarget)
    cost: CostAwareRebalancer = field(default_factory=CostAwareRebalancer)
    risk: RiskManager = field(default_factory=RiskManager)

    # Parámetros filtro Corazón
    fg_extreme_hi: float = 85.0
    fg_extreme_lo: float = 15.0
    funding_extreme: float = 0.05
    heart_risk_off: float = 0.25  # multiplicador (0=full stop)

    # Estado
    current_weights: Dict[str, float] = field(default_factory=lambda: {"perla": 0.0, "diamante": 0.0})

    def _apply_heart_filter(
        self,
        weights: Dict[str, float],
        fear_greed: Optional[float],
        funding_rate: Optional[float],
    ) -> Dict[str, float]:
        fg_bad = (fear_greed is not None) and (fear_greed >= self.fg_extreme_hi or fear_greed <= self.fg_extreme_lo)
        fr_bad = (funding_rate is not None) and (abs(funding_rate) >= self.funding_extreme)
        if fg_bad or fr_bad:
            return {k: v * self.heart_risk_off for k, v in weights.items()}
        return weights

    def step(
        self,
        features: Dict[str, float],
        asset_returns_window: pd.Series,
        strategy_returns_window: pd.DataFrame,
        portfolio_value: float = 100_000.0,
        kelly_hint: Optional[Dict[str, Tuple[float, float, float]]] = None,  # {strat: (WR, avg_win, avg_loss)}
        weekly_mdd: Optional[Dict[str, float]] = None,  # {strat: mdd}
        expected_benefit_hint: float = 0.0,
    ) -> Dict[str, float]:
        """
        Devuelve nuevos pesos recomendados (perla, diamante) para la siguiente barra.
        """
        # 1) Régimen + pesos base
        regime, score = self.regime.update(
            adx_daily=features.get("adx_daily", 0.0),
            adx_4h=features.get("adx_4h", 0.0),
            ema200_slope_4h=features.get("ema200_slope_4h", 0.0),
            fear_greed=features.get("fear_greed"),
            funding_rate=features.get("funding_rate"),
            atr_norm=features.get("atr_norm"),
            bb_width=features.get("bb_width"),
        )
        w = self.basew.get(regime)

        # 2) Filtro Corazón (veto/risk-off)
        w = self._apply_heart_filter(w, features.get("fear_greed"), features.get("funding_rate"))

        # 3) Correlación dinámica
        w = self.corr.adjust(w, strategy_returns_window)

        # 4) Volatility Targeting
        w = self.vol.scale(w, asset_returns_window)

        # 5) Kelly modificado (techo)
        if kelly_hint:
            kelly_vals = {k: self.risk.kelly_fraction(*kelly_hint[k]) for k in w.keys() if k in kelly_hint}
            w = self.risk.apply_kelly_cap(w, kelly_vals)

        # 6) Castigo por drawdown semanal
        if weekly_mdd:
            w = self.risk.punish_drawdown(w, weekly_mdd)

        # 7) Cap de riesgo total
        w = self.risk.cap_total_risk(w)

        # 8) Rebalanceo consciente de costes
        if self.cost.should_rebalance(self.current_weights, w, portfolio_value, expected_benefit_hint):
            self.current_weights = self.cost.apply(self.current_weights, w)
        # Si no conviene, mantener pesos actuales
        return dict(self.current_weights)


# ------------------------------ Ejemplo mini ------------------------------- #
if __name__ == "__main__":
    # Ejemplo mínimo de llamada (datos ficticios)
    core = AllocatorCore()
    feats = {
        "adx_daily": 28.0, "adx_4h": 22.0, "ema200_slope_4h": 0.003,
        "fear_greed": 55.0, "funding_rate": 0.002,
        "atr_norm": 0.018, "bb_width": 0.035
    }
    asset_rets = pd.Series(np.random.normal(0, 0.01, 200)) / 100.0
    strat_rets = pd.DataFrame({
        "perla": np.random.normal(0, 0.006, 200),
        "diamante": np.random.normal(0, 0.007, 200)
    })
    kelly_hint = {
        "perla": (0.58, 1.2, 0.8),     # (WR, avg_win, avg_loss) ejemplo
        "diamante": (0.64, 1.1, 0.9),
    }
    weekly_mdd = {"perla": 0.04, "diamante": 0.02}

    new_w = core.step(
        features=feats,
        asset_returns_window=asset_rets,
        strategy_returns_window=strat_rets,
        portfolio_value=100_000.0,
        kelly_hint=kelly_hint,
        weekly_mdd=weekly_mdd,
        expected_benefit_hint=250.0,  # señal de mejora esperada
    )
    print("[Allocator v0.2] Pesos recomendados:", new_w)

#   ¿Quieres que tambien    te    deje    un    tests_allocator_core.ipynb    mínimo(o    script.py) para    validar    2–3    escenarios    de    régimen / correlación    y    ver    la    traza    de    decisiones?