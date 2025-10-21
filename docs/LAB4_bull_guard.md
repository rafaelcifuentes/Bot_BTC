# LAB4 Bull-guard (cerrado)
- Regla salida base: `close < ema21` (sin next_close)
- Guard: A=d1<SMA200, B=2w<SMA30w (≈200d)
- Ventana validación: 2023-10-01 → 2024-03-01
- Resultados rápidos:
  - exit_like=304, permitidos=4, bloqueados=300 (98.7% bloqueadas)
  - NET_BTC OFF=1.0, ON=1.0 (0.00% lift) en esta ventana
- Ajuste por histórico: `weekly_sma_len=30` por cobertura insuficiente de SMA200w.
