# ⬇️ Pega aquí el contenido completo del documento EN actualizado ⬇️
# (Mini-Accum KISS – Executive Summary & Reconstruction Blueprint — Updated)
# ... pega TODO el texto ...
Mini‑Accum KISS – Executive Summary & Reconstruction Blueprint (Updated)

1 Overview

Mini‑Accum KISS is a Bitcoin swing‑trading bot designed to maximise BTC accumulation on a spot basis while keeping risk firmly under control.  It trades the BTC/USDC pair on 4‑hour and daily candles and adheres to the principles of Keep It Simple Satoshi (KISS): transparent rules, strict risk controls, modular enhancements and promotion based on evidence.  Unlike many retail bots, it has a single goal: beat buy‑and‑hold (HODL) in net BTC while keeping drawdowns at or below the HODL benchmark.  The system never uses leverage or shorts; trades are spot only.

2 Trading Logic: v1 (DD15_RB1_H30_G200_BULL0)

The current baseline version (v1) has been validated across walk‑forward (WF) and out‑of‑sample (OOS) periods up to mid‑2025.  It uses a simple but robust set of moving average crossovers on 4‑hour candles, a macro filter on daily candles, an ADX filter and a risk guard.  The bot is implemented as a state machine that holds one of two positions:
	•	LONG (in BTC) – the bot owns BTC and holds it for price appreciation.
	•	FLAT (in USDC) – the bot is out of BTC and holds the stable coin.

The trading logic is implemented in a Python module (live_wrapper.py).  A paper version runs hourly as a canary (DRYRUN=1) while the live version (DRYRUN=0, DO_TRADE=1) executes real market orders subject to gating.

2.1 Indicators
Indicator	Period	Source	Purpose
EMA_fast_4h	21	4h	Measures short‑term swing momentum
EMA_slow_4h	55	4h	Measures longer swing momentum
EMA_macro_1d (G200)	200	1D	Macro trend filter (price above EMA_macro)
ADX_macro_1d	14	1D	Trend strength filter (must exceed threshold)
ATR_1d	14	1D	For optional ATR‑based stop/target (post‑v1.1)

2.2 ConditionsPhyton
# input: 4h and 1d OHLC data
ema_fast_4h  = EMA(close_4h, 21)
ema_slow_4h  = EMA(close_4h, 55)
ema_macro_1d = EMA(close_1d, 200)
adx_macro_1d = ADX(high_1d, low_1d, close_1d, 14)

# macro trend filter (G200)
cond_macro_trend = (close_1d[-1] > ema_macro_1d[-1])
# macro trend strength (ADX "sano") – common threshold 20–25
cond_macro_adx   = (adx_macro_1d[-1] > 20)
# swing trend consensus (H30)
cond_swing_trend = (ema_fast_4h[-1] > ema_slow_4h[-1])
# risk guard (DD15) – 15 % stop from entry price
stop_loss_price = entry_price * (1 - 0.15)
2.3 State Machine

Python
# assume current_position ∈ {LONG, FLAT} and entry_price recorded on entry
if current_position == FLAT:
    # Entry: requires macro trend, ADX strength and 4h EMA cross
    if cond_macro_trend and cond_macro_adx and cond_swing_trend:
        SIGNAL = BUY
elif current_position == LONG:
    # Exit: break in macro trend or swing trend or hit stop
    if (not cond_macro_trend) or (not cond_swing_trend) or (close_4h[-1] < stop_loss_price):
        SIGNAL = SELL
When SIGNAL = BUY, the bot submits a buy market order with up to 100 % of the USDC balance (capped by USD_MAX, typically 10 USDT).  When SIGNAL = SELL, it liquidates the entire BTC position.  In DRYRUN mode, orders are logged but not executed.

2.4 Risk & Position Management
	•	DD15 – a hard 15 % drawdown stop; if the position drops 15 % from entry, the bot exits immediately.
	•	RB1 – rebalance 1 % of the portfolio weekly; smooths the curve and reduces costs.
	•	H30 – time‑to‑live; if a trade lasts more than 30 four‑hour candles (≈5 days) without hitting exits, the bot exits.
	•	G200 & BULL0 – macro filter uses a 200‑day EMA; no bullish bias (BULL0).  A bullish bias would delay exits but is disabled in v1.

2.5 Strategy by Cycle (Regimes)

Mini‑Accum KISS operates differently depending on the year of the four‑year Bitcoin halving cycle.  This is a simple, deterministic rule: no complex look‑ahead or discretionary inputs.  The idea is to exploit typical market behaviour in each phase while keeping the bot KISS‑compliant.

2.5.1 E1_Y2 – Tactical strategy for Year +2 (pico/corrección)

When it applies: automatically in the year +2 after a halving (e.g. 2014, 2018, 2022, 2026).  Historically these years coincide with peaks and corrections following the bull run.  The baseline v1 strategy tends to stay flat during these correction years, missing relief rallies.  The E1_Y2 preset aims to capture those rebounds while maintaining low drawdown.

Signals (1‑day bars):

Component	Purpose
EMA12 & EMA26	Faster momentum; react to relief rallies and dead‑cat bounces
RSI (14) bands	Buy when RSI crosses above 35; exit when RSI ≥ 65 (avoid buying into overbought and filter fakeouts)
ADX (14)	Require ADX ≥ 22 to ensure directional strength (only trade impulsive rebounds)
Macro SMA/EMA 200D	Keep the macro filter ON to avoid trading against the regime
dwell = 3	Minimum number of bars between flips (reduces churn; default 3)

The risk controls remain the same as v1 (DD15, RB1, H30).  An optional ATR‑based exit is off by default.

Behaviour: In Year +2 the bot may re‑enter BTC when short‑term momentum (EMA12/26) and macro trend align, and momentum is strong (ADX≥22).  It exits quickly when momentum fades.  The result is few trades, very low drawdown and high net BTC when the market rebounds.  In 2022, using this preset produced sats_mult ≈ 2.9× with mdd_vs_hodl ≈ 0.10 and only ~6 flips, whereas v1 remained almost flat.

2.5.2 KISS v1 TOP – Baseline strategy for other years

When it applies: automatically in the halving year (Y0), Year +1, and Year +3.  These years typically encompass the bull run and the accumulation/range phase.  The baseline v1 (EMA21/55 + macro filter) has proven robust in those regimes.

Signals (1‑day bars): same as described in §2–§2.4: EMA21/55 crossover, 200‑day macro filter, ADX≥22, drawdown gate (DD15), rebalance 1 %, time‑to‑live H30.  The system stays flat when the macro trend is down or trend strength is weak, and re‑enters when the trend resumes.

Behaviour: In the halving year and Year +1 (bull early), v1 enters after confirmation and holds for extended trends; in Year +3 (range), it captures swings within the macro filter without over‑trading.  Results from 2023–2024 (WF) and 2025H1 (OOS) show net BTC gains while maintaining mdd_vs_hodl < 1 with a modest number of flips.

2.5.3 Regime map (summary)

Year of the halving cycle	Preset used	Rationale
Halving year (Y0)	v1 TOP	Entry after confirmation; avoids guessing breakouts
Year +1	v1 TOP	Participate in early bull with controlled risk
Year +2	E1_Y2	Capture relief rallies during correction without high drawdown
Year +3	v1 TOP	Range/accumulation phase – capture swings without over‑trading

Implementation: the bot reads the calendar year and computes years_since_halving = current_year – last_halving_year.  If years_since_halving == 2, it loads the E1_Y2 preset; otherwise it loads the KISS v1 TOP preset.  The switching is fully deterministic and retains the KISS philosophy.

	•	Tactical Preset Year+2: configs/mini_accum/presets/E1_Y2.yaml
(1D: EMA12/26, RSI 35/65, ADX≥22, MA200D ON, dwell=3; DD15 • RB1 • H30; exit ATR OFF; realistic costs )
	•	Baseline Preset  (other years than Year+2): configs/mini_accum/presets/CORE_2025.yaml
(1D: EMA21/55; DD15 • RB1 • H30 • G200 • BULL0; realistic costs)
	•	Regime selector (optional, reproducible):
scripts/mini_accum/dev/run_regime_year.sh
→ Y+2 uses E1_Y2; other years use CORE_2025. Leaves KPIs with suffix OOS__REGIME.

3 System Components & Scripts

3.1 Python Logic (main engine)

The trading logic resides in a Python module (e.g. scripts/mini_accum/live_wrapper.py).  It:
	1.	Reads environment variables (EXCHANGE, DRYRUN, DO_TRADE, USD, USD_MAX, CAP, LOG_LEVEL).
	2.	Uses ccxt to connect to the exchange and fetch balances and OHLCV data (4h and 1D).
	3.	Computes indicators (EMA, ADX, ATR) using pandas or pandas‑ta/ta‑lib.
	4.	Evaluates the state machine described above and the regime logic in §2.5.
	5.	Writes the current signal and state to signals/mini_accum/latest.json.
	6.	If DRYRUN=0 and DO_TRADE=1, executes market orders via ccxt; otherwise prints simulated [PAPER] flip.
	7.	Logs the session to logs/canary_live.<timestamp>.log.

3.2 ZSH Wrappers
	•	bb_day.zsh – hourly canary runner.  Exports environment variables (EXCHANGE=binance, DRYRUN=1, FRESHNESS_MAX_HOURS=8) then calls the Python logic.  It refreshes the signal when stale, writes a log in evidence/dayN_<date>/ and enforces rate‑limit and process locks (file lock and directory lock) to prevent overlapping runs.
	•	bb_dailyreport.zsh – builds REPORT.md summarising daily canary results.  It counts logs for the day, classifies them as GREEN (ready + done), PAUSE or YELLOW, checks whether the day’s attest block in logs/cron.log is ATTEST OK, and renders a Markdown report listing up to 12 valid logs.  Cron runs it at 23:59 UTC; a fallback at 00:10 UTC creates a report for the previous day if missing.
	•	pack_canary.zsh – packages each day’s evidence (REPORT.md, logs, latest.json, mini_accum.status) into artifacts/canary_pack_<date>.tgz.
	•	canary_live.zsh – optional pilot‑live wrapper.  Requires DRYRUN=0 and DO_TRADE=1; runs the main Python script with real orders; used only during pilot tests.
	•	selector_shadow.zsh and lab4_bg_shadow.py – compute friction and LAB4 signals in shadow; they do not influence trading.

3.3 Dependencies

Type	Tool / Library	Purpose
Python	ccxt	Connect to Binance and manage orders
	pandas, pandas‑ta/ta‑lib	Compute EMA, ADX, ATR
	json, datetime, pathlib	Data handling and timestamp parsing
Shell	zsh	Runner scripts (bb_day.zsh, bb_dailyreport.zsh, etc.)
Scheduler	cron	Schedule hourly canary runs and nightly report
OS	macOS/Linux	Cron and file locking

4 Operational Workflow

4.1 Soak Test & Promotion Criteria
	•	Soak Test: run the canary hourly (DRYRUN=1) with guards (rate‑limit + lock) and daily reporting for seven consecutive UTC days.  A day counts only if it has at least one GREEN canary log and ATTEST OK.  Each day’s report and latest.json are packaged.
	•	Status (2025‑10‑28): The soak test restarted on 2025‑10‑27 when guard rails were correctly installed.  The first successful day (Oct 28) counts as Day 1.  To meet the 7‑day recommendation, six more consecutive GREEN days are required.

4.2 Cron Schedule

Time UTC	Script	Purpose
**07 * * * ***	bb_day.zsh	Hourly canary run (DRYRUN=1)
**06 * * * ***	shadow_keepalive.zsh	Refresh stale signals
**09 * * * ***	write_status.zsh	Update mini_accum.status for health check
08:05	runner_cron.sh (daily)	Refresh signals and KPIs via mini_accum runner
08:12	attest.sh (daily)	Run attestation (seal check)
23:59	bb_dailyreport.zsh	Generate daily report for current day
00:10	bb_dailyreport.zsh <yesterday>	Catch‑up if previous day’s report missing

Cron lines are environment‑guarded with MIX_DISABLE=1 etc. to avoid mixture of modules.

4.3 Promotion Gate (Pilot Live)

To promote to a manual Pilot Live session (DRYRUN=0, real orders) you need:
	1.	Streak 7/7: seven consecutive GREEN canary days with ATTEST OK.
	2.	Latest status OK: health/mini_accum.status has age <2h and health=ok.
	3.	No live trades outside quarantine: no canary logs with DRYRUN=0 outside pilot runs.
	4.	Report and pack present: REPORT.md and canary_pack for each day.
	5.	No errors in cron since the last health update.

Once gates are passed, a manual Pilot Live can be run by executing canary_live.zsh with DRYRUN=0, DO_TRADE=1 and minimal USD_MAX (<10 USDT).  This runs the Python logic with real orders; logs are quarantined and evaluated.  The cron remains DRYRUN=1.

5 Roadmap & Modules

5.1 Strategy Versions
Time UTC	Script	Purpose
**07 * * * ***	bb_day.zsh	Hourly canary run (DRYRUN=1)
**06 * * * ***	shadow_keepalive.zsh	Refresh stale signals
**09 * * * ***	write_status.zsh	Update mini_accum.status for health check
08:05	runner_cron.sh (daily)	Refresh signals and KPIs via mini_accum runner
08:12	attest.sh (daily)	Run attestation (seal check)
23:59	bb_dailyreport.zsh	Generate daily report for current day
00:10	bb_dailyreport.zsh <yesterday>	Catch‑up if previous day’s report missing

5.2 Ops Versions
Version OPS	Objective	Components & Status
v2.0‑ops	Baseline deployment & monitoring	Health stable, selector by friction (DRIVE/SPORT), cron health at 09:05, watchdog, NAS backup
v2.1‑ops	Output channels (signals)	Stream signals to CSV/Slack/Telegram opt‑in; degrade gracefully when no env vars
v2.2‑ops	Minimal operation & guardrails	Pre‑flight checks, weekly runbook, automatic rollback if health≠0
v2.3‑ops	LAB4 bull‑guard shadow	Shadow test of new bull guard; promote only if passes KPIs

5.3 Projected ROI (NetBTC)

This roadmap estimates annualised NetBTC ROI ranges based on walk‑forward (WF 2022‑2024) and OOS 2025H1 results.  A conservative “neutral” case anchors to OOS 2025H1 (≈30 % annualised), while a “bull” case extrapolates the strong WF CAGR (≈64 %).  For version names that introduce cycle logic (E1 in Year +2), ranges account for the higher bear‑market edge:

Version	Key modules added	Projected ROI (NetBTC)	Notes
v1.0	Baseline (DD15 • RB1 • H30 • G200 • BULL0)	~29–35 % (neutral), ~60–65 % (bull)	Neutral anchored on OOS 2025H1 ≈30 %; bull anchored on WF22–24 ≈64 %.
v1.1	+ SL/TP defensive (ATR)	~28–33 % (neutral), ~58–65 % (bull)	Defensive stops may reduce upside slightly but improve drawdowns.
KISS‑estacional (new)	E1 in Year +2; v1 in all other years	~35–55 % (neutral), ~70–85 % (bull)	Tactical E1 raises bear‑edge (2022: ≈2.9×) without sacrificing bull performance.
v2	+ hibernation_on_chop + bull_hold	~35–45 % (neutral), ~65–80 % (bull)	Avoids whipsaw and stays in trend; less churn.
v3	+ trailing_exit_bull + pullback_entry	~40–52 % (neutral), ~70–90 % (bull)	Finer timing, marginal but consistent gains.
v4	+ risk sizing by score	~40–55 % (neutral), ~70–95 % (bull)	High potential but dependent on quality of the score.
v5	+ DCA adaptativo + RSI confirmations	~40–55 % (neutral), ~70–95 % (bull)	

6 Reproduction Guide

To rebuild this bot from scratch:
	1.	Clone the repository structure: /scripts/mini_accum contains the ZSH wrappers and Python logic.  /reports/mini_accum holds KPIs and snapshots.  /evidence/dayN_* stores daily logs and reports.
	2.	Install Python ≥3.12 and dependencies (see §3.3).  Use a virtual environment located at .venv and install packages from requirements.txt (ccxt, pandas, pandas‑ta, python‑dotenv, etc.).
	3.	Set environment variables via .env or within scripts: EXCHANGE, DRYRUN, DO_TRADE, USD, USD_MAX, CAP, LOG_LEVEL.  Use testnet keys for DRYRUN (Binance testnet).  For canary runs set DRYRUN=1 and EXCHANGE=binance.
	4.	Schedule cron jobs exactly as listed in §4.2.  Ensure bb_day.zsh runs hourly at minute 07 with rate‑limit and process locks, bb_dailyreport.zsh runs at 23:59 and 00:10, and health scripts run at minutes 6 and 9.
	5.	Run the soak test for 7 UTC days: observe logs in evidence/dayN_*, ensure one GREEN log per day and no rate‑limit violations.  Package each day with pack_canary.zsh.
	6.	Perform gates check (see §4.3) and, only if passing, run a manual pilot live session via canary_live.zsh with minimal capital and DO_TRADE=1.
	7.	Upgrade modules following the roadmap: test v1.1 and v2.0 modules individually on shadow to measure NetBTC lift and MDD reduction before promoting.

7 Conclusion

The Mini‑Accum KISS bot is a disciplined, modular system focused on long‑term BTC accumulation with strict risk controls.  The v1 baseline, tested across multiple years and market conditions, has demonstrated strong performance with low drawdowns.  With the addition of cycle‑based switching (E1_Y2 in Year +2 after halving and KISS v1 TOP otherwise), the bot gains a tactical edge in correction years while maintaining stability in other regimes.  The detailed pseudocode, script descriptions, scheduling, soak test procedure, gates and roadmap provided here allow for deterministic reconstruction and operation.  After completing the soak test and passing the gates, the system can be run in a controlled pilot live session and subsequently upgraded with modules following the roadmap.
# ⬆️ Pega aquí el contenido completo del documento EN actualizado ⬆️
