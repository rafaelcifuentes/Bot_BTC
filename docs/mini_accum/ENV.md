# ENV (mini_accum)
- python: $(python3 -V | awk '{print $2}')
- lockfile: requirements.txt / poetry.lock
- baseline costs: fee=2 bps/side, slip=1 bps/side
- seeds: ver presets (.seed / seed_btc)

## Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
