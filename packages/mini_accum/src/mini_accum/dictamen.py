(.venv) (base) rafaelcifuentes@Mac-mini-de-Rafael-3 Bot_BTC % def norm_key(s): 
    return s.replace("_plus","")
base['freeze_key']   = base['freeze'].map(norm_key)
stress['freeze_key'] = stress['freeze'].map(norm_key)
zsh: no matches found: norm_key(s):
zsh: unknown file attribute: _