# runner_corazon.py
from runner_profileA_RF_sentiment_EXP import main
if __name__ == "__main__":
    import os
    os.environ.setdefault("RUNNER_NAME", "Corazón")
    main()