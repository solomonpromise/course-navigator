"""Train the six neural experiments; test reporting is a separate final step."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coursenav.training import train_reference
for name in sys.argv[1:] or ["rnn", "lstm", "gru", "transformer", "frozen", "finetuned"]:
    print('\nTRAINING', name, flush=True)
    result = train_reference(name)
    print(name, 'seconds=', round(result['train_seconds'], 2), flush=True)
