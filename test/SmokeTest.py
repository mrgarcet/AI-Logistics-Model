'''
The SmokeTest file serves only one purpose and that is to
print the version number of the install libraries this is done
to ensure that all the necessaries libraries were installed.
'''


from pathlib import Path, PurePath
import sys, textwrap
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))                          # ← makes `src.*` importable

mdl = Path("../models/bert_intent_classifier")
if not mdl.exists():
    print("❌  model folder missing:", mdl); raise SystemExit

from src.nlp.infer import predict_intent
tests = [
    "How many units of item 12 do we have?",
    "Add 5 to item 3",
    "Update stock on item 7 to 100",
]
for q in tests:
    print(textwrap.shorten(q, 40), "→", predict_intent(q))