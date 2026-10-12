import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hsr_ttr.public_data import prepare_public_cache

rebuilt = prepare_public_cache(ROOT/'outputs/rebuilt_data/beijing_shanghai_6macros.json', ROOT/'data/public_hsr/high-speed trains operation data.csv', force=True)
saved = json.loads((ROOT/'data/processed/beijing_shanghai_6macros.json').read_text(encoding='utf-8'))
assert rebuilt['macros'] == saved['macros']
print('Source reconstruction verified:', [m['train_count'] for m in rebuilt['macros']])
