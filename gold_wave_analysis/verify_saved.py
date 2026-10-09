"""Execute the read-only audit notebook and compare event summaries to SQL."""
import json
import sqlite3
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
assert Path.cwd()==root, 'Run in gold_wave_analysis'
notebook=json.loads((root/'inspect_gold_waves.ipynb').read_text(encoding='utf8'))
scope={'display':lambda value:None}
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        exec(compile(''.join(cell['source']),cell['id'],'exec'),scope)
with sqlite3.connect((root/'gold_waves.sqlite').as_uri()+'?mode=ro',uri=True) as c:
    query='''SELECT timeframe,impulse_usd,direction,COUNT(*) events,
        SUM(CASE WHEN outcome='continued' THEN 1 ELSE 0 END) continued,
        SUM(CASE WHEN outcome='fully_retraced' THEN 1 ELSE 0 END) fully_retraced
        FROM events GROUP BY timeframe,impulse_usd,direction'''
    actual=pd.read_sql_query(query,c)
    stored=pd.read_sql_query('SELECT timeframe,impulse_usd,direction,events,continued,fully_retraced FROM event_summary WHERE events>0',c)
    key=['timeframe','impulse_usd','direction']
    pd.testing.assert_frame_equal(actual.sort_values(key).reset_index(drop=True),stored.sort_values(key).reset_index(drop=True),check_dtype=False)
    columns={r[1] for r in c.execute('PRAGMA table_info(price_bars)')}
    assert {'timestamp_utc','timeframe','symbol','quality_flag','open','high','low','close'}<=columns
    duplicate=c.execute('SELECT COUNT(*) FROM (SELECT symbol,timeframe,timestamp_utc,COUNT(*) n FROM price_bars GROUP BY 1,2,3 HAVING n>1)').fetchone()[0]
    assert duplicate==0
    print('SQL counts independently match stored summaries; zero duplicate bar keys.')
    print(pd.read_sql_query('SELECT timeframe,usable_rows,missing_ohlc,variable_zero_volume FROM data_quality',c).to_string(index=False))
print('All saved-data and notebook checks passed.')
