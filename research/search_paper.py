import sys, re
sys.stdout.reconfigure(encoding='utf-8')
with open('research/base_paper_extracted.txt', encoding='utf-8') as f:
    text = f.read()

keywords = ['update', 'retrain', 'adapt', 'stream', 'online learning', 'real-time', 'batch']
for kw in keywords:
    matches = [m.start() for m in re.finditer(re.escape(kw), text, re.IGNORECASE)]
    print(f'Keyword "{kw}": {len(matches)} occurrences')
    for idx in matches[:4]:
        snippet = text[max(0, idx-100):min(len(text), idx+150)].replace('\n', ' ')
        print(f'   ...{snippet}...')
