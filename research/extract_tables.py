import sys, re
sys.stdout.reconfigure(encoding='utf-8')
with open('research/paper_methodology_text.txt', encoding='utf-8') as f:
    text = f.read()

for t in ['Table 2', 'Table 3', 'Table 4', 'Table 5', 'Table 6', 'Table 7']:
    pos = text.find(t)
    print(f'=== {t} (pos {pos}) ===')
    if pos != -1:
        print(text[pos:pos+1200])
        print('\n' + '='*40 + '\n')
