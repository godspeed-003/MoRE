"""measure_counted_pages.py -- the ACL counted-body page total, measured.

    python code/measure_counted_pages.py [path/to/main.pdf]

ARR limits a long paper to EIGHT pages of CONTENT. Limitations, Ethics,
references and appendices are exempt, so the counted total is where the
Limitations heading starts, not the PDF length.

TWO TRAPS, both of which silently give a wrong answer.

1. MATCH THE HEADING, NOT THE WORD. "Limitations" also appears in running prose
   (the ACL build points at the starred section by name because \ref to a
   \section* prints the previous numbered section's number). A plain text search
   finds that first and reports a body of ~1.1 pages. Filter on font size.

2. A TWO-COLUMN PAGE IS NOT MEASURED BY HEIGHT. A heading 9% down the page is
   9% of the way into whichever COLUMN it sits in. If that is column 2, the page
   is already (1 + 0.09)/2 = 54% used, not 9%. Getting this wrong understated
   every measurement in one session by about half a page -- 8.09 reported against
   8.54 actual -- and made text cuts look more effective than they were.
"""
import fitz,sys
f=sys.argv[1] if len(sys.argv)>1 else 'paper/acl_submission/main.pdf'
d=fitz.open(f)
for i,p in enumerate(d):
    W,H=p.rect.width,p.rect.height
    for blk in p.get_text('dict')['blocks']:
        for ln in blk.get('lines',[]):
            t=''.join(s['text'] for s in ln['spans']).strip()
            if t=='Limitations' and max(s['size'] for s in ln['spans'])>11.5:
                col=0 if ln['bbox'][0]<W/2 else 1
                print(f"COUNTED BODY = {i+(col+ln['bbox'][1]/H)/2:.2f} pages (limit 8)")
                raise SystemExit
print("heading not found")
