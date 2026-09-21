"""Download public primary papers; preserve URLs, hashes and extraction logs."""
from pathlib import Path
import concurrent.futures, urllib.request, hashlib, json
from pypdf import PdfReader
import pypdfium2
ROOT=Path(__file__).resolve().parents[1]/'literature'
SOURCES={
 'gehre2016': 'https://www.graphics.rwth-aachen.de/media/papers/AdaptingFCNs_lowres.pdf',
 'pietroni2021': 'https://www.quadmesh.cloud/ReliableQuad.pdf',
 'dong2025': 'https://jiaranzhou.github.io/papers/TOG2025.pdf',
 'bommes2009': 'https://graphics.rwth-aachen.de/media/papers/bommes_zimmer_2009_siggraph_011.pdf',
 'ebke2013': 'https://cgg.unibe.ch/media/papers/ebck2013_1.pdf',
}
def fetch(pair):
    name,url=pair;p=ROOT/(name+'.pdf')
    try:
        if p.exists(): raise FileExistsError(p)
        req=urllib.request.Request(url,headers={'User-Agent':'ResearchEvidenceAudit/1.0'})
        data=urllib.request.urlopen(req,timeout=50).read()
        assert data.startswith(b'%PDF'), 'Not a PDF'
        p.write_bytes(data)
        doc=PdfReader(p)
        (ROOT/(name+'.txt')).write_text('\n'.join(f'\n--- PAGE {i+1} ---\n'+page.extract_text() for i,page in enumerate(doc.pages)),encoding='utf-8')
        return dict(id=name,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),pages=len(doc.pages),status='downloaded')
    except Exception as exc:return dict(id=name,url=url,status='failed',error=repr(exc))
ROOT.mkdir(exist_ok=False)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(fetch,SOURCES.items()))
for row in results:
    if row['status']=='downloaded':
        doc=pypdfium2.PdfDocument(ROOT/(row['id']+'.pdf'))
        doc[0].render(scale=1.1).to_pil().save(ROOT/(row['id']+'_page1.png'))
(ROOT/'sources.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
