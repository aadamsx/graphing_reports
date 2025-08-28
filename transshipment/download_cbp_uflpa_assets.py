# download_cbp_uflpa_assets.py  (fixed)
import re, io, sys, requests, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
BASE = "https://www.cbp.gov"
HDRS = {"User-Agent":"Mozilla/5.0"}

UFLPA_STATS = "https://www.cbp.gov/newsroom/stats/trade/uyghur-forced-labor-prevention-act-statistics"
UFLPA_DICT_PAGE = "https://www.cbp.gov/document/stats/uyghur-forced-labor-prevention-act-data-dictionary"
WRO_PAGE = "https://www.cbp.gov/document/stats/withhold-release-orders-findings"
ECOM_PAGE = "https://www.cbp.gov/trade/basic-import-export/e-commerce"

def dl(url, outname=None):
    url = urljoin(BASE, url)
    r = requests.get(url, headers=HDRS, timeout=60); r.raise_for_status()
    if not outname:
        outname = url.split("/")[-1] or "download.bin"
    with open(outname, "wb") as f: f.write(r.content)
    print("saved:", outname, "←", url)
    return outname

def first_pdf_link(page_url):
    r = requests.get(page_url, headers=HDRS, timeout=60); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.select('a[href$=".pdf"]'):
        return urljoin(BASE, a["href"])
    raise SystemExit("No PDF link found on page.")

def latest_csv_link(page_url):
    r = requests.get(page_url, headers=HDRS, timeout=60); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    csvs = [urljoin(BASE, a["href"]) for a in soup.select('a[href$=".csv"]')]
    if not csvs: raise SystemExit("No CSV links found on page.")
    # pick by yyyy-mm in path if present
    def key(u):
        m = re.search(r"/(20\d{2})-(\d{2})/", u)
        return (int(m.group(1)), int(m.group(2))) if m else (0,0)
    return sorted(csvs, key=key, reverse=True)[0]

def deminimis_csv(outname="cbp_deminimis.csv"):
    r = requests.get(ECOM_PAGE, headers=HDRS, timeout=60); r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))  # avoid FutureWarning
    target = None
    for t in tables:
        if t.astype(str).apply(lambda c: c.str.contains("De minimis", case=False, na=False)).any().any():
            target = t; break
    if target is None:
        print("de minimis: table not found"); return None
    target.to_csv(outname, index=False); print("saved:", outname, "←", ECOM_PAGE); return outname

if __name__ == "__main__":
    try:
        pdf_url = first_pdf_link(UFLPA_DICT_PAGE)
        dl(pdf_url)
    except Exception as e:
        print("UFLPA PDF:", e)
    try:
        wro_url = latest_csv_link(WRO_PAGE)
        dl(wro_url)
    except Exception as e:
        print("WRO CSV:", e)
    deminimis_csv()
