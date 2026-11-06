"""Download the UCI Bike Sharing (hourly) dataset and verify it against an independent copy.

Dataset : Bike Sharing Dataset, hour.csv (17,379 hourly records, Capital Bikeshare, Washington DC, 2011-2012)
Origin  : Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge.
          Progress in Artificial Intelligence. UCI Machine Learning Repository, doi:10.24432/C5W894. License: CC BY 4.0.
Copies  : two independent GitHub copies of the UCI file must match byte for byte.
Run     : python src/download_data.py
"""
from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URLS = [
    "https://raw.githubusercontent.com/muditp19/UCI_Bike-sharing-dataset/master/hour.csv",
    "https://raw.githubusercontent.com/ClaudiaCartaya/Bike-sharing/master/hour.csv",
]
OUT = ROOT / "data" / "hour.csv"


def main() -> None:
    blobs = [urllib.request.urlopen(u, timeout=60).read() for u in URLS]
    digests = [hashlib.sha256(b).hexdigest() for b in blobs]
    assert len(set(digests)) == 1, "the independent copies differ"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(blobs[0])
    rows = blobs[0].count(b"\n") - 1
    assert rows == 17379, f"expected 17,379 rows, got {rows}"
    print(f"Saved {OUT.name}: {rows} rows | sha256 {digests[0][:16]}... | both copies identical")
    (ROOT / "data" / "SOURCE.md").write_text(
        "# Data source\n\n"
        "**Bike Sharing Dataset** (hourly), Capital Bikeshare system, Washington DC, 2011-2012. 17,379 hourly records, 17 columns.\n\n"
        "- Creators: Hadi Fanaee-T and Joao Gama (LIAAD, University of Porto)\n"
        "- Citation: Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. "
        "Progress in Artificial Intelligence. UCI Machine Learning Repository, https://doi.org/10.24432/C5W894\n"
        "- Rental data: Capital Bikeshare public system data; weather: freemeteo.com; holidays: DC government schedule\n"
        "- License: Creative Commons Attribution 4.0 (CC BY 4.0)\n"
        "- UCI page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset\n"
        "- Copies used (verified identical): " + "; ".join(URLS) + "\n"
        f"- SHA-256: {digests[0]}\n\n"
        "The raw file is kept unmodified; cleaning happens in `bikeshare.data`.\n")


if __name__ == "__main__":
    main()
