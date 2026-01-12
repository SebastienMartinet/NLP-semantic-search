import os
import time
import json
import requests
from tqdm import tqdm

ADS_API_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_TOKEN = os.getenv("ADS_API_TOKEN")

if ADS_TOKEN is None:
    raise RuntimeError("Set ADS_API_TOKEN environment variable")

HEADERS = {
    "Authorization": f"Bearer {ADS_TOKEN}"
}

QUERY = (
    'abs:(nucleosynthesis OR "reaction rates" OR "p-process" OR "r-process" OR "i-process") '
    'AND property:refereed AND year:2005-2025'
)

FIELDS = [
    "bibcode",
    "title",
    "abstract",
    "year",
    "pub",
    "author",
    "citation_count"
]

def fetch_ads(start=0, rows=50, retries=5):
    """Fetch a batch of abstracts with retries on failure."""
    params = {
        "q": QUERY,
        "fl": ",".join(FIELDS),
        "start": start,
        "rows": rows,
        "sort": "citation_count desc"
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(ADS_API_URL, headers=HEADERS, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"[Attempt {attempt}] Failed to fetch batch starting at {start}: {e}")
            time.sleep(1)
    print(f"Skipping batch starting at {start} after {retries} failed attempts.")
    return {"response": {"docs": []}}


def main(max_records=10000, batch_size=50):
    results = []
    start = 0
    pbar = tqdm(total=max_records)

    while len(results) < max_records:
        data = fetch_ads(start=start, rows=batch_size)
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            print("No more documents returned by the API.")
            break

        for d in docs:
            if "abstract" not in d:
                continue

            results.append({
                "bibcode": d.get("bibcode"),
                "title": d.get("title", [""])[0] if isinstance(d.get("title"), list) else d.get("title", ""),
                "abstract": d.get("abstract"),
                "year": d.get("year"),
                "journal": d.get("pub"),
                "authors": d.get("author", []),
                "citations": d.get("citation_count", 0)
            })

            if len(results) >= max_records:
                break
        pbar.update(len(docs))        
        start += batch_size
        time.sleep(0.2)  # polite delay
    pbar.close()
    print(f"Fetched {len(results)} abstracts")

    os.makedirs("data", exist_ok=True)
    with open("data/abstracts.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
