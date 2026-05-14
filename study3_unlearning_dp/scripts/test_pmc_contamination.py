#!/usr/bin/env python3
"""
Test PMC clinical case reports for memorization in Llama-3.1-8B-Instruct.

Downloads published clinical case reports from PubMed Central (Open Access),
runs completion attacks to measure EMR, and compares against known baselines:
  - Wikipedia medical articles: EMR=0.035 (confirmed memorized)
  - MTSamples:                  EMR=0.026 (not memorized — predictability floor)
  - Synthetic text:             EMR=0.042 (not memorized — random baseline)

If PMC case reports show EMR >> MTSamples and comparable to Wikipedia,
they're a clinical dataset confirmed memorized in pretraining.

Usage (on cluster):
    python test_pmc_contamination.py --download_only   # Step 1: download
    python test_pmc_contamination.py --test_only       # Step 2: run completion attack
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
BASE = "/fs1/shared/model/llm/Llama-3.1-8B-Instruct"
DATA_DIR = f"{REPO}/data/pmc_case_reports"
SPLITS_DIR = f"{DATA_DIR}/splits_pmc_cases"


def download_pmc_case_reports(n_articles=200):
    """Download clinical case reports from PMC Open Access via E-utilities."""
    import urllib.request
    import time

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = Path(DATA_DIR) / "pmc_case_reports.jsonl"

    query = (
        '"case report"[Title] AND '
        '"patient"[Abstract] AND '
        'english[Language] AND '
        'open access[Filter]'
    )
    encoded = urllib.parse.quote(query)

    print(f"Searching PubMed for case reports...")
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pmc&term={encoded}&retmax={n_articles}&sort=relevance&retmode=json"
    )

    with urllib.request.urlopen(search_url) as resp:
        data = json.loads(resp.read())

    ids = data["esearchresult"]["idlist"]
    print(f"Found {len(ids)} PMC IDs")

    articles = []
    batch_size = 20
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        id_str = ",".join(batch)
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pmc&id={id_str}&rettype=xml"
        )

        try:
            with urllib.request.urlopen(fetch_url) as resp:
                xml_data = resp.read().decode("utf-8")
        except Exception as e:
            print(f"  Batch {i//batch_size} failed: {e}")
            time.sleep(2)
            continue

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            root = ET.fromstring(f"<root>{xml_data}</root>")

        for article in root.iter("article"):
            pmcid = ""
            for aid in article.iter("article-id"):
                if aid.get("pub-id-type") == "pmc":
                    pmcid = aid.text or ""
                    break

            title_el = article.find(".//article-title")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            body = article.find(".//body")
            if body is None:
                continue

            paragraphs = []
            for p in body.iter("p"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

            full_text = "\n\n".join(paragraphs)
            if len(full_text) < 500:
                continue

            if len(full_text) > 4000:
                full_text = full_text[:4000]

            articles.append({
                "pmcid": pmcid,
                "title": title,
                "text": full_text,
                "source": "pmc_case_report",
                "n_chars": len(full_text),
            })

        print(f"  Downloaded batch {i//batch_size + 1}/{(len(ids)+batch_size-1)//batch_size}, "
              f"total articles: {len(articles)}")
        time.sleep(0.5)

    with open(out_path, "w") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")

    print(f"\nSaved {len(articles)} case reports to {out_path}")
    return articles


def build_splits(articles):
    """Build HuggingFace DatasetDict from downloaded articles."""
    from datasets import Dataset, DatasetDict

    n = len(articles)
    members = articles[:n]
    nonmembers_texts = [
        {"text": f"The patient presented with symptoms of {i} disease requiring immediate intervention."}
        for i in range(min(50, n // 2))
    ]

    splits = DatasetDict({
        "members": Dataset.from_list([{"text": a["text"], "title": a.get("title", "")} for a in members]),
        "nonmembers": Dataset.from_list(nonmembers_texts),
    })

    splits.save_to_disk(SPLITS_DIR)
    print(f"Saved splits to {SPLITS_DIR}")
    print(f"  members: {len(splits['members'])}")
    print(f"  nonmembers: {len(splits['nonmembers'])}")
    return splits


def run_completion_attack():
    """Run completion attack on PMC case reports."""
    import yaml

    tag = "compl_pmc_case_reports"
    config = {
        "base_model": BASE,
        "splits_path": SPLITS_DIR,
        "output_dir": f"{REPO}/outputs/attacks/{tag}",
        "prefix_ratio": 0.5,
        "max_length": 512,
        "max_examples": 500,
    }

    config_path = f"{REPO}/config/{tag}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    sys.path.insert(0, REPO)
    sys.argv = ["", "--config", config_path]
    from attacks.completion import main
    main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_only", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--n_articles", type=int, default=200)
    args = ap.parse_args()

    if args.test_only:
        print("Running completion attack on existing PMC data...")
        run_completion_attack()
        return

    articles = download_pmc_case_reports(args.n_articles)
    if not articles:
        print("No articles downloaded!")
        return

    build_splits(articles)

    if not args.download_only:
        print("\nRunning completion attack...")
        run_completion_attack()


if __name__ == "__main__":
    main()
