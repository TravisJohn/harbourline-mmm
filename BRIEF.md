# Harbourline Bank: MMM data foundations

## The engagement

Harbourline Bank is a mid-sized Australian bank (home loans, credit cards, everyday and savings accounts). Our agency has been brought in as the marketing engineering consultant.

Their Head of Marketing Effectiveness:

> "Our data science team is standing up a marketing mix model in Meridian next quarter. They need two years of clean, weekly data: what we spent and delivered in each channel, and how many applications we got. Right now it's spread across GA4, two Meta ad accounts and DV360 downloads, and nobody trusts any of it. Build us the pipeline that gets it MMM-ready, and tell us what you had to decide along the way."

Paid search is run by another agency. Their data hasn't been shared yet.

## What we've been given

| Source | Location | Notes from the client |
|---|---|---|
| GA4 BigQuery export | `data/raw/ga4/events_YYYYMMDD.parquet` | Web, iOS and Android streams, one file per day |
| Meta Ads Manager exports, main account | `data/raw/meta/au_account/` | Monthly downloads by the social team |
| Meta Ads Manager exports, legacy account | `data/raw/meta/legacy_account/` | "The old account, brand used to run there" |
| DV360 standard reports | `data/raw/dv360/` | Quarterly downloads by the programmatic team |

Period: 30 Sep 2024 to 27 Sep 2026 (104 weeks).

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_data.py
```

`_truth/` holds the model that generated the data. We leave it closed until the end and use it to check what we built.
