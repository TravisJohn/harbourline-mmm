"""
Harbourline Bank - raw data generator (MMM data foundations engagement).

Writes the raw extracts the client has handed over into data/raw/:
  data/raw/ga4/events_YYYYMMDD.parquet          GA4 BigQuery export, one file per day
  data/raw/meta/au_account/*.csv                Meta Ads Manager exports, main AU ad account
  data/raw/meta/legacy_account/*.csv            Meta Ads Manager exports, legacy ad account
  data/raw/dv360/*.csv                          DV360 standard report downloads, one per quarter

Usage:  python generate_data.py          (about 5-10 minutes, deterministic)

The _truth/ folder holds the generating model. Leave it closed until we've built the pipeline.
"""
import hashlib
import json
import math
import os
import shutil
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20261021
rng = np.random.default_rng(SEED)

START = date(2024, 9, 30)   # Monday
END = date(2026, 9, 27)     # Sunday -> 104 full weeks
DAYS = pd.date_range(START, END, freq="D")
N = len(DAYS)

OUT = "data/raw"
TRUTH = "_truth"

SYD_OFFSET = {}  # date -> hours (AEST 10 / AEDT 11)


def syd_offset(d):
    # AEDT from first Sunday in October to first Sunday in April
    def first_sunday(y, m):
        x = date(y, m, 1)
        return x + timedelta(days=(6 - x.weekday()) % 7)
    y = d.year
    if d >= first_sunday(y, 10) or d < first_sunday(y, 4):
        return 11
    return 10


def la_offset(d):
    # PDT from 2nd Sunday March to 1st Sunday November
    def nth_sunday(y, m, n):
        x = date(y, m, 1)
        x = x + timedelta(days=(6 - x.weekday()) % 7)
        return x + timedelta(weeks=n - 1)
    y = d.year
    return -7 if nth_sunday(y, 3, 2) <= d < nth_sunday(y, 11, 1) else -8


# ---------------------------------------------------------------------------
# 1. True world: demand, media plans, response
# ---------------------------------------------------------------------------
t = np.arange(N)
doy = np.array([d.dayofyear for d in DAYS])
dow = np.array([d.dayofweek for d in DAYS])
dates_py = [d.date() for d in DAYS]

# simulated cash rate (hidden driver of home-loan demand)
cash_rate = np.full(N, 4.35)
for d0, r in [(date(2025, 2, 18), 4.10), (date(2025, 5, 20), 3.85), (date(2025, 8, 12), 3.60), (date(2026, 5, 19), 3.85)]:
    cash_rate[np.array(dates_py) >= d0] = r

# seasonality multipliers
spring = 1 + 0.18 * np.exp(-0.5 * ((doy - 290) / 30) ** 2)          # Sep-Nov property season
feb = 1 + 0.10 * np.exp(-0.5 * ((doy - 50) / 15) ** 2)              # back from holidays
eofy = 1 + 0.08 * np.exp(-0.5 * ((doy - 170) / 12) ** 2)            # EOFY
xmas = 1 - 0.35 * np.exp(-0.5 * ((((doy - 360 + 182) % 365) - 182) / 6) ** 2)  # Christmas/New Year dip
dow_mult = np.array([1.12, 1.10, 1.06, 1.03, 0.95, 0.80, 0.86])[dow]
trend = 1 + 0.08 * t / N

promo_hl = np.zeros(N)   # home loan cashback offers
promo_cc = np.zeros(N)   # credit card balance-transfer offers
for a, b in [(date(2024, 10, 14), date(2024, 11, 24)), (date(2025, 3, 3), date(2025, 4, 13)),
             (date(2025, 10, 6), date(2025, 11, 30)), (date(2026, 3, 2), date(2026, 4, 12))]:
    promo_hl[(np.array(dates_py) >= a) & (np.array(dates_py) <= b)] = 1
for a, b in [(date(2025, 1, 13), date(2025, 2, 23)), (date(2025, 7, 7), date(2025, 8, 17)),
             (date(2026, 1, 12), date(2026, 2, 22)), (date(2026, 7, 6), date(2026, 8, 16))]:
    promo_cc[(np.array(dates_py) >= a) & (np.array(dates_py) <= b)] = 1

rate_effect_hl = np.exp(-0.22 * (cash_rate - 4.35))

baseline = {
    "home_loan": 16 * trend * spring * feb * xmas * dow_mult * rate_effect_hl * (1 + 0.20 * promo_hl),
    "credit_card": 26 * trend * feb * eofy * xmas * dow_mult * (1 + 0.25 * promo_cc),
    "savings": 38 * trend * eofy * xmas * dow_mult * (1 + 0.03 * (4.35 - cash_rate) * -1),
}

# unobserved paid search (run by another agency) - tracks demand, plus brand halo added later
search_level = 0.7 + 0.3 * (spring * feb * eofy) / (spring * feb * eofy).mean()

CHANNELS = {
    #  name              weekly decay  half-sat K (adstocked daily AUD)  hill slope  betas per product (submits/day at saturation)
    "meta_brand":        dict(decay=0.70, K=2200, s=1.2, beta={"home_loan": 2.5, "credit_card": 3.5, "savings": 4.0}),
    "meta_prospecting":  dict(decay=0.40, K=3800, s=1.6, beta={"home_loan": 6.5, "credit_card": 12.0, "savings": 10.0}),
    "meta_retargeting":  dict(decay=0.15, K=600,  s=1.0, beta={"home_loan": 2.5, "credit_card": 4.0, "savings": 3.0}),
    "dv360_display":     dict(decay=0.30, K=3000, s=1.3, beta={"home_loan": 1.5, "credit_card": 3.0, "savings": 2.0}),
    "dv360_youtube":     dict(decay=0.55, K=3500, s=1.4, beta={"home_loan": 5.5, "credit_card": 5.0, "savings": 6.0}),
    "dv360_ctv":         dict(decay=0.65, K=5000, s=1.5, beta={"home_loan": 7.0, "credit_card": 2.5, "savings": 3.5}),
}


def flights(n_weeks, on_len=(3, 9), off_len=(1, 4), p_start_on=0.7):
    on = rng.random() < p_start_on
    out = []
    while len(out) < n_weeks:
        L = rng.integers(*(on_len if on else off_len))
        out += [on] * L
        on = not on
    return np.array(out[:n_weeks], dtype=float)


W = N // 7
week_season = (spring * feb * eofy).reshape(W, 7).mean(1)

weekly = {}
weekly["meta_prospecting"] = 24000 * flights(W, (4, 10), (1, 3), 0.9) * week_season * rng.uniform(0.75, 1.25, W)
weekly["meta_brand"] = 14000 * flights(W, (3, 7), (2, 6), 0.6) * rng.uniform(0.7, 1.3, W)
weekly["dv360_display"] = 18000 * flights(W, (3, 8), (1, 5), 0.8) * week_season * rng.uniform(0.7, 1.3, W)
weekly["dv360_youtube"] = 22000 * flights(W, (2, 6), (2, 6), 0.5) * rng.uniform(0.7, 1.3, W)
ctv = 30000 * flights(W, (3, 6), (2, 5), 1.0) * rng.uniform(0.8, 1.2, W)
ctv[: (date(2025, 8, 4) - START).days // 7] = 0          # CTV launched Aug 2025
weekly["dv360_ctv"] = ctv
# retargeting is always-on and scales with site traffic (endogenous)
weekly["meta_retargeting"] = 3500 * week_season * (1 + 0.4 * (weekly["meta_prospecting"] > 0)) * rng.uniform(0.85, 1.15, W)

spend = {}
for c, wk in weekly.items():
    daily = np.repeat(wk / 7, 7) * rng.uniform(0.85, 1.15, N)
    if c.startswith("dv360"):
        daily *= np.array([1.05, 1.05, 1.05, 1.05, 1.0, 0.9, 0.9])[dow]
    spend[c] = np.round(daily, 2)


def adstock(x, weekly_decay):
    d = weekly_decay ** (1 / 7)
    out = np.zeros_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = v + d * acc
        out[i] = acc
    return out * (1 - d)  # normalise so a constant spend s maps to s


contrib = {}
for c, cfg in CHANNELS.items():
    a = adstock(spend[c], cfg["decay"])
    h = a ** cfg["s"] / (a ** cfg["s"] + cfg["K"] ** cfg["s"])
    for p, b in cfg["beta"].items():
        contrib[(c, p)] = b * h * dow_mult

# brand halo onto search (search contribution is real but its spend is not in our data)
brand_ad = adstock(spend["meta_brand"] + spend["dv360_youtube"] + spend["dv360_ctv"], 0.6)
search_contrib = {p: k * search_level * (1 + 0.25 * brand_ad / (brand_ad.mean() + 1)) * dow_mult
                  for p, k in {"home_loan": 3.0, "credit_card": 4.0, "savings": 4.5}.items()}

PRODUCTS = ["home_loan", "credit_card", "savings"]
expected = {p: baseline[p] + search_contrib[p] + sum(contrib[(c, p)] for c in CHANNELS) for p in PRODUCTS}
submits = {p: rng.poisson(expected[p]) for p in PRODUCTS}

# ---------------------------------------------------------------------------
# 2. Meta exports
# ---------------------------------------------------------------------------
os.makedirs(f"{OUT}/meta/au_account", exist_ok=True)
os.makedirs(f"{OUT}/meta/legacy_account", exist_ok=True)

LEGACY_END = date(2025, 6, 30)   # brand moved to the AU account from 1 Jul 2025
fx_audusd = 0.655 + 0.02 * np.sin(t / 90) + np.cumsum(rng.normal(0, 0.0015, N))

au_campaigns = [
    # (campaign_id, name(s), channel, product_weight, adsets)
    ("120210448811230", "HLB | Home Loans | Prospecting | FY25", "meta_prospecting", 0.35,
     [("120210448811231", "Broad 25-54 | AU"), ("120210448811232", "LAL 1% Applicants")]),
    ("120210448811240", "HLB | Credit Cards | Prospecting", "meta_prospecting", 0.35,
     [("120210448811241", "Broad 18-45 | AU"), ("120210448811242", "Interest - Travel")]),
    ("120210448811250", "HLB | Everyday & Savings | Prospecting", "meta_prospecting", 0.30,
     [("120210448811251", "Broad 18-35 | AU"), ("120210448811252", "LAL 3% Customers")]),
    ("120210448811260", "HLB | Retargeting | All Products", "meta_retargeting", 1.0,
     [("120210448811261", "Site visitors 30d"), ("120210448811262", "App engagers 14d")]),
    ("120231907765410", "HLB | Brand | Awareness | AU", "meta_brand", 1.0,
     [("120231907765411", "Reach | 18-65 | Video"), ("120231907765412", "Reach | 18-65 | Stories")]),
]
legacy_campaigns = [
    ("23851200774410", "Harbourline Brand Awareness - Video", "meta_brand", 0.65,
     [("23851200774411", "AU 18-65 - In-stream")]),
    ("23851200774420", "Harbourline Brand - Reach", "meta_brand", 0.35,
     [("23851200774421", "AU 18-65 - Feed"), ("23851200774422", "AU 18-65 - Reels")]),
]
RENAME_DATE = date(2026, 1, 19)

meta_cpm = {"meta_prospecting": 9.5, "meta_retargeting": 14.0, "meta_brand": 6.0}
meta_ctr = {"meta_prospecting": 0.009, "meta_retargeting": 0.016, "meta_brand": 0.003}
meta_rows_au, meta_rows_legacy = [], []
meta_clicks = np.zeros((N, 3))  # prospecting, retargeting, brand -> used for GA4 sessions

for i, d in enumerate(dates_py):
    for (cid, cname, ch, w, adsets) in au_campaigns + legacy_campaigns:
        legacy = cid.startswith("2385")
        if ch == "meta_brand" and legacy != (d <= LEGACY_END):
            continue
        if ch == "meta_brand":
            camp_spend = spend[ch][i] * w
        elif ch == "meta_prospecting":
            mix = {"120210448811230": 0.35 * (1 + 0.6 * promo_hl[i]), "120210448811240": 0.35 * (1 + 0.6 * promo_cc[i]),
                   "120210448811250": 0.30}
            camp_spend = spend[ch][i] * mix[cid] / sum(mix.values())
        else:
            camp_spend = spend[ch][i]
        if camp_spend <= 0:
            continue
        name = cname
        if cid == "120210448811230" and d >= RENAME_DATE:
            name = "HLB | Home Loans | Prospecting | Always On"
        aw = rng.dirichlet(np.ones(len(adsets)) * 4)
        for (aid, aname), share in zip(adsets, aw):
            s = camp_spend * share
            imp = int(s / meta_cpm[ch] * 1000 * rng.uniform(0.85, 1.15))
            if imp == 0:
                continue
            clicks = rng.binomial(imp, meta_ctr[ch])
            freq = rng.uniform(1.05, 1.6) if ch != "meta_retargeting" else rng.uniform(1.6, 3.2)
            reach = int(imp / freq)
            results = rng.poisson(clicks * (0.08 if ch == "meta_retargeting" else 0.035))  # pixel-claimed applications
            k = {"meta_prospecting": 0, "meta_retargeting": 1, "meta_brand": 2}[ch]
            meta_clicks[i, k] += clicks
            row = {"cid": cid, "cname": name, "aid": aid, "aname": aname, "imp": imp, "reach": reach,
                   "clicks": clicks, "results": results, "spend_aud": s}
            if legacy:
                # legacy account: US-locale export, account timezone America/Los_Angeles, currency USD
                row["spend"] = round(s * fx_audusd[i], 2)
                meta_rows_legacy.append((i, row))
            else:
                row["spend"] = round(s, 2)
                meta_rows_au.append((d, row))

# legacy account days are Los Angeles days: LA day d ~ 25% of Sydney day d + 75% of Sydney day d+1
legacy_by_la_day = {}
for i, row in meta_rows_legacy:
    for la_i, frac in [(i, 0.25), (i - 1, 0.75)]:
        if la_i < 0:
            continue
        key = (la_i, row["aid"])
        r = legacy_by_la_day.setdefault(key, {**row, "imp": 0, "reach": 0, "clicks": 0, "results": 0, "spend": 0.0})
        for f in ["imp", "reach", "clicks", "results"]:
            r[f] += int(round(row[f] * frac))
        r["spend"] += row["spend"] * frac


def write_meta(rows_by_month, folder, fmt_date, currency, prefix, overlap_months=()):
    cols = ["Day", "Campaign name", "Campaign ID", "Ad set name", "Ad set ID", f"Amount spent ({currency})",
            "Impressions", "Reach", "Frequency", "Link clicks", "CTR (link click-through rate)", "Results",
            "Result indicator", "Cost per result", "Reporting starts", "Reporting ends"]
    months = sorted(rows_by_month)
    for m in months:
        rows = list(rows_by_month[m])
        if m in overlap_months:  # analyst re-pulled with the previous month's last day included
            prev = months[months.index(m) - 1]
            last_day = max(r[0] for r in rows_by_month[prev])
            rows = [r for r in rows_by_month[prev] if r[0] == last_day] + rows
        recs = []
        start = min(r[0] for r in rows)
        end = max(r[0] for r in rows)
        for d, r in rows:
            sp = round(r["spend"], 2)
            recs.append({
                "Day": fmt_date(d), "Campaign name": r["cname"], "Campaign ID": r["cid"],
                "Ad set name": r["aname"], "Ad set ID": r["aid"], f"Amount spent ({currency})": sp,
                "Impressions": r["imp"], "Reach": r["reach"],
                "Frequency": round(r["imp"] / r["reach"], 6) if r["reach"] else "",
                "Link clicks": r["clicks"] if r["clicks"] else "",
                "CTR (link click-through rate)": round(100 * r["clicks"] / r["imp"], 6) if r["imp"] else "",
                "Results": r["results"] if r["results"] else "",
                "Result indicator": "offsite_conversion.fb_pixel_custom.ApplicationSubmitted" if r["results"] else "",
                "Cost per result": round(sp / r["results"], 6) if r["results"] else "",
                "Reporting starts": fmt_date(start), "Reporting ends": fmt_date(end),
            })
        df = pd.DataFrame(recs, columns=cols)
        df.to_csv(f"{folder}/{prefix}_{m}.csv", index=False)


au_by_month = {}
for d, r in meta_rows_au:
    au_by_month.setdefault(d.strftime("%Y-%m"), []).append((d, r))
write_meta(au_by_month, f"{OUT}/meta/au_account", lambda d: d.strftime("%d/%m/%Y"), "AUD",
           "Harbourline-AU-Ads-Export", overlap_months=("2025-02", "2025-09", "2026-03", "2026-07"))

legacy_by_month = {}
for (la_i, aid), r in legacy_by_la_day.items():
    d = dates_py[la_i]
    legacy_by_month.setdefault(d.strftime("%Y-%m"), []).append((d, r))
for m in legacy_by_month:
    legacy_by_month[m].sort(key=lambda x: (x[0], x[1]["aid"]))
write_meta(legacy_by_month, f"{OUT}/meta/legacy_account", lambda d: d.strftime("%m/%d/%Y"), "USD",
           "Harbourline-Bank-(Legacy)-Campaigns")

# ---------------------------------------------------------------------------
# 3. DV360 quarterly report downloads
# ---------------------------------------------------------------------------
os.makedirs(f"{OUT}/dv360", exist_ok=True)
ADV, ADV_ID = "Harbourline Bank", "4410093"
dv_lines = {
    "dv360_display": [
        ("HLB_Display_Prospecting", "71200341", "Display", [
            ("HLB_FY25_HomeLoans_Display_Prospecting_Contextual", "88410021"),
            ("HLB_FY25_CreditCards_Display_Prospecting_InMarket", "88410022"),
            ("HLB_Savings_Display_Prosp_OpenAuction", "88410023"),
        ]),
    ],
    "dv360_youtube": [
        ("HLB_YouTube_Reach", "71200355", "YouTube & partners", [
            ("HLB_FY25_Brand_YouTube_Bumper_A18-54", "88410031"),
            ("HLB_FY25_HomeLoans_Youtube_Skippable_InMarket", "88410032"),
            ("HLB_YT_Savings_NonSkip_15s", "88410033"),
        ]),
    ],
    "dv360_ctv": [
        ("HLB_FY26_CTV_Video", "71200402", "Video", [
            ("HLB_FY26_Brand_OLV_CTV_PremiumBVOD_30s", "88410041"),
            ("HLB_FY26_HomeLoans_OLV_CTV_PMP_15s", "88410042"),
        ]),
    ],
}
dv_cpm = {"dv360_display": 4.2, "dv360_youtube": 13.0, "dv360_ctv": 38.0}
dv_ctr = {"dv360_display": 0.0012, "dv360_youtube": 0.0025, "dv360_ctv": 0.0}
markup = 1.12        # Revenue (Adv Currency) = billed to client
data_fee = 0.06      # Total Media Cost = media cost + data/partner fees
dv_clicks = np.zeros((N, 3))
dv_rows = []
for i, d in enumerate(dates_py):
    for k, (ch, ios) in enumerate(dv_lines.items()):
        if spend[ch][i] <= 0:
            continue
        for io, io_id, li_type, lis in ios:
            shares = rng.dirichlet(np.ones(len(lis)) * 5)
            for (li, li_id), sh in zip(lis, shares):
                rev = spend[ch][i] * sh
                tmc = rev / markup
                mc = tmc / (1 + data_fee)
                imp = int(mc / dv_cpm[ch] * 1000 * rng.uniform(0.9, 1.1))
                clicks = rng.binomial(imp, dv_ctr[ch]) if dv_ctr[ch] else 0
                dv_clicks[i, k] += clicks
                views = int(imp * rng.uniform(0.55, 0.75)) if ch != "dv360_display" else 0
                conv = rng.poisson(imp / 4000 * (1.5 if ch == "dv360_display" else 1.0))  # post-view heavy floodlight
                dv_rows.append([d, ADV, ADV_ID, "AUD", io, io_id, li, li_id, li_type, imp, clicks,
                                views, conv, round(mc, 6), round(tmc, 6), round(rev, 6)])
# make-good credits from the trading desk (negative revenue, no delivery)
for d in [date(2025, 1, 31), date(2025, 6, 30), date(2025, 12, 31), date(2026, 6, 30)]:
    dv_rows.append([d, ADV, ADV_ID, "AUD", "HLB_Display_Prospecting", "71200341",
                    "HLB_Savings_Display_Prosp_OpenAuction", "88410023", "Display", 0, 0, 0, 0,
                    0.0, 0.0, -round(rng.uniform(1500, 4000), 2)])

dv_cols = ["Date", "Advertiser", "Advertiser ID", "Advertiser Currency", "Insertion Order", "Insertion Order ID",
           "Line Item", "Line Item ID", "Line Item Type", "Impressions", "Clicks", "TrueView: Views",
           "Total Conversions", "Media Cost (Advertiser Currency)", "Total Media Cost (Advertiser Currency)",
           "Revenue (Adv Currency)"]
dv = pd.DataFrame(dv_rows, columns=dv_cols)
dv["q"] = pd.PeriodIndex(pd.to_datetime(dv["Date"]), freq="Q")
for q, g in dv.groupby("q"):
    g = g.drop(columns="q").sort_values(["Date", "Line Item ID"])
    fn = f"{OUT}/dv360/DV360_Harbourline_Standard_{str(q).replace('Q', '_Q')}.csv"
    body = g.copy()
    body["Date"] = pd.to_datetime(body["Date"]).dt.strftime("%Y/%m/%d")
    with open(fn, "w", newline="") as f:
        body.to_csv(f, index=False)
        tot = [""] * 9 + [int(g[c].sum()) for c in ["Impressions", "Clicks", "TrueView: Views", "Total Conversions"]] \
              + [round(g[c].sum(), 6) for c in dv_cols[-3:]]
        f.write(",".join(str(x) for x in tot) + "\n\n")
        f.write(f"Report Time:,{(pd.Timestamp(q.end_time.date()) + pd.Timedelta(days=3)).strftime('%Y/%m/%d')} 09:14:22 AEDT\n")
        f.write(f"Date Range:,{q.start_time.strftime('%Y/%m/%d')} to {q.end_time.strftime('%Y/%m/%d')}\n")
        f.write("Group By:,Date,Advertiser,Insertion Order,Line Item\n")
        f.write("MRC Accredited Metrics\n")
        f.write("Active View metrics are accredited only when Measurement Source = Measured or Measurable\n")

# ---------------------------------------------------------------------------
# 4. GA4 export
# ---------------------------------------------------------------------------
os.makedirs(f"{OUT}/ga4", exist_ok=True)

STREAMS = {"WEB": "4392017781", "IOS": "4392133904", "ANDROID": "4392133377"}
APP_INFO = {"IOS": ("com.harbourline.mobile", "6.14.2"), "ANDROID": ("au.com.harbourline.mobile", "6.14.1")}
REGIONS = ["New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia",
           "Australian Capital Territory", "Tasmania", "Northern Territory"]
REGION_P = np.array([0.33, 0.26, 0.20, 0.10, 0.07, 0.02, 0.015, 0.005]); REGION_P /= REGION_P.sum()
CITIES = {"New South Wales": ["Sydney", "Newcastle", "Wollongong"], "Victoria": ["Melbourne", "Geelong"],
          "Queensland": ["Brisbane", "Gold Coast"], "Western Australia": ["Perth"], "South Australia": ["Adelaide"],
          "Australian Capital Territory": ["Canberra"], "Tasmania": ["Hobart"], "Northern Territory": ["Darwin"]}
HOUR_P = np.array([1, .5, .3, .2, .2, .4, 1.2, 2.4, 3.6, 4.4, 4.6, 4.6, 5.0, 4.8, 4.4, 4.2, 4.0, 4.0,
                   4.4, 5.2, 5.6, 5.0, 3.6, 2.0]); HOUR_P /= HOUR_P.sum()

CUSTOMERS = 180_000
customer_hash = {}


def cust_hash(ci):
    h = customer_hash.get(ci)
    if h is None:
        h = hashlib.sha256(f"hlb-cust-{ci}".encode()).hexdigest()
        customer_hash[ci] = h
    return h


def pseudo_id(kind, n):
    return f"{n % 2147483647}.{1600000000 + (n * 7919) % 120000000}" if kind == "WEB" else \
        format((n * 2654435761) % (1 << 64), "016X") + format((n * 40503) % (1 << 64), "016X")


def ep(**kw):
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        val = {"string_value": None, "int_value": None, "float_value": None, "double_value": None}
        if isinstance(v, bool) or isinstance(v, (int, np.integer)):
            val["int_value"] = int(v)
        elif isinstance(v, float):
            val["double_value"] = v
        else:
            val["string_value"] = str(v)
        out.append({"key": k, "value": val})
    return out


SCHEMA = pa.schema([
    ("event_date", pa.string()),
    ("event_timestamp", pa.int64()),
    ("event_name", pa.string()),
    ("event_params", pa.list_(pa.struct([("key", pa.string()), ("value", pa.struct([
        ("string_value", pa.string()), ("int_value", pa.int64()), ("float_value", pa.float64()),
        ("double_value", pa.float64())]))]))),
    ("event_previous_timestamp", pa.int64()),
    ("event_bundle_sequence_id", pa.int64()),
    ("event_server_timestamp_offset", pa.int64()),
    ("user_id", pa.string()),
    ("user_pseudo_id", pa.string()),
    ("privacy_info", pa.struct([("analytics_storage", pa.string()), ("ads_storage", pa.string()),
                                ("uses_transient_token", pa.string())])),
    ("user_first_touch_timestamp", pa.int64()),
    ("device", pa.struct([("category", pa.string()), ("mobile_brand_name", pa.string()),
                          ("operating_system", pa.string()), ("operating_system_version", pa.string()),
                          ("language", pa.string()), ("web_info", pa.struct([("browser", pa.string()),
                                                                             ("hostname", pa.string())]))])),
    ("geo", pa.struct([("country", pa.string()), ("region", pa.string()), ("city", pa.string())])),
    ("app_info", pa.struct([("id", pa.string()), ("version", pa.string())])),
    ("traffic_source", pa.struct([("name", pa.string()), ("medium", pa.string()), ("source", pa.string())])),
    ("stream_id", pa.string()),
    ("platform", pa.string()),
    ("collected_traffic_source", pa.struct([("manual_campaign_name", pa.string()), ("manual_source", pa.string()),
                                            ("manual_medium", pa.string()), ("manual_content", pa.string()),
                                            ("gclid", pa.string()), ("dclid", pa.string()), ("srsltid", pa.string())])),
    ("is_active_user", pa.bool_()),
    ("batch_event_index", pa.int64()),
    ("batch_page_id", pa.int64()),
    ("batch_ordering_id", pa.int64()),
])

# traffic for a session: (source, medium, campaign, content, gclid, dclid)
def utm_meta(product, kind):
    src = rng.choice(["facebook", "instagram", "fb", "ig"], p=[0.5, 0.35, 0.1, 0.05])
    med = "paid_social"
    camp = {"home_loan": "hlb_homeloans_prospecting", "credit_card": "hlb_creditcards_prospecting",
            "savings": "hlb_savings_prospecting"}[product]
    if kind == "meta_retargeting":
        camp = "hlb_retargeting_allproducts"
    elif kind == "meta_brand":
        camp = "hlb_brand_awareness"
    if rng.random() < 0.06:
        med = "cpc"          # one creative set mis-tagged
    if rng.random() < 0.05:
        camp = camp.upper()
    return (src, med, camp, None, None, None)


def utm_dv(kind, product):
    src = {"dv360_display": "dv360", "dv360_youtube": "youtube", "dv360_ctv": "dv360"}[kind]
    med = {"dv360_display": "display", "dv360_youtube": "video", "dv360_ctv": "ctv"}[kind]
    camp = f"hlb_{kind.split('_')[1]}_{product.replace('_', '')}"
    dclid = "CJ" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789"), 22)) \
        if rng.random() < 0.7 else None
    return (src, med, camp, None, None, dclid)


def gclid():
    return "Cj0KCQjw" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP0123456789_-"), 60))


REFERRERS = ["canstar.com.au", "finder.com.au", "ratecity.com.au", "mozo.com.au"]


def pick_traffic(i, product, platform, intent_weights):
    """intent_weights: dict channel->weight"""
    keys = list(intent_weights)
    w = np.array([intent_weights[k] for k in keys]); w /= w.sum()
    k = keys[rng.choice(len(keys), p=w)]
    if platform != "WEB" and k not in ("direct", "push"):
        k = "direct" if rng.random() < 0.7 else k
    if k == "direct":
        return (None, None, None, None, None, None)
    if k == "organic":
        return (rng.choice(["google", "bing"], p=[0.94, 0.06]), "organic", None, None, None, None)
    if k == "search":
        return ("google", "cpc", f"hlb_search_{product.replace('_', '')}_{'brand' if rng.random()<0.55 else 'generic'}",
                None, gclid(), None)
    if k == "email":
        return ("sfmc", "email", f"edm_{dates_py[i].strftime('%Y%m')}_{product}", None, None, None)
    if k == "push":
        return ("harbourline_app", "push", f"push_{product}_offer", None, None, None)
    if k == "referral":
        return (rng.choice(REFERRERS), "referral", None, None, None, None)
    if k.startswith("meta"):
        return utm_meta(product, k)
    return utm_dv(k, product)


PAGES = {
    "home_loan": ["/home-loans", "/home-loans/variable-rate", "/home-loans/calculators/borrowing-power",
                  "/home-loans/first-home-buyers", "/home-loans/refinance"],
    "credit_card": ["/credit-cards", "/credit-cards/low-rate", "/credit-cards/rewards-platinum", "/credit-cards/compare"],
    "savings": ["/bank-accounts", "/bank-accounts/bonus-saver", "/bank-accounts/everyday", "/bank-accounts/term-deposits"],
}
APPLY = {"home_loan": "/home-loans/apply", "credit_card": "/credit-cards/apply", "savings": "/bank-accounts/open"}
SCREENS = {"home_loan": ["HomeLoanOverview", "BorrowingPowerCalc"], "credit_card": ["CardsCatalogue", "CardDetail"],
           "savings": ["AccountsCatalogue", "BonusSaverDetail"]}

TAG_BREAK = (date(2025, 7, 14), date(2025, 7, 22))     # web application_submit missing
CC_RENAME = date(2025, 3, 3)                           # before: cc_application_complete
PII_WINDOW = (date(2025, 2, 10), date(2025, 5, 6))     # email in callback page_location
MISSING_DAY = date(2025, 11, 3)                        # export failed
UID_UPPER_APP_FROM = date(2025, 9, 1)                  # app release starts sending upper-case user_id
QA_DAYS = set(dates_py[j] for j in rng.choice(N, 14, replace=False) if dates_py[j].weekday() < 5)
DUP_DAYS = set(dates_py[j] for j in rng.choice(N, 45, replace=False))

anon_counter = [10_000_000]
qa_submit_counts = {}
truth_rows = []
total_events = 0


def new_anon():
    anon_counter[0] += int(rng.integers(1, 50))
    return anon_counter[0]


for i, d in enumerate(dates_py):
    if d == MISSING_DAY:
        truth_rows.append(None)
        continue
    off = syd_offset(d)
    day_start_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(hours=off)
    rows = []
    seq = [0]

    # channel weights for converting sessions (click-through share of each channel's true contribution)
    def conv_weights(p):
        w = {"direct": 0.24 * baseline[p][i], "organic": 0.20 * baseline[p][i], "referral": 0.09 * baseline[p][i],
             "email": 0.04 * baseline[p][i], "push": 0.03 * baseline[p][i], "search": search_contrib[p][i] * 2.2}
        click = {"meta_prospecting": 0.45, "meta_retargeting": 0.9, "meta_brand": 0.08,
                 "dv360_display": 0.12, "dv360_youtube": 0.10, "dv360_ctv": 0.0}
        for c in CHANNELS:
            if click[c] > 0 and contrib[(c, p)][i] > 0:
                w[c] = contrib[(c, p)][i] * click[c]
        # whatever isn't click-through shows up as direct/organic
        w["direct"] += sum(contrib[(c, p)][i] * (1 - click[c]) for c in CHANNELS) * 0.6
        w["organic"] += sum(contrib[(c, p)][i] * (1 - click[c]) for c in CHANNELS) * 0.4
        return w

    def session(platform, kind, product=None, traffic=None, customer=None, qa=False):
        start = day_start_utc + timedelta(hours=int(rng.choice(24, p=HOUR_P)), seconds=float(rng.uniform(0, 3600)))
        consent = rng.random() > (0.30 if platform == "WEB" else 0.12)
        if qa:
            consent = True
        if customer is not None:
            upid_n = customer * 3 + {"WEB": 0, "IOS": 1, "ANDROID": 2}[platform] + 500_000_000
        else:
            upid_n = new_anon()
        upid = pseudo_id(platform, upid_n) if consent else None
        sid = int(start.timestamp())
        snum = int(rng.integers(1, 30)) if customer is not None else int(rng.choice([1, 1, 1, 2, 3, 4]))
        region = rng.choice(REGIONS, p=REGION_P)
        city = rng.choice(CITIES[region])
        if platform == "WEB":
            cat = rng.choice(["desktop", "mobile", "tablet"], p=[0.42, 0.53, 0.05])
            device = {"category": cat, "mobile_brand_name": None if cat == "desktop" else rng.choice(["Apple", "Samsung", "Google"]),
                      "operating_system": {"desktop": rng.choice(["Windows", "Macintosh"]), "mobile": rng.choice(["iOS", "Android"]),
                                           "tablet": "iOS"}[cat],
                      "operating_system_version": None, "language": "en-au",
                      "web_info": {"browser": rng.choice(["Chrome", "Safari", "Edge", "Firefox"], p=[.55, .33, .09, .03]),
                                   "hostname": "uat.harbourline.com.au" if qa else "www.harbourline.com.au"}}
            app = None
        else:
            device = {"category": "mobile", "mobile_brand_name": "Apple" if platform == "IOS" else rng.choice(["Samsung", "Google", "Oppo"]),
                      "operating_system": "iOS" if platform == "IOS" else "Android",
                      "operating_system_version": "iOS 19.1" if platform == "IOS" else "Android 16",
                      "language": "en-au", "web_info": None}
            app = {"id": APP_INFO[platform][0], "version": APP_INFO[platform][1]}
        src, med, camp, content, gcl, dcl = traffic or (None, None, None, None, None, None)
        uid = None
        if customer is not None and kind in ("login",) or (customer is not None and rng.random() < 0.8):
            uid = cust_hash(customer)
            if platform != "WEB" and d >= UID_UPPER_APP_FROM:
                uid = uid.upper()
        first_touch = int((start - (timedelta(days=int(rng.integers(0, 900))) if snum > 1 else timedelta(0))).timestamp() * 1e6)
        ts = [start]
        bidx = [0]

        def emit(name, **params):
            ts[0] = ts[0] + timedelta(seconds=float(rng.exponential(25)))
            p = dict(params)
            if consent:
                p["ga_session_id"] = sid
                p["ga_session_number"] = snum
            if qa:
                p["debug_mode"] = 1
            seq[0] += 1
            rows.append({
                "event_date": d.strftime("%Y%m%d"),
                "event_timestamp": int(ts[0].timestamp() * 1e6),
                "event_name": name,
                "event_params": ep(**p),
                "event_previous_timestamp": None,
                "event_bundle_sequence_id": seq[0] if platform != "WEB" else None,
                "event_server_timestamp_offset": int(rng.integers(100000, 900000)) if platform != "WEB" else None,
                "user_id": uid if consent else None,
                "user_pseudo_id": upid,
                "privacy_info": {"analytics_storage": "Yes" if consent else "No",
                                 "ads_storage": ("Yes" if rng.random() < 0.6 else "No") if consent else "No",
                                 "uses_transient_token": "No" if consent else "Yes"},
                "user_first_touch_timestamp": first_touch if consent else None,
                "device": device,
                "geo": {"country": "Australia", "region": region, "city": city},
                "app_info": app,
                "traffic_source": {"name": camp or "(direct)", "medium": med or "(none)", "source": src or "(direct)"}
                                  if consent else {"name": None, "medium": None, "source": None},
                "stream_id": STREAMS[platform],
                "platform": platform,
                "collected_traffic_source": ({"manual_campaign_name": camp, "manual_source": src, "manual_medium": med,
                                              "manual_content": content, "gclid": gcl, "dclid": dcl, "srsltid": None}
                                             if (bidx[0] == 0 and (src or gcl)) else None),
                "is_active_user": True,
                "batch_event_index": bidx[0],
                "batch_page_id": int(ts[0].timestamp() * 1000) if platform == "WEB" else None,
                "batch_ordering_id": int(rng.integers(1, 40)) if platform == "WEB" else None,
            })
            bidx[0] += 1

        host = "https://uat.harbourline.com.au" if qa else "https://www.harbourline.com.au"

        def page(path, query=""):
            if platform == "WEB":
                emit("page_view", page_location=f"{host}{path}{query}", page_title=path.strip('/').replace('/', ' | ').title() or "Home",
                     engagement_time_msec=int(rng.integers(500, 60000)))
            else:
                emit("screen_view", firebase_screen=path, engagement_time_msec=int(rng.integers(500, 60000)))

        emit("session_start")
        if snum == 1 and consent:
            emit("first_visit" if platform == "WEB" else "first_open")
        qs = ""
        if platform == "WEB" and src and camp:
            qs = f"?utm_source={src}&utm_medium={med}&utm_campaign={camp}"
        if kind == "login":
            page("/" if platform == "WEB" else "Login", qs)
            emit("login", method="password" if rng.random() < 0.6 else "biometric")
            page("/online-banking/accounts" if platform == "WEB" else "AccountsHome")
            if rng.random() < 0.3:
                page("/online-banking/transfer" if platform == "WEB" else "PayAnyone")
            return
        if kind == "browse":
            pth = PAGES[product] if platform == "WEB" else SCREENS[product]
            first = rng.choice(pth)
            if platform == "WEB" and product == "home_loan" and promo_hl[i] and rng.random() < 0.4:
                first = "/home-loans/cashback-offer"
            page(first, qs)
            for _ in range(int(rng.integers(0, 4))):
                page(rng.choice(pth))
            if rng.random() < 0.5:
                emit("user_engagement", engagement_time_msec=int(rng.integers(1000, 120000)))
            return
        if kind == "lead":
            page("/home-loans", qs)
            q = ""
            if PII_WINDOW[0] <= d <= PII_WINDOW[1] and rng.random() < 0.6:
                q = f"?submitted=true&email={rng.choice(['jsmith','m.nguyen','priya.k','tom.w','a.chen','rkelly'])}{rng.integers(10,99)}%40example.com"
            page("/home-loans/request-callback")
            if platform == "WEB":
                emit("generate_lead", product="home_loan", lead_type="callback",
                     page_location=f"{host}/home-loans/request-callback{q}")
            return
        # apply_start / submit
        pth = PAGES[product] if platform == "WEB" else SCREENS[product]
        page(rng.choice(pth), qs)
        page(APPLY[product] if platform == "WEB" else f"Apply_{product}")
        cc_old = product == "credit_card" and d < CC_RENAME
        emit("application_start", product=None if cc_old else product, form_name=f"{product}_application")
        if kind == "submit":
            for step in range(2, 5 if product == "home_loan" else 4):
                emit("form_step", form_name=f"{product}_application", step_number=step)
            app_id = f"{product[:2].upper()}{d.strftime('%y%m%d')}{int(rng.integers(100000, 999999))}"
            web_broken = platform == "WEB" and TAG_BREAK[0] <= d <= TAG_BREAK[1]
            if cc_old:
                emit("cc_application_complete", card_type=rng.choice(["low_rate", "rewards_platinum"]), application_id=app_id)
            elif not web_broken:
                emit("application_submit", product=product, application_id=app_id)
            page(APPLY[product] + "/confirmation" if platform == "WEB" else f"ApplyConfirmation_{product}")

    web_share = {"home_loan": 0.9, "credit_card": 0.7, "savings": 0.5}
    for p in PRODUCTS:
        w = conv_weights(p)
        for _ in range(int(submits[p][i])):
            plat = "WEB" if rng.random() < web_share[p] else ("IOS" if rng.random() < 0.6 else "ANDROID")
            cust = int(rng.integers(0, CUSTOMERS)) if rng.random() < (0.35 if p != "home_loan" else 0.25) else None
            session(plat, "submit", p, pick_traffic(i, p, plat, w), cust)
        for _ in range(rng.poisson(1.8 * expected[p][i])):
            plat = "WEB" if rng.random() < web_share[p] else ("IOS" if rng.random() < 0.6 else "ANDROID")
            session(plat, "start", p, pick_traffic(i, p, plat, w), None)
        # browsing (mix of media-driven landings and everything else)
        nb = rng.poisson(40 * baseline[p][i] / baseline[p].mean())
        for _ in range(nb):
            plat = "WEB" if rng.random() < 0.8 else ("IOS" if rng.random() < 0.6 else "ANDROID")
            session(plat, "browse", p, pick_traffic(i, p, plat, w), None)
    # paid landings that bounce (media clicks -> sessions)
    for k, ch in enumerate(["meta_prospecting", "meta_retargeting", "meta_brand"]):
        for _ in range(rng.binomial(int(meta_clicks[i, k]), 0.07)):
            p = rng.choice(PRODUCTS)
            session("WEB", "browse", p, utm_meta(p, ch), None)
    for k, ch in enumerate(["dv360_display", "dv360_youtube", "dv360_ctv"]):
        for _ in range(rng.binomial(int(dv_clicks[i, k]), 0.15)):
            p = rng.choice(PRODUCTS)
            session("WEB", "browse", p, utm_dv(ch, p), None)
    for _ in range(rng.poisson(0.6 * expected["home_loan"][i])):
        session("WEB", "lead", "home_loan", pick_traffic(i, "home_loan", "WEB", conv_weights("home_loan")), None)
    # existing customers logging in (most traffic, irrelevant to acquisition)
    for _ in range(rng.poisson(110 * dow_mult[i])):
        session("WEB", "login", customer=int(rng.integers(0, CUSTOMERS)))
    for _ in range(rng.poisson(170 * dow_mult[i])):
        session("IOS" if rng.random() < 0.6 else "ANDROID", "login", customer=int(rng.integers(0, CUSTOMERS)))
    # QA test submissions on UAT pushed through the production stream
    nqa = 0
    if d in QA_DAYS:
        nqa = int(rng.integers(12, 40))
        for _ in range(nqa):
            session("WEB", "submit", rng.choice(PRODUCTS), None, None, qa=True)
    qa_submit_counts[d] = nqa

    if d in DUP_DAYS:
        dup_idx = rng.choice(len(rows), int(len(rows) * rng.uniform(0.005, 0.02)), replace=False)
        rows += [rows[j] for j in dup_idx]
    rows.sort(key=lambda r: r["event_timestamp"])
    tbl = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(tbl, f"{OUT}/ga4/events_{d.strftime('%Y%m%d')}.parquet", compression="zstd")
    total_events += len(rows)
    if i % 60 == 0:
        print(f"  GA4 {d}  ({total_events:,} events so far)")

# ---------------------------------------------------------------------------
# 5. Ground truth (sealed)
# ---------------------------------------------------------------------------
os.makedirs(TRUTH, exist_ok=True)
truth = pd.DataFrame({"date": dates_py, "cash_rate_simulated": cash_rate, "promo_home_loan": promo_hl,
                      "promo_credit_card": promo_cc})
for c in CHANNELS:
    truth[f"spend_aud__{c}"] = spend[c]
for p in PRODUCTS:
    truth[f"baseline__{p}"] = baseline[p].round(4)
    truth[f"search_contrib__{p}"] = search_contrib[p].round(4)
    for c in CHANNELS:
        truth[f"contrib__{c}__{p}"] = contrib[(c, p)].round(4)
    truth[f"expected__{p}"] = expected[p].round(4)
    truth[f"true_submits__{p}"] = submits[p]
truth["qa_test_submits"] = [qa_submit_counts.get(d, 0) for d in dates_py]
truth.to_csv(f"{TRUTH}/true_daily.csv", index=False)

summary = {"seed": SEED, "period": [str(START), str(END)],
           "channels": {c: {"weekly_decay": v["decay"], "half_saturation_daily_aud": v["K"], "hill_slope": v["s"],
                            "beta_submits_per_day_at_saturation": v["beta"],
                            "total_spend_aud": round(float(spend[c].sum()), 2),
                            "total_contribution_submits": {p: round(float(contrib[(c, p)].sum()), 1) for p in PRODUCTS},
                            "cost_per_incremental_submit_aud": round(float(spend[c].sum() / max(1e-9, sum(contrib[(c, p)].sum() for p in PRODUCTS))), 2)}
                        for c, v in CHANNELS.items()},
           "unobserved_paid_search_contribution": {p: round(float(search_contrib[p].sum()), 1) for p in PRODUCTS},
           "baseline_total": {p: round(float(baseline[p].sum()), 1) for p in PRODUCTS},
           "true_submits_total": {p: int(submits[p].sum()) for p in PRODUCTS},
           "dv360_spend_definition": "spend_aud = Revenue (Adv Currency); Total Media Cost = Revenue/1.12; Media Cost = Total Media Cost/1.06",
           "meta_legacy": "USD at simulated daily AUDUSD (truth column not provided); LA-timezone days ~ 25% Sydney day d + 75% Sydney day d+1",
           "planted_issues": [
               "GA4: no export file for 2025-11-03",
               "GA4: web application_submit not firing 2025-07-14..2025-07-22 (confirmation page_view still fires)",
               "GA4: credit card submits are event 'cc_application_complete' (no product param) before 2025-03-03",
               "GA4: exact duplicate rows on 45 random days (0.5-2%)",
               "GA4: email address in page_location of generate_lead events 2025-02-10..2025-05-06",
               "GA4: QA test submissions from uat.harbourline.com.au with debug_mode=1 on ~10 weekdays",
               "GA4: consent-denied events (null user_pseudo_id, no ga_session_id) still carry key events",
               "GA4: app user_id upper-cased from 2025-09-01; web stays lower-case",
               "GA4: ~6% of Meta traffic tagged utm_medium=cpc; some utm_campaign upper-cased; source fb/ig variants",
               "GA4: login/online-banking traffic dominates sessions and is not acquisition",
               "Meta: two ad accounts; legacy is USD, US date format (mm/dd/yyyy), America/Los_Angeles days, ended 2025-06-30",
               "Meta: AU account dd/mm/yyyy; monthly files 2025-02, 2025-09, 2026-03, 2026-07 repeat previous month's last day",
               "Meta: campaign 120210448811230 renamed 2026-01-19 (same ID)",
               "Meta: zero-delivery days absent (no zero rows); blanks for zero clicks/results",
               "Meta/DV360: platform-claimed conversions (Results, Total Conversions) are not incrementality",
               "DV360: totals row + blank line + metadata footer at the end of every file",
               "DV360: three spend columns (media cost, total media cost, revenue); client is billed Revenue",
               "DV360: negative make-good credit rows with zero delivery at half-year ends",
               "DV360: CTV is Line Item Type 'Video' - identify via name token CTV; inconsistent naming (YouTube/Youtube/YT, Prospecting/Prosp)",
               "DV360: CTV only from Aug 2025 (short history)",
               "Missing channel: Google paid search runs (gclid traffic in GA4) but no spend was supplied",
               "Hidden drivers: cash rate changes, home-loan and credit-card promo windows (cashback-offer page hints)",
           ]}
with open(f"{TRUTH}/true_params.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(f"{TRUTH}/README.md", "w") as f:
    f.write("# Sealed\n\nThe generating model and every planted issue. Open only to validate the pipeline/MMM at the end.\n")

print(f"Done: {total_events:,} GA4 events, {len(dv):,} DV360 rows, "
      f"{len(meta_rows_au):,} Meta AU rows, {len(legacy_by_la_day):,} Meta legacy rows.")
