

from __future__ import annotations

import argparse
import io
import re
import threading
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, dcc, html
except ImportError as exc:
    raise SystemExit(
        f"Missing package: {exc.name}. In the VS Code terminal run:\n"
        "python -m pip install dash plotly pandas openpyxl"
    ) from exc


UCI_ZIP = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/352/online%2Bretail"
CUTOFF = pd.Timestamp("2011-09-09")
BG, PAPER, INK, MUTED = "#faf8f2", "#ffffff", "#332e25", "#807567"
RUST, AMBER, OLIVE, GRID = "#bd623c", "#ddb66e", "#74895d", "#eee8df"
NON_PRODUCT_CODES = {
    "POST", "D", "DOT", "M", "MANUAL", "BANK CHARGES", "PADS", "CRUK",
    "C2", "AMAZONFEE", "SAMPLES",
}
CATEGORY_RULES = [
    ("Christmas & Seasonal", r"CHRISTMAS|XMAS|ADVENT|EASTER|HALLOWEEN"),
    ("Bags", r"\bBAG\b|\bBAGS\b|PURSE|WALLET"),
    ("Cards & Stationery", r"CARD|NOTEBOOK|PEN\b|PENCIL|POSTCARD|STATIONERY|ENVELOPE|GIFT TAG"),
    ("Candles & Holders", r"CANDLE|HOLDER|LANTERN|\bLIGHT\b|\bLIGHTS\b|LAMP"),
    ("Kitchen & Dining", r"MUG|\bCUP\b|PLATE|BOWL|SPOON|FORK|KNIFE|KITCHEN|\bTIN\b|\bJAR\b|TEAPOT|CUTLERY|BAKING|TEA TOWEL|CAKE"),
    ("Jewellery & Accessories", r"BRACELET|NECKLACE|EARRING|\bRING\b|CHARM|JEWEL"),
    ("Home Decor", r"HEART|\bSIGN\b|FRAME|\bCLOCK\b|CUSHION|DECORATION|ORNAMENT|HANGING|MIRROR|VASE|WALL ART"),
]
ORDER_LABELS = ["1", "2", "3–4", "5–9", "10+"]
RECENCY_LABELS = ["0–30", "31–60", "61–90", "91–180", "181+"]
SPEND_LABELS = ["≤£100", ">£100–250", ">£250–500", ">£500–1k", ">£1k–2.5k", ">£2.5k"]


def resolve_data(explicit: str | None) -> Path:
    """Use a local file when possible; otherwise fetch UCI's original workbook."""
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise SystemExit(f"Dataset not found: {candidate}")
        return candidate
    locations = [Path(__file__).resolve().parent, Path.cwd(), Path.home() / "Downloads"]
    for folder in locations:
        for name in ("Online Retail.xlsx", "Online_Retail.xlsx", "online_retail.xlsx"):
            candidate = folder / name
            if candidate.is_file():
                return candidate
    target_dir = Path.home() / "OnlineRetailDashboard"
    target = target_dir / "Online Retail.xlsx"
    if target.is_file():
        return target
    print("No local workbook found. Downloading the original UCI Online Retail file once...")
    try:
        with urllib.request.urlopen(UCI_ZIP, timeout=120) as response:
            payload = response.read(70_000_001)
        if len(payload) > 70_000_000:
            raise ValueError("Unexpectedly large archive")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next(n for n in archive.namelist() if Path(n).name.lower() == "online retail.xlsx")
            if archive.getinfo(member).file_size > 60_000_000:
                raise ValueError("Unexpectedly large workbook")
            target_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
        print(f"Saved dataset: {target}")
        return target
    except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as exc:
        raise SystemExit(
            f"Could not download the dataset ({exc}).\n"
            f"Download 'Online Retail.xlsx' from {SOURCE_URL}, put it next to this script, "
            "and run the script again."
        ) from exc


def category_for(description: str) -> str:
    for name, pattern in CATEGORY_RULES:
        if re.search(pattern, str(description).upper()):
            return name
    return "Other / General Gift"


def load_purchases(path: Path) -> tuple[pd.DataFrame, dict]:
    required = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    print(f"Loading {path.name} (Excel can take a little while)...")
    if path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, low_memory=False)
    else:
        raise SystemExit("--data must point to an .xlsx or .csv file with the UCI column names.")
    raw.columns = raw.columns.str.strip()
    missing = set(required) - set(raw.columns)
    if missing:
        raise SystemExit(f"Missing dataset columns: {', '.join(sorted(missing))}")
    raw = raw[required].copy()
    audit = {"raw_rows": len(raw), "missing_ids": int(raw.CustomerID.isna().sum())}
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        raw[col] = raw[col].astype(str).str.strip()
    raw["CustomerID"] = pd.to_numeric(raw.CustomerID, errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw.Quantity, errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw.UnitPrice, errors="coerce")
    raw["InvoiceDate"] = pd.to_datetime(raw.InvoiceDate, errors="coerce", dayfirst=False)
    raw = raw.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
    raw["CustomerID"] = raw.CustomerID.astype("int64")
    raw["IsCancelled"] = raw.InvoiceNo.str.upper().str.startswith("C")
    raw["IsNonProduct"] = raw.StockCode.str.upper().isin(NON_PRODUCT_CODES)
    audit["cancel_lines"] = int(raw.IsCancelled.sum())
    before = len(raw)
    raw = raw.drop_duplicates()
    audit["duplicates"] = before - len(raw)
    p = raw.loc[
        (~raw.IsCancelled) & (~raw.IsNonProduct) & (raw.Quantity > 0) & (raw.UnitPrice > 0)
    ].copy()
    if p.empty:
        raise SystemExit("No positive product purchase lines remain after cleaning.")
    p["Revenue"] = p.Quantity * p.UnitPrice
    p["Month"] = p.InvoiceDate.dt.to_period("M").dt.to_timestamp()

    # Match each SKU to its most common description, then apply the notebook's
    # keyword rules. Category names remain exploratory because many SKUs are unmatched.
    desc = p.groupby(["StockCode", "Description"], dropna=False).size().reset_index(name="n")
    desc = desc.sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
    desc = desc.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    categories = desc.map(category_for)
    p["Category"] = p.StockCode.map(categories)
    audit["named_sku_share"] = float((categories != "Other / General Gift").mean())
    audit["included_rows"] = len(p)
    audit["customers"] = p.CustomerID.nunique()
    audit["first_date"] = p.InvoiceDate.min().date()
    audit["last_date"] = p.InvoiceDate.max().date()
    print(f"Ready: {len(p):,} included purchase lines, {audit['customers']:,} identified customers.")
    return p, audit


def make_cohort(p: pd.DataFrame) -> pd.DataFrame:
    before = p.loc[p.InvoiceDate < CUTOFF]
    after = p.loc[p.InvoiceDate >= CUTOFF]
    if before.empty or after.empty:
        raise SystemExit("This dashboard needs purchases on both sides of 2011-09-09.")
    group = before.groupby("CustomerID", sort=False)
    c = group.agg(
        LastPurchase=("InvoiceDate", "max"), PriorOrders=("InvoiceNo", "nunique"),
        PriorSpend=("Revenue", "sum"),
    ).reset_index()
    # Country is normally stable per customer; use the most frequent value if not.
    countries = group.Country.agg(lambda s: s.mode().iloc[0])
    c["Country"] = c.CustomerID.map(countries)
    c["Repeat"] = c.CustomerID.isin(after.CustomerID.unique()).astype(int)
    c["RecencyDays"] = (CUTOFF - c.LastPurchase).dt.days
    c["OrderBin"] = pd.cut(c.PriorOrders, [0, 1, 2, 4, 9, np.inf], labels=ORDER_LABELS)
    c["RecencyBin"] = pd.cut(c.RecencyDays, [-1, 30, 60, 90, 180, np.inf], labels=RECENCY_LABELS)
    c["SpendBin"] = pd.cut(c.PriorSpend, [0, 100, 250, 500, 1000, 2500, np.inf], labels=SPEND_LABELS)
    return c


def market_filter(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "uk":
        return df.loc[df.Country.eq("United Kingdom")]
    if market == "outside":
        return df.loc[~df.Country.eq("United Kingdom")]
    return df


def money(amount: float, compact: bool = True) -> str:
    if compact and abs(amount) >= 1_000_000:
        return f"£{amount / 1_000_000:,.2f}m"
    return f"£{amount:,.0f}"


def card(label: str, value: str, detail: str, accent: str = "") -> html.Div:
    return html.Div([
        html.P(label, className="rj-stat-label"),
        html.P(value, className="rj-stat-value"),
        html.P(detail, className="rj-stat-detail"),
    ], className="rj-stat " + accent)


def base_figure(height: int = 350) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="none", height=height, autosize=True,
        margin=dict(l=58, r=22, t=28, b=54),
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="Arial, sans-serif", color=INK, size=12),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="white", size=13)),
        showlegend=False, bargap=0.34,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=MUTED), fixedrange=True),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED), fixedrange=True),
        hovermode="closest", dragmode=False,
    )
    return fig


def blank_figure(message: str, height: int = 350) -> go.Figure:
    fig = base_figure(height)
    fig.add_annotation(x=.5, y=.5, xref="paper", yref="paper", text=message,
                       showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def trend_figure(sales: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, metric: str) -> go.Figure:
    if sales.empty:
        return blank_figure("No purchases in this selection")
    series = sales.groupby("Month").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"))
    months = pd.date_range(start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp(), freq="MS")
    series = series.reindex(months, fill_value=0)
    values = series["Revenue" if metric == "revenue" else "Orders"]
    partial = [(m.date() < start.date()) or
               ((m + pd.offsets.MonthEnd(0)).date() > end.date()) for m in months]
    fig = base_figure()
    fig.add_bar(
        x=list(range(len(months))), y=values.tolist(),
        marker=dict(color=[AMBER if part else RUST for part in partial], cornerradius=5),
        customdata=[[m.strftime("%B %Y"), "Partial month" if part else "Full month"]
                    for m, part in zip(months, partial)],
        hovertemplate="%{customdata[0]}<br>" +
            ("Purchase revenue: £%{y:,.2f}" if metric == "revenue" else "Distinct orders: %{y:,}") +
            "<br>%{customdata[1]}<extra></extra>",
    )
    tick_indices = list(range(0, len(months), 2 if len(months) > 7 else 1))
    if len(months) - 1 not in tick_indices:
        tick_indices.append(len(months) - 1)
    labels = [months[i].strftime("%b") + ("<br>" + months[i].strftime("%Y")
              if i == 0 or i == len(months) - 1 else "") for i in tick_indices]
    fig.update_xaxes(tickmode="array", tickvals=tick_indices, ticktext=labels, tickangle=0,
                     range=[-.7, len(months) - .3])
    fig.update_yaxes(tickprefix="£" if metric == "revenue" else "", tickformat="~s", rangemode="tozero", nticks=5)
    return fig


def geography_figure(sales: pd.DataFrame) -> go.Figure:
    if sales.empty:
        return blank_figure("No purchases in this selection")
    totals = sales.groupby("Country", observed=True).Revenue.sum().sort_values(ascending=False)
    top = totals.head(6).copy()
    if len(totals) > 6:
        top.loc["Other countries"] = totals.iloc[6:].sum()
    shares = top / sales.Revenue.sum() * 100
    fig = base_figure()
    fig.add_bar(
        x=shares.iloc[::-1].tolist(), y=top.index[::-1].tolist(), orientation="h",
        marker=dict(color=[RUST if name == "United Kingdom" else OLIVE for name in top.index[::-1]], cornerradius=4),
        text=[f"{v:.1f}%" for v in shares.iloc[::-1]], textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=INK),
        customdata=top.iloc[::-1].to_numpy().reshape(-1, 1),
        hovertemplate="%{y}<br>£%{customdata[0]:,.2f}<br>%{x:.1f}% of selected revenue<extra></extra>",
        width=.56,
    )
    fig.update_xaxes(range=[0, max(10, shares.max() * 1.26)], visible=False)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=12, color=INK))
    fig.update_layout(margin=dict(l=115, r=42, t=24, b=28))
    return fig


def product_figure(sales: pd.DataFrame, mode: str) -> go.Figure:
    if sales.empty:
        return blank_figure("No products in this selection", 420)
    fig = base_figure(420)
    if mode == "category":
        top = sales.groupby("Category").Revenue.sum().sort_values(ascending=False).iloc[::-1]
        labels = [str(x).replace(" & ", " &<br>") for x in top.index]
        values, custom = top.tolist(), [[str(n), "Estimated theme"] for n in top.index]
        colors = [AMBER if n == "Other / General Gift" else OLIVE for n in top.index]
        tickvals = list(range(len(top)))
    else:
        group = sales.groupby("StockCode", observed=True).agg(
            Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Name=("Description", "first"))
        col = "Units" if mode == "units" else "Revenue"
        top = group.nlargest(8, col).iloc[::-1]
        labels = [str(n).title()[:24] + ("…" if len(str(n)) > 24 else "") for n in top.Name]
        values = top[col].tolist()
        custom = [[str(n).title(), f"SKU {sku}"] for sku, n in zip(top.index, top.Name)]
        colors, tickvals = [OLIVE] * len(top), list(range(len(top)))
    fig.add_bar(
        x=values, y=tickvals, orientation="h", customdata=custom,
        marker=dict(color=colors, cornerradius=4), width=.52,
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>" +
                      ("Units: %{x:,.0f}" if mode == "units" else "Purchase revenue: £%{x:,.2f}") + "<extra></extra>",
    )
    fig.update_xaxes(tickprefix="£" if mode != "units" else "", tickformat="~s",
                     showgrid=True, gridcolor=GRID, nticks=5)
    fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=labels,
                     showgrid=False, automargin=True, tickfont=dict(color=INK, size=12))
    fig.update_layout(margin=dict(l=188, r=26, t=16, b=44))
    return fig


def rate_figure(c: pd.DataFrame, lens: str) -> go.Figure:
    if c.empty:
        return blank_figure("No eligible customers in this market", 380)
    col, labels = {
        "orders": ("OrderBin", ORDER_LABELS),
        "recency": ("RecencyBin", RECENCY_LABELS),
        "spend": ("SpendBin", SPEND_LABELS),
    }.get(lens, ("OrderBin", ORDER_LABELS))
    summary = c.groupby(col, observed=False).Repeat.agg(["mean", "size"]).reindex(labels)
    baseline = c.Repeat.mean()
    fig = base_figure(380)
    fig.add_bar(
        x=list(range(len(labels))), y=summary["mean"].tolist(),
        marker=dict(color=[OLIVE if pd.notna(rate) and rate >= baseline else RUST for rate in summary["mean"]], cornerradius=5),
        customdata=[[str(label), int(n)] for label, n in zip(labels, summary["size"])],
        text=[f"{v:.0%}" if n else "" for v, n in zip(summary["mean"], summary["size"])],
        textposition="outside", textfont=dict(size=14, color=INK), cliponaxis=False,
        hovertemplate="%{customdata[0]}<br>Purchased again: %{y:.1%}<br>Customers: %{customdata[1]:,}<extra></extra>",
    )
    fig.add_hline(y=baseline, line_dash="dot", line_color="#97836c", line_width=1.5)
    fig.update_yaxes(range=[0, 1.12], tickmode="array", tickvals=[0, .25, .5, .75, 1], tickformat=".0%")
    display_labels = [str(x).replace("–", "–<br>") if lens == "spend" else str(x) for x in labels]
    fig.update_xaxes(
        tickmode="array", tickvals=list(range(len(labels))),
        ticktext=[f"{label}<br><span style='font-size:10px;color:{MUTED}'>n={int(n):,}</span>"
                  for label, n in zip(display_labels, summary["size"])],
        range=[-.65, len(labels) - .35], tickfont=dict(size=12),
    )
    fig.update_layout(margin=dict(l=46, r=18, t=25, b=70))
    return fig


def heatmap_figure(c: pd.DataFrame) -> go.Figure:
    if c.empty:
        return blank_figure("No eligible customers in this market", 380)
    grouped = c.groupby(["RecencyBin", "OrderBin"], observed=False).Repeat.agg(["mean", "size"])
    idx = pd.MultiIndex.from_product([RECENCY_LABELS, ORDER_LABELS], names=["RecencyBin", "OrderBin"])
    grouped = grouped.reindex(idx)
    rate = grouped["mean"].unstack().reindex(index=RECENCY_LABELS, columns=ORDER_LABELS)
    counts = grouped["size"].unstack().reindex(index=RECENCY_LABELS, columns=ORDER_LABELS).fillna(0).astype(int)
    shown = rate.where(counts >= 20)
    fig = base_figure(380)
    # A neutral backing layer keeps small-sample cells visible without implying 0%.
    fig.add_trace(go.Heatmap(
        z=np.zeros((5, 5)), x=ORDER_LABELS, y=RECENCY_LABELS,
        colorscale=[[0, "#f3f0e9"], [1, "#f3f0e9"]], showscale=False,
        xgap=7, ygap=7, customdata=counts.to_numpy(),
        text=[[f"n={counts.iloc[i, j]}" if counts.iloc[i, j] < 20 else "" for j in range(5)] for i in range(5)],
        texttemplate="%{text}", textfont=dict(size=11, color=MUTED),
        hovertemplate="Prior orders: %{x}<br>Recency: %{y} days<br>Customers: %{customdata:,}<br>Rate hidden below 20 customers<extra></extra>",
    ))
    fig.add_trace(go.Heatmap(
        z=shown.to_numpy(), x=ORDER_LABELS, y=RECENCY_LABELS,
        zmin=0, zmax=1, colorscale=[[0, "#fbedd6"], [.45, "#eac790"], [.7, "#b4bc8c"], [1, "#66835b"]],
        xgap=7, ygap=7,
        text=[[(f"<b>{shown.iloc[i, j]:.0%}</b>" if pd.notna(shown.iloc[i, j]) else "")
              for j in range(5)] for i in range(5)],
        texttemplate="%{text}", textfont=dict(size=12, color=INK), customdata=counts.to_numpy(),
        hovertemplate="Prior orders: %{x}<br>Recency: %{y} days<br>Repeat rate: %{z:.1%}<br>Customers: %{customdata:,}<extra></extra>",
        hoverongaps=False,
        colorbar=dict(thickness=9, len=.86, outlinewidth=0, tickformat=".0%", tickvals=[0,.5,1], tickfont=dict(size=11)),
    ))
    fig.update_xaxes(type="category", title=dict(text="Orders before the cutoff", font=dict(size=11, color=MUTED)), tickfont=dict(size=12))
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False, tickfont=dict(size=12))
    fig.update_layout(margin=dict(l=57, r=30, t=12, b=54))
    return fig


def graph(graph_id: str, height: int = 350) -> dcc.Graph:
    # The DOM owns height. Plotly may resize width, but it cannot collapse the chart
    # when a page is hidden or an update changes the figure.
    return dcc.Graph(
        id=graph_id, className="rj-graph", responsive=True,
        style={"height": f"{height}px", "minHeight": f"{height}px", "width": "100%", "flex": "0 0 auto"},
        config={"displayModeBar":False, "displaylogo":False, "scrollZoom":False, "responsive":True},
    )


def switch(control_id: str, choices: list[tuple[str, str]], value: str, nav: bool = False) -> dcc.RadioItems:
    return dcc.RadioItems(id=control_id,
        options=[{"label":label, "value":val} for label, val in choices], value=value,
        className="rj-switch" + (" rj-navigation" if nav else ""), inline=True)


def panel(title: str, caption, chart, controls=None, foot=None) -> html.Section:
    return html.Section([
        html.Div([html.Div([html.H3(title), html.Div(caption, className="rj-caption")]), controls], className="rj-panel-head"),
        chart,
        html.Div(foot, className="rj-panel-foot") if foot is not None else None,
    ], className="rj-panel")


CSS = """
:root{--rj-paper:#fff;--rj-page:#faf8f2;--rj-ink:#332e25;--rj-muted:#807567;--rj-line:#ebe5db;--rj-rust:#bd623c;--rj-olive:#74895d;
 --Dash-Fill-Interactive-Strong:#bd623c;--Dash-Fill-Interactive-Weak:#fff3e3;--Dash-Stroke-Strong:#bd623c;--Dash-Text-Primary:#332e25;--Dash-Text-Strong:#332e25}
body{margin:0;background:var(--rj-page);color:var(--rj-ink);font-family:'Segoe UI',Arial,sans-serif;font-size:14px;line-height:1.5}
.rj-app,.rj-app div,.rj-app section,.rj-app header,.rj-app p,.rj-app button,.rj-app label{box-sizing:border-box}
.rj-app{max-width:1440px;margin:0 auto;padding:30px 38px 44px}
.rj-header{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;align-items:center;padding-bottom:22px;border-bottom:1px solid var(--rj-line)}
.rj-brand{display:flex;align-items:center;gap:10px;font-size:10px;letter-spacing:.18em;font-weight:750;color:#927357}
.rj-mark{display:flex;gap:3px;align-items:flex-end;width:21px;height:21px}.rj-mark span{width:5px;background:#bd623c;border-radius:2px}.rj-mark span:nth-child(1){height:10px}.rj-mark span:nth-child(2){height:20px;background:#e2b671}.rj-mark span:nth-child(3){height:15px;background:#74895d}
.rj-header h1{font-family:Georgia,serif;font-size:38px;line-height:1.15;font-weight:normal;letter-spacing:-1.1px;margin:12px 0 5px}.rj-header h1 em{font-style:normal;color:var(--rj-rust)}
.rj-subtitle{color:var(--rj-muted);font-size:13px;margin:0}.rj-edition{font-size:11px;color:var(--rj-muted);text-align:right;line-height:1.9}.rj-edition strong{color:var(--rj-ink);font-weight:650;display:block}
.rj-toolbar{display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;padding:19px 0}
.rj-scope{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.rj-scope-label{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--rj-muted)}
.rj-switch{display:flex;flex-wrap:wrap;gap:3px;align-items:center;border:1px solid #e9e1d5;background:#f3efe7;border-radius:8px;padding:3px;width:max-content;max-width:100%}
.rj-switch label{position:relative;display:inline-flex!important;align-items:center;justify-content:center;min-height:31px;padding:6px 11px!important;font-size:11px;line-height:1.3;font-weight:600;border-radius:5px;cursor:pointer;color:#776956;transition:background .15s,color .15s}
.rj-switch label:has(input:checked){background:#fff;box-shadow:0 1px 4px #4d382a14;color:#493123}
.rj-switch input{position:absolute;opacity:0;width:1px;height:1px;margin:0}.rj-switch label:has(input:focus-visible){outline:2px solid var(--rj-rust);outline-offset:2px}
.rj-navigation{background:transparent;border:0;padding:0;gap:6px}.rj-navigation label{padding:10px 19px!important;min-height:41px;font-size:13px;border-radius:8px}.rj-navigation label:has(input:checked),.rj-navigation label:has(input:checked):hover{background:#4f3c2f;color:white!important;box-shadow:none}
.rj-reset{background:none;border:0;color:#8c6e54;text-decoration:underline;text-underline-offset:4px;font-size:11px;padding:9px 3px;cursor:pointer;white-space:nowrap}
.rj-page{min-width:0}.rj-page-title{display:flex;gap:12px;align-items:center;margin:8px 0 14px}.rj-page-number{display:grid;place-items:center;width:31px;height:31px;border-radius:50%;background:#f5e6cc;color:#966330;font-size:11px;font-weight:700;flex:none}
.rj-page-title h2{font-size:19px;margin:0;font-weight:600;letter-spacing:-.3px}.rj-page-title p{color:var(--rj-muted);font-size:12px;margin:2px 0 0}
.rj-period{display:grid;grid-template-columns:210px minmax(0,1fr);align-items:center;gap:25px;background:#fff;border:1px solid var(--rj-line);border-radius:10px;padding:17px 22px 14px;margin-bottom:20px;min-width:0}
.rj-period-label{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--rj-muted);margin-bottom:4px}.rj-period-value{font-size:13px;font-weight:650;white-space:nowrap}.rj-range{min-width:0;padding:0 10px 2px}
.rj-app .rc-slider-track{background:#d8a462}.rj-app .rc-slider-rail{background:#eee8dd}.rj-app .rc-slider-handle{border-color:#bd623c;background:#fff;box-shadow:none}.rj-app .rc-slider-dot-active{border-color:#d8a462}.rj-app .rc-slider-mark-text{font-size:10px;color:#807567}
.rj-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--rj-line);border-radius:12px;background:#fff;margin:0 0 20px;overflow:hidden}
.rj-stats.rj-three{grid-template-columns:repeat(3,minmax(0,1fr))}.rj-stat{position:relative;padding:21px 24px;min-width:0}.rj-stat+.rj-stat{border-left:1px solid var(--rj-line)}
.rj-stat-label{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--rj-muted);font-weight:650;margin:0 0 8px}.rj-stat-value{font-size:32px;line-height:1.1;letter-spacing:-1px;font-variant-numeric:tabular-nums;margin:0 0 7px;font-weight:600}.rj-stat-detail{font-size:11px;color:var(--rj-muted);margin:0}.rj-stat.rj-feature .rj-stat-value{color:var(--rj-rust)}.rj-stat.rj-green .rj-stat-value{color:var(--rj-olive)}
.rj-columns{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:18px;margin-bottom:18px;align-items:start}.rj-columns.rj-equal{grid-template-columns:repeat(2,minmax(0,1fr))}
.rj-panel{border:1px solid var(--rj-line);border-radius:12px;background:white;min-width:0;padding:19px 20px 10px;display:flex;flex-direction:column;height:auto;box-shadow:0 3px 10px #4c3a2710}
.rj-panel-head{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;min-height:69px;padding:0 3px 9px}.rj-panel h3{font-size:16px;line-height:1.35;letter-spacing:-.25px;font-weight:600;margin:0 0 5px}.rj-caption{font-size:11px;line-height:1.55;color:var(--rj-muted);max-width:590px}.rj-panel .rj-switch{flex:none}
.rj-equal .rj-panel-head{min-height:104px}
.rj-graph{position:relative;display:block;width:100%;min-width:0;flex-shrink:0}.rj-panel-foot{font-size:11px;color:var(--rj-muted);padding:8px 3px 5px;min-height:29px;border-top:1px solid #f5f1ea}
.rj-legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.rj-legend span{display:flex;align-items:center;gap:6px}.rj-dot{display:inline-block;width:7px;height:7px;background:var(--rj-rust);border-radius:2px;flex:none}.rj-dot.rj-amber{background:#ddb66e}.rj-dot.rj-green{background:var(--rj-olive)}.rj-dash{display:inline-block;border-top:2px dotted #97836c;width:17px}
.rj-story{margin:18px 0 23px;padding:16px 20px;border-radius:9px;background:#f8efdc;display:grid;grid-template-columns:145px 1fr;gap:18px;color:#6b563b;font-size:12px;line-height:1.65}.rj-story-label{font-size:10px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;padding-top:2px}.rj-story p{margin:0}.rj-story strong{color:#4d3b28;font-weight:650}
.rj-timeline{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:0 0 18px}.rj-time-cell{background:#fff;border:1px solid var(--rj-line);border-radius:9px;padding:12px 17px;display:flex;align-items:center;gap:13px;font-size:12px;color:#6e6253}.rj-time-tag{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;background:#f6ead4;padding:5px 8px;border-radius:4px;color:#986b32;white-space:nowrap}.rj-time-tag.rj-result{background:#eaf0e1;color:#5f7946}
.rj-footer{margin-top:26px;padding-top:18px;border-top:1px solid var(--rj-line);display:flex;justify-content:space-between;gap:24px;color:var(--rj-muted);font-size:11px}.rj-footer details{max-width:850px}.rj-footer summary{cursor:pointer;color:#695847;font-weight:600}.rj-footer details p{line-height:1.75;margin:10px 0}.rj-footer a{color:#956037}.rj-footer-note{white-space:nowrap}
@media(max-width:1150px){.rj-columns,.rj-columns.rj-equal{grid-template-columns:1fr}.rj-panel-head{min-height:60px}.rj-equal .rj-panel-head{min-height:82px}.rj-header h1{font-size:34px}}
@media(max-width:780px){.rj-app{padding:20px 19px 35px}.rj-header{grid-template-columns:1fr}.rj-edition{display:none}.rj-toolbar{gap:14px}.rj-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.rj-stat:nth-child(3){border-left:0;border-top:1px solid var(--rj-line)}.rj-stat:nth-child(4){border-top:1px solid var(--rj-line)}.rj-stat{padding:18px}.rj-stat-value{font-size:28px}.rj-period{grid-template-columns:1fr;gap:18px;padding:16px}.rj-timeline{grid-template-columns:1fr;gap:8px}.rj-story{grid-template-columns:1fr;gap:6px}.rj-footer{display:block}.rj-footer-note{margin-top:14px}.rj-panel{padding:17px 12px 10px}}
@media(max-width:480px){.rj-app{padding:15px 10px 28px}.rj-header h1{font-size:32px}.rj-scope{gap:7px}.rj-scope-label{width:100%}.rj-navigation label{padding:9px 14px!important}.rj-three .rj-stat{padding:14px 10px}.rj-three .rj-stat:nth-child(3){border-left:1px solid var(--rj-line);border-top:0}.rj-three .rj-stat-value{font-size:25px}.rj-three .rj-stat-label{font-size:9px;letter-spacing:.03em;min-height:27px}.rj-three .rj-stat-detail{font-size:10px}.rj-panel-head{gap:9px;min-height:78px}.rj-panel .rj-switch label{padding:6px 8px!important;font-size:10px}}
@media(prefers-reduced-motion:reduce){.rj-switch label{transition:none}}
"""


def build_app(p: pd.DataFrame, audit: dict) -> Dash:
    cohort = make_cohort(p)
    date_min = pd.Timestamp(audit["first_date"])
    date_max = pd.Timestamp(audit["last_date"])
    months = pd.date_range(date_min.to_period("M").to_timestamp(), date_max.to_period("M").to_timestamp(), freq="MS")
    last_month = len(months) - 1
    marks = {i: m.strftime("%b '%y") for i, m in enumerate(months) if i % 3 == 0 or i == last_month}
    slider_options = {"allow_direct_input": False} if "allow_direct_input" in dcc.RangeSlider()._prop_names else {}
    app = Dash(__name__, title="Online Retail | The retail picture", update_title=None)
    app.index_string = "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}<style>" + CSS + "</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
    sales_metrics = [card("Purchase revenue", "…", "Positive product purchases", "rj-feature"),
                     card("Orders", "…", "Distinct purchase invoices"),
                     card("Customers", "…", "Identified buyers in selection"),
                     card("Revenue / order", "…", "Selected revenue ÷ orders")]
    repeat_metrics = [card("Existing customers", "…", "History before the cutoff"),
                      card("Purchased again", "…", "In the outcome window"),
                      card("Repeat rate", "…", "Returned ÷ existing customers", "rj-green")]
    sales_page = html.Div([
        html.Div([html.Span("01", className="rj-page-number"),
                  html.Div([html.H2("Sales overview"), html.P("Follow the money, the markets, and the products.")])], className="rj-page-title"),
        html.Div([
            html.Div([html.Div("Sales period", className="rj-period-label"), html.Div(id="period-label", className="rj-period-value")]),
            html.Div(dcc.RangeSlider(id="months", min=0, max=last_month, step=1, value=[0,last_month],
                marks=marks, allowCross=False, updatemode="mouseup", **slider_options), className="rj-range"),
        ], className="rj-period"),
        html.Div(sales_metrics, id="sales-kpis", className="rj-stats"),
        html.Div([
            panel("Purchases through the year", "Move the period handles to focus on a few months.", graph("trend"),
                  switch("trend-mode", [("Revenue","revenue"),("Orders","orders")], "revenue"),
                  html.Div([html.Span([html.I(className="rj-dot"),"Full month"]),
                            html.Span([html.I(className="rj-dot rj-amber"),"Partial month"])], className="rj-legend")),
            panel("Where purchases came from", "Share of revenue in the selected market and period.", graph("geography"),
                  foot="Countries are based on the transaction record."),
        ], className="rj-columns"),
        html.Div([html.Div("In this selection", className="rj-story-label"), html.P(id="sales-note")], className="rj-story"),
        panel("The products behind the purchases", html.Span(id="product-caption"), graph("products",420),
              switch("product-mode",[("Revenue","sku"),("Units","units"),("Estimated themes","category")],"sku"),
              "Hover over a bar for the full product description and SKU."),
    ], id="sales-page", className="rj-page")
    retention_page = html.Div([
        html.Div([html.Span("02", className="rj-page-number"),
                  html.Div([html.H2("The repeat-purchase story"), html.P("An observed outcome for customers who already had purchase history.")])], className="rj-page-title"),
        html.Div([
            html.Div([html.Span("History",className="rj-time-tag"),html.Span("1 Dec 2010 – 8 Sep 2011")],className="rj-time-cell"),
            html.Div([html.Span("Outcome",className="rj-time-tag rj-result"),html.Span("9 Sep 2011 – 9 Dec 2011")],className="rj-time-cell"),
        ],className="rj-timeline"),
        html.Div(repeat_metrics,id="repeat-kpis",className="rj-stats rj-three"),
        html.Div([
            panel("Which customers came back?", "Group customers by their behavior before the cutoff.",graph("rate-chart",380),
                  switch("lens",[("Prior orders","orders"),("Recency","recency"),("Prior spend","spend")],"orders"),
                  html.Div([html.I(className="rj-dash"),html.Span(id="rate-baseline")],className="rj-legend")),
            panel("Recency meets frequency", "Rows show days since the last purchase; columns show prior orders.",graph("heatmap",380),
                  foot="Hover for customer counts. Neutral cells have fewer than 20 customers; rates are hidden."),
        ],className="rj-columns rj-equal"),
        html.Div([html.Div("Read it this way",className="rj-story-label"),html.P(id="repeat-note")],className="rj-story"),
    ],id="retention-page",className="rj-page",style={"display":"none"})
    app.layout = html.Main([
        html.Header([
            html.Div([
                html.Div([html.Span([html.Span(),html.Span(),html.Span()],className="rj-mark"),"ONLINE RETAIL / ANALYTICS"],className="rj-brand"),
                html.H1(["The retail ",html.Em("picture.")]),
                html.P("What sold. Where it went. Who came back.",className="rj-subtitle"),
            ]),
            html.Div([html.Strong("A YEAR IN PURCHASES"),html.Div("December 2010 — December 2011"),html.Div("UK gift retailer · amounts in GBP")],className="rj-edition"),
        ],className="rj-header"),
        html.Div([
            switch("view",[("01  Sales overview","sales"),("02  Repeat purchase","retention")],"sales",nav=True),
            html.Div([html.Span("Market",className="rj-scope-label"),
                      switch("market",[("All countries","all"),("UK","uk"),("Outside UK","outside")],"all"),
                      html.Button("Reset",id="reset",n_clicks=0,className="rj-reset")],className="rj-scope"),
        ],className="rj-toolbar"),
        sales_page,retention_page,
        html.Footer([
            html.Details([
                html.Summary("Data notes & definitions"),
                html.P("Purchase revenue sums quantity × unit price on positive, identified, non-cancelled product purchases. "
                       "Returns, discounts, postage, service lines and anonymous purchases are excluded. It is not an accounting net-sales or profit figure."),
                html.P(f"Cleaning: {audit['raw_rows']:,} source lines; {audit['missing_ids']:,} lines without customer ID; "
                       f"{audit['duplicates']:,} exact duplicates removed; {audit['included_rows']:,} purchase lines retained. "
                       "Orders are distinct invoice numbers. The market filter uses each transaction's country for sales and the most frequent pre-cutoff country for customers."),
                html.P(f"Product themes are keyword estimates. {audit['named_sku_share']:.1%} of SKUs match a named theme. "
                       "'Other / General Gift' is a catch-all. Recency counts whole elapsed days before 9 Sep 2011. "
                       "The repeat denominator includes customers with at least one pre-cutoff purchase, and repeat means at least one new purchase in the outcome window."),
                html.P("The original model used a customer train/test split within one historical snapshot. Its results do not prove a fixed performance ceiling, a campaign effect, or later-period performance. "
                       "Customer groups here use explicit ranges, avoiding arbitrary splitting of tied customers into RFM quartiles."),
                html.P(["Dataset: Chen, D. (2015), ",html.A("UCI Online Retail",href=SOURCE_URL,target="_blank"),
                        ". DOI 10.24432/C5BW33. CC BY 4.0."]),
            ]),
            html.Div("ONLINE RETAIL · 2010–2011",className="rj-footer-note"),
        ],className="rj-footer"),
    ],className="rj-app")

    # Keep both pages mounted so filter and reset callbacks always have every
    # component present. Resize after visibility changes, without re-creating charts.
    app.clientside_callback("""
        function(view) {
            window.setTimeout(function() {window.dispatchEvent(new Event('resize'));}, 90);
            return [view === 'sales' ? {display:'block'} : {display:'none'},
                    view === 'retention' ? {display:'block'} : {display:'none'}];
        }
    """,Output("sales-page","style"),Output("retention-page","style"),Input("view","value"))

    @app.callback(Output("market","value"),Output("months","value"),Output("trend-mode","value"),
                  Output("product-mode","value"),Output("lens","value"),Input("reset","n_clicks"),prevent_initial_call=True)
    def reset_view(_clicks):
        return "all",[0,last_month],"revenue","sku","orders"

    @app.callback(Output("sales-kpis","children"),Output("trend","figure"),Output("geography","figure"),
                  Output("products","figure"),Output("product-caption","children"),Output("sales-note","children"),
                  Output("period-label","children"),Input("market","value"),Input("months","value"),
                  Input("trend-mode","value"),Input("product-mode","value"))
    def update_sales(market, month_range, trend_mode, product_mode):
        lo,hi = month_range or [0,last_month]
        lo,hi = max(0,min(int(lo),last_month)),max(0,min(int(hi),last_month))
        lo,hi = min(lo,hi),max(lo,hi)
        start,end = max(months[lo],date_min),min(months[hi]+pd.offsets.MonthEnd(0),date_max)
        sales = market_filter(p,market)
        sales = sales.loc[(sales.InvoiceDate>=start)&(sales.InvoiceDate<end+pd.Timedelta(days=1))]
        revenue = float(sales.Revenue.sum())
        orders,buyers = sales.InvoiceNo.nunique(),sales.CustomerID.nunique()
        stats = [card("Purchase revenue",money(revenue),"Positive product purchases","rj-feature"),
                 card("Orders",f"{orders:,}","Distinct purchase invoices"),
                 card("Customers",f"{buyers:,}","Identified buyers in selection"),
                 card("Revenue / order",money(revenue/orders) if orders else "—","Selected revenue ÷ orders")]
        caption = (f"Exploratory themes: {1-audit['named_sku_share']:.1%} of SKUs have no named match."
                   if product_mode=="category" else "Eight leading product SKUs in the selected market and period.")
        if not revenue:
            note = "No included purchases were found for this selection. Widen the period or change the market."
        else:
            top_country = sales.groupby("Country").Revenue.sum().sort_values(ascending=False)
            if market=="uk":
                n_products=sales.StockCode.nunique()
                note=[html.Strong(f"{n_products:,} different products "),"were purchased by identified UK customers in this period. ",
                      "The totals include positive product purchases; returned goods and discounts are excluded."]
            else:
                note=[html.Strong(f"{top_country.index[0]} contributes {top_country.iloc[0]/revenue:.1%} "),
                      "of this selection's purchase revenue. ","A partial final month should not be read as a full-month fall in sales."]
        period=months[lo].strftime("%b %Y") if lo==hi else f"{months[lo]:%b %Y} – {months[hi]:%b %Y}"
        return stats,trend_figure(sales,start,end,trend_mode),geography_figure(sales),product_figure(sales,product_mode),caption,note,period

    @app.callback(Output("repeat-kpis","children"),Output("rate-chart","figure"),Output("heatmap","figure"),
                  Output("repeat-note","children"),Output("rate-baseline","children"),
                  Input("market","value"),Input("lens","value"))
    def update_repeat(market,lens):
        c=market_filter(cohort,market)
        n,returned=len(c),int(c.Repeat.sum())
        rate=returned/n if n else 0
        stats=[card("Existing customers",f"{n:,}","History before the cutoff"),
               card("Purchased again",f"{returned:,}","In the outcome window"),
               card("Repeat rate",f"{rate:.1%}" if n else "—","Returned ÷ existing customers","rj-green")]
        one,many=c.loc[c.PriorOrders==1,"Repeat"],c.loc[c.PriorOrders>=5,"Repeat"]
        if len(one)>=20 and len(many)>=20:
            note=[html.Strong(f"{one.mean():.0%} of one-order customers "),f"(n={len(one):,}) and ",
                  html.Strong(f"{many.mean():.0%} of customers with 5+ orders "),f"(n={len(many):,}) purchased again. ",
                  "This shows an association with prior behavior; it does not show that a campaign caused customers to return."]
        else:
            note="Some groups in this market are small. Read the customer counts with the percentages; sparse heatmap cells intentionally show only their sample size."
        return stats,rate_figure(c,lens),heatmap_figure(c),note,f"Market average: {rate:.1%} · n means number of customers"
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Dash dashboard for UCI Online Retail.")
    parser.add_argument("--data", help="Path to the original Online Retail .xlsx or equivalent .csv")
    parser.add_argument("--port", type=int, default=8050, help="Local port (default: 8050)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard tab automatically")
    args = parser.parse_args()
    data_path = resolve_data(args.data)
    purchases, audit = load_purchases(data_path)
    app = build_app(purchases, audit)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Open {url} (Ctrl+C to stop).")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open_new_tab(url)).start()
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
