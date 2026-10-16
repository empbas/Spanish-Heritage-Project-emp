"""
Hispanic Heritage Historical Data Explorer
A Streamlit learning app for World Bank data, polynomial modeling, and function analysis.
"""
from __future__ import annotations

import io
import math
import re
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

# PINK THEME
st.markdown("""
<style>
/* Main page background */
.stApp {
    background-color: #FFF0F5;
    color: #54283D;
}

/* Main content area */
section[data-testid="stMain"] {
    background-color: #FFF0F5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F8C8DC;
}

section[data-testid="stSidebar"] > div {
    background-color: #F8C8DC;
}

/* Headings and text */
h1, h2, h3, h4, p, label,
[data-testid="stMarkdownContainer"] {
    color: #713B55;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #D96C9D;
    color: white;
    border: 1px solid #C65386;
    border-radius: 12px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #BD4F82;
    color: white;
    border-color: #A93D70;
}

/* Input boxes and dropdowns */
.stTextInput input,
.stNumberInput input,
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    background-color: #FFF8FB;
    border-color: #E7A1BF;
}

/* Sliders and selected controls */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #C65386;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FADBE8;
    border: 1px solid #E7A1BF;
    padding: 16px;
    border-radius: 14px;
}

/* Tables and data editors */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    background-color: #FFF8FB;
    border: 1px solid #E7A1BF;
    border-radius: 12px;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFE3EE;
    border: 1px solid #E7A1BF;
    border-radius: 12px;
}

/* Information, success, and warning messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Horizontal dividers */
hr {
    border-color: #E7A1BF;
}

/* Links */
a {
    color: #A83269 !important;
}

/* Selection highlight */
::selection {
    background-color: #F2A9C9;
    color: #54283D;
}
</style>
""", unsafe_allow_html=True)

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if mortality patterns stayed the same."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment; national estimate."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population, expressed in current US dollars; not household income or average personal wealth."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net total)", "Net number of migrants over the period; it is not a count of emigration alone. Values can be negative."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "deaths per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, expressed in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP."),
}
CONTEXTS = {
    "Business": ("Revenue", "US dollars", "Time", "years", "A business model can describe how revenue changes over time."),
    "Engineering": ("Position", "meters", "Time", "seconds", "Position models describe location; derivatives can represent velocity and acceleration."),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "A concentration curve may describe a substance level over time; it is illustrative, not medical advice."),
    "Sports science": ("Position", "meters", "Time", "seconds", "Position and performance models can be studied through rates of change."),
    "Biology": ("Population", "organisms", "Time", "days", "Population models describe a changing count or estimate."),
    "Bioengineering": ("Signal", "arbitrary units", "Time", "seconds", "A signal model can describe changing sensor output."),
    "Economic forecasting": ("Indicator", "indicator units", "Time", "years", "Economic forecasts are uncertain and should not be treated as guaranteed outcomes."),
    "Weather": ("Temperature", "°C", "Time", "hours", "Temperature models describe change; real weather is more complex."),
    "Custom": ("Output", "output units", "Input", "input units", "Describe your own model and its assumptions."),
}
SUGGESTIONS = {
    "Business": "0.5*x**2 + 4*x + 10",
    "Engineering": "2*x**2 - 3*x + 1",
    "Medical": "100*exp(-0.25*x)",
    "Sports science": "sin(x) + 0.1*x",
    "Biology": "10*exp(0.15*x)",
    "Bioengineering": "sin(2*x)",
    "Economic forecasting": "0.2*x**2 + 2*x + 50",
    "Weather": "10 + 8*sin(2*pi*x/24)",
    "Custom": "x**2 - 4*x + 3",
}
WB_BASE = "https://api.worldbank.org/v2"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"{WB_BASE}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), "The API returned an empty or unexpected response."
        records = []
        for item in payload[1]:
            if item and item.get("date") and item.get("value") is not None:
                try:
                    records.append({"year": int(item["date"]), "value": float(item["value"])})
                except (TypeError, ValueError):
                    continue
        df = pd.DataFrame(records, columns=["year", "value"]).sort_values("year") if records else pd.DataFrame(columns=["year", "value"])
        return df, None if not df.empty else "The API returned no non-null observations for this period."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "The World Bank API request timed out. Try again later."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not parse the API response: {exc}"

def validate_data(df: pd.DataFrame, indicator_name: str):
    errors = []
    if df.empty:
        return ["There are no observations to validate."]
    if not {"year", "value"}.issubset(df.columns):
        return ["Table must contain year and value columns."]
    years = pd.to_numeric(df["year"], errors="coerce")
    values = pd.to_numeric(df["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Every year must be a whole-number integer.")
    if years.duplicated().any():
        errors.append("Duplicate years are not allowed.")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
        errors.append("Every value must be a finite number.")
    if indicator_name in ("Life expectancy at birth", "Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate") and (values < 0).any():
        errors.append("This indicator cannot contain negative values.")
    if indicator_name == "Life expectancy at birth" and (values > 125).any():
        errors.append("Life expectancy above 125 years should be checked.")
    if indicator_name == "Unemployment, national estimate" and (values > 100).any():
        errors.append("Unemployment percentage cannot exceed 100.")
    return errors

def fit_polynomial(df: pd.DataFrame, degree: int):
    data = df[["year", "value"]].copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna().drop_duplicates("year").sort_values("year")
    if len(data) < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; only {len(data)} are available.")
    base = int(data["year"].min())
    t = data["year"].to_numpy(dtype=float) - base
    y = data["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    model = np.poly1d(coeff)
    pred = model(t)
    return data, base, coeff, model, pred

def poly_text(coeff, base):
    terms = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        coeff_text = f"{c:.8g}"
        if power == 0:
            term = coeff_text
        elif power == 1:
            term = f"{coeff_text}(year − {base})"
        else:
            term = f"{coeff_text}(year − {base})^{power}"
        terms.append(term)
    return " + ".join(terms).replace("+ -", "− ") if terms else "0"

def safe_expression(source: str):
    """Parse a student-entered expression in x using a limited SymPy namespace."""
    allowed = {"x": sp.Symbol("x"), "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi,
               "Abs": sp.Abs}
    if len(source) > 180:
        raise ValueError("Expression is too long (maximum 180 characters).")
    if not re.fullmatch(r"[0-9xX+\-*/().,\s^A-Za-z_]+", source):
        raise ValueError("Use x, numbers, arithmetic, parentheses, and supported functions only.")
    source = source.replace("^", "**").replace("X", "x")
    expr = sp.sympify(source, locals=allowed, evaluate=True)
    if expr.free_symbols - {allowed["x"]}:
        raise ValueError("Only x is allowed as a variable.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Expression contains an undefined or infinite constant.")
    return expr

def numeric_fn(expr):
    return sp.lambdify(sp.Symbol("x"), expr, modules=["numpy"])

def fmt_num(v, digits=6):
    try:
        v = float(v)
        return f"{v:.{digits}g}" if math.isfinite(v) else "undefined"
    except (TypeError, ValueError):
        return "undefined"

def make_chart(data, base, model, sample_step, prediction_years=None, query_years=None):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(data["year"], data["value"], label="Observed data", color="#1769aa", s=28, zorder=3)
    lo, hi = int(data["year"].min()), int(data["year"].max())
    smooth_x = np.linspace(lo, max(hi, prediction_years or hi), 700)
    smooth_y = model(smooth_x - base)
    inside = smooth_x <= hi
    ax.plot(smooth_x[inside], smooth_y[inside], label="Polynomial fit (historical range)", color="#e87500", linewidth=2)
    if (~inside).any():
        ax.plot(smooth_x[~inside], smooth_y[~inside], label="Extrapolated segment (uncertain)", color="#b22222", linestyle="--", linewidth=2)
    sampled = data.iloc[::sample_step]
    ax.scatter(sampled["year"], sampled["value"], marker="x", label=f"Sample markers (every {sample_step} row(s))", color="#3b7a57", s=45)
    if query_years:
        qx = np.array(sorted(set(query_years)), dtype=float)
        qy = model(qx - base)
        ax.scatter(qx, qy, marker="D", label="Calculated estimates", color="#7d3c98", s=42, zorder=4)
    ax.axvline(hi, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Indicator value (source units)")
    ax.set_title("Historical observations and polynomial model")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def make_pdf(title, details, chart_png, preview, calculated):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=32, leftMargin=32, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in details:
        story.append(Paragraph(str(line).replace("&", "&amp;"), styles["BodyText"]))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))
    story.append(RLImage(io.BytesIO(chart_png), width=650, height=330))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    prev = preview.head(12).copy()
    table_data = [list(prev.columns)] + prev.astype(str).values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e8edf3")),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(t)
    if calculated is not None and not calculated.empty:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Calculated-year results", styles["Heading2"]))
        tt = Table([list(calculated.columns)] + calculated.astype(str).values.tolist()[:25], repeatRows=1)
        tt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8edf3")),("GRID",(0,0),(-1,-1),.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(tt)
    doc.build(story)
    return buffer.getvalue()

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB · Authentic historical data, model fitting, and function analysis")
st.info("Research question: How has the selected indicator changed over time, and what can a mathematical model describe without claiming to know the future? Brazil is included as a Latin American comparison; Latin America is diverse, so it is not a stand-in for Hispanic identity or experience.")

with st.sidebar:
    st.header("Data controls")
    country_name = st.selectbox("Featured country", list(COUNTRIES))
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS))
    comparison_names = st.multiselect("Comparison countries", [x for x in COUNTRIES if x != country_name], default=["Brazil (BR)"] if country_name != "Brazil (BR)" else ["Mexico (MX)"])
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (rows)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

current_year = datetime.now().year
start_year, end_year = current_year - 70, current_year - 1
code, units, definition = INDICATORS[indicator_name]
st.caption(f"Requested period: {start_year}–{end_year} inclusive · Retrieval date: {datetime.now(timezone.utc).date().isoformat()} (UTC)")

with st.spinner("Retrieving official World Bank observations…"):
    source_df, api_error = fetch_world_bank(COUNTRIES[country_name], code, start_year, end_year)
if api_error:
    st.warning(api_error)
source_df = source_df.sort_values("year").reset_index(drop=True)
if source_df.empty:
    st.error("No source observations are available. The app will not invent replacement data. Try another indicator/country or retry later.")
    st.stop()

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Indicator and source")
    st.write(f"**Definition:** {definition}")
    st.write(f"**Units:** {units}")
    st.write(f"**Coverage:** {source_df.year.min()}–{source_df.year.max()} · **{len(source_df)} non-null observations** (coverage can be shorter than the requested period).")
    st.markdown(f"[Official World Bank indicator page](https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{code}?format=json)")
with right:
    st.subheader("Source data")
    st.dataframe(source_df, use_container_width=True, hide_index=True)

st.subheader("Validate and edit working data")
st.write("The source table above is preserved separately. Edit the working year/value table below to explore a clearly labeled scenario.")
if "data_key" not in st.session_state:
    st.session_state.data_key = None
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0
data_key = (country_name, indicator_name)
if st.session_state.data_key != data_key:
    st.session_state.source_copy = source_df.copy()
    st.session_state.working_df = source_df.copy()
    st.session_state.data_key = data_key
    st.session_state.edited = False
if st.button("Reset to source data"):
    st.session_state.working_df = st.session_state.source_copy.copy()
    st.session_state.edited = False
    st.session_state.editor_version += 1
    st.session_state.pop("calculated_results", None)
    st.session_state.pop("single_result", None)
    st.rerun()
edited_df = st.data_editor(st.session_state.working_df, num_rows="dynamic", use_container_width=True, key=f"editor_{country_name}_{code}_{st.session_state.editor_version}")
if not edited_df.equals(st.session_state.working_df):
    st.session_state.working_df = edited_df.copy()
    st.session_state.edited = True
    st.session_state.pop("calculated_results", None)
    st.session_state.pop("single_result", None)
validation_errors = validate_data(st.session_state.working_df, indicator_name)
if validation_errors:
    for err in validation_errors:
        st.error(err)
    st.stop()
working = st.session_state.working_df.copy()
working["year"] = working["year"].astype(int)
working["value"] = working["value"].astype(float)
working = working.sort_values("year").reset_index(drop=True)
if st.session_state.edited:
    st.warning("USER-MODIFIED SCENARIO: outputs and exports use edited data, not only the original source.")
else:
    st.success("Working data currently matches the source data.")

try:
    data, base, coeff, model, fitted = fit_polynomial(working, degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

r2 = r2_score(data["value"], fitted) if len(data) > 1 else float("nan")
mae = mean_absolute_error(data["value"], fitted)
residuals = data["value"].to_numpy() - fitted
historical_min, historical_max = int(data.year.min()), int(data.year.max())
mean_value = float(data.value.mean())
equation = "f(year) = " + poly_text(coeff, base)
st.subheader("Polynomial model")
st.latex(sp.latex(sum(sp.Float(float(c), 10) * (sp.Symbol("year") - base)**(len(coeff)-i-1) for i,c in enumerate(coeff))))
metric_cols = st.columns(5)
metric_cols[0].metric("Degree", degree)
metric_cols[1].metric("R² (in-sample)", f"{r2:.4f}")
metric_cols[2].metric("MAE (in-sample)", f"{mae:.4g} {units}")
metric_cols[3].metric("Observation count", len(data))
metric_cols[4].metric("Mean", f"{mean_value:.5g} {units}")
st.caption(f"Centered time variable: t = year − {base}. Equation: {equation}. Historical range: {historical_min}–{historical_max}. Fit metrics describe these same observations; they are not a guarantee of future accuracy.")
if len(data) >= max(degree + 3, 8):
    holdout_n = max(1, int(round(len(data) * .2)))
    train = data.iloc[:-holdout_n]
    test = data.iloc[-holdout_n:]
    if len(train) >= degree + 1:
        _, hb, hc, hm, _ = fit_polynomial(train, degree)
        hold_pred = hm(test.year.to_numpy() - hb)
        st.write(f"**Chronological holdout MAE (separate later-years test):** {mean_absolute_error(test.value, hold_pred):.5g} {units} across {len(test)} held-out observation(s). This score uses a model fitted only to earlier years.")
else:
    st.caption("Chronological holdout score not shown: there are not enough observations for a meaningful degree-specific train/test split.")
with st.expander("Residuals by year"):
    residual_table = data[["year", "value"]].copy()
    residual_table["fitted_value"] = fitted
    residual_table["residual (actual − fitted)"] = residuals
    st.dataframe(residual_table, use_container_width=True, hide_index=True)
st.caption("Polynomial fits can swing sharply outside the observed range. Uncertainty is discussed with residuals and MAE; this app does not label a simplified band as a validated 95% prediction interval.")

st.subheader("Calculate a single-year estimate")
single_year = st.number_input("Year to calculate (1950–2100)", min_value=1950, max_value=2100, value=min(2100, max(1950, historical_max + 1)), step=1)
if st.button("Calculate Value", type="primary"):
    st.session_state.single_result = {"year": int(single_year), "value": float(model(single_year-base)), "key": (country_name, indicator_name, degree, tuple(data.year), tuple(data.value))}
current_key = (country_name, indicator_name, degree, tuple(data.year), tuple(data.value))
if "single_result" in st.session_state:
    result = st.session_state.single_result
    if result["key"] != current_key:
        st.session_state.pop("single_result", None)
    else:
        within = historical_min <= result["year"] <= historical_max
        observed_match = data.loc[data.year == result["year"], "value"]
        st.write(f"**Year:** {result['year']} · **Model estimate:** {result['value']:.7g} {units} · **Classification:** {'within fitted historical range' if within else 'outside fitted range (extrapolation)'}")
        if not observed_match.empty:
            st.write(f"At this year the actual observed value is {observed_match.iloc[0]:.7g} {units}; the model estimate is a fitted value, not the observation.")
        if abs(result["value"]) > max(1000, abs(mean_value)*100):
            st.warning("This estimate is unusually large relative to the historical mean. Check the polynomial degree and treat extrapolation cautiously.")

st.subheader("Batch estimates")
batch_mode = st.radio("Input method", ["Year range", "Custom years"], horizontal=True)
batch_years = []
invalid_year_tokens = []
if batch_mode == "Year range":
    c1,c2,c3 = st.columns(3)
    with c1: batch_start = st.number_input("Start year", 1950, 2100, historical_max+1 if historical_max < 2100 else 2100, key="batch_start")
    with c2: batch_end = st.number_input("End year", 1950, 2100, min(2100, historical_max+10), key="batch_end")
    with c3: batch_increment = st.slider("Range step (years)", 1, 10, 1)
    if batch_end < batch_start:
        st.error("End year must be greater than or equal to start year.")
    else:
        batch_years = list(range(int(batch_start), int(batch_end)+1, int(batch_increment)))
else:
    custom = st.text_input("Comma-separated years", f"{historical_max+1}, {historical_max+5}, {historical_max+10}")
    for token in custom.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            yr = int(token)
            if str(yr) != token and not re.fullmatch(r"[+-]?\d+", token):
                raise ValueError
            if not 1950 <= yr <= 2100:
                invalid_year_tokens.append(token)
            else:
                batch_years.append(yr)
        except ValueError:
            invalid_year_tokens.append(token)
if invalid_year_tokens:
    st.error("Invalid year input(s): " + ", ".join(invalid_year_tokens) + ". Use whole years from 1950 through 2100.")
batch_years = sorted(set(batch_years))
if batch_years and st.button("Calculate batch results"):
    vals = model(np.array(batch_years, dtype=float)-base)
    st.session_state.calculated_results = pd.DataFrame({"year": batch_years, "estimate": vals, "units": units,
        "range_status": ["within fitted range" if historical_min <= y <= historical_max else "outside fitted range (extrapolation)" for y in batch_years]})
    st.session_state.batch_key = current_key
if "calculated_results" in st.session_state:
    if st.session_state.get("batch_key") != current_key:
        st.session_state.pop("calculated_results", None)
    else:
        st.dataframe(st.session_state.calculated_results, use_container_width=True, hide_index=True)

st.subheader("Country comparison")
comparison_frames = []
for cname in comparison_names:
    cdf, cerr = fetch_world_bank(COUNTRIES[cname], code, start_year, end_year)
    if cerr:
        st.caption(f"{cname}: {cerr}")
        continue
    cdf["country"] = cname
    comparison_frames.append(cdf)
if comparison_frames:
    comp = pd.concat([pd.DataFrame({"year": data.year, "value": data.value, "country": country_name})] + comparison_frames, ignore_index=True)
    figc, axc = plt.subplots(figsize=(10,4))
    for cname, group in comp.groupby("country"):
        axc.plot(group.year, group.value, marker="o", markersize=2.5, linewidth=1.5, label=cname)
    axc.set_title(f"{indicator_name}: country comparison")
    axc.set_xlabel("Year"); axc.set_ylabel(units); axc.grid(True, alpha=.25); axc.legend(fontsize=8)
    figc.tight_layout()
    st.pyplot(figc)
    st.caption("Country coverage can differ because source observations are missing or begin/end in different years. Comparisons use each country's reported observations; no gaps are filled with fabricated values.")
    plt.close(figc)
else:
    st.info("Select one or more comparison countries to display available source series.")

st.subheader("Historical model graph")
query_years = []
if "calculated_results" in st.session_state and st.session_state.get("batch_key") == current_key:
    query_years = st.session_state.calculated_results["year"].tolist()
display_end = max(historical_max, historical_max + horizon)
fig = make_chart(data, base, model, sample_step, display_end, query_years)
if "single_result" in st.session_state and st.session_state.single_result.get("key") == current_key:
    sr = st.session_state.single_result
    ax = fig.axes[0]
    ax.scatter([sr["year"]], [sr["value"]], marker="*", s=140, label="Single-year estimate", color="#c0392b", zorder=5)
    ax.legend(fontsize=8)
st.pyplot(fig)
chart_buf = io.BytesIO()
fig.savefig(chart_buf, format="png", dpi=150, bbox_inches="tight")
chart_png = chart_buf.getvalue()
plt.close(fig)

st.subheader("Mathematical pathway and function lab")
course = st.selectbox("Assigned mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
context = st.selectbox("Context", list(CONTEXTS))
default_y, default_yu, default_x, default_xu, default_desc = CONTEXTS[context]
desc = st.text_input("Context description", default_desc)
var_x = st.text_input("Independent variable name", default_x)
unit_x = st.text_input("Independent variable units", default_xu)
var_y = st.text_input("Dependent variable name", default_y)
unit_y = st.text_input("Dependent variable units", default_yu)
function_source = st.text_input("Function in x (examples: x**2 - 4*x + 3, exp(-x), sin(x))", SUGGESTIONS[context])
analysis_cols = st.columns(3)
with analysis_cols[0]: x_min = st.number_input("Analysis interval start", value=0.0, min_value=-100000.0, max_value=100000.0)
with analysis_cols[1]: x_max = st.number_input("Analysis interval end", value=10.0, min_value=-100000.0, max_value=100000.0)
with analysis_cols[2]: x_point = st.number_input("Current input x", value=2.0, min_value=-100000.0, max_value=100000.0)
show_f = st.checkbox("Show f(x)", True)
show_d1 = st.checkbox("Show first derivative", True)
show_d2 = st.checkbox("Show second derivative", course == "Calculus")
try:
    expr = safe_expression(function_source)
    x = sp.Symbol("x")
    d1 = sp.diff(expr, x)
    d2 = sp.diff(d1, x)
    fn, dfn, d2fn = numeric_fn(expr), numeric_fn(d1), numeric_fn(d2)
    if x_max <= x_min:
        raise ValueError("Analysis interval end must be greater than its start.")
    st.latex("f(x) = " + sp.latex(expr))
    if course == "Calculus":
        st.latex("f'(x) = " + sp.latex(d1))
        st.latex("f''(x) = " + sp.latex(d2))
    elif course == "Precalculus":
        st.write("Precalculus focus: degree and coefficients shape the graph; turning points and concavity can be investigated through changes in average rate.")
        st.write(f"Expression degree (polynomial only): {sp.Poly(expr, x).degree() if expr.is_polynomial(x) else 'not a polynomial'}")
        st.latex("f'(x) = " + sp.latex(d1))
    else:
        st.write("Honors Algebra 2 focus: domain/range on the selected interval, graph trends, end behavior, and average rate of change.")
    with np.errstate(all="ignore"):
        grid = np.linspace(float(x_min), float(x_max), 800)
        ys = np.asarray(fn(grid), dtype=float)
        y1s = np.asarray(dfn(grid), dtype=float)
        y2s = np.asarray(d2fn(grid), dtype=float)
    if ys.ndim == 0: ys = np.full_like(grid, ys)
    if y1s.ndim == 0: y1s = np.full_like(grid, y1s)
    if y2s.ndim == 0: y2s = np.full_like(grid, y2s)
    ys = np.where(np.isfinite(ys), ys, np.nan)
    y1s = np.where(np.isfinite(y1s), y1s, np.nan)
    y2s = np.where(np.isfinite(y2s), y2s, np.nan)
    figf, axf = plt.subplots(figsize=(10,4.8))
    if show_f: axf.plot(grid, ys, label="f(x)", linewidth=2)
    if show_d1: axf.plot(grid, y1s, label="f'(x)", linewidth=1.5)
    if show_d2: axf.plot(grid, y2s, label="f''(x)", linewidth=1.5)
    if math.isfinite(float(fn(x_point))):
        yp = float(fn(x_point))
        axf.scatter([x_point], [yp], color="#c0392b", zorder=5, label=f"Current point ({x_point:g}, {yp:.4g})")
        slope = float(dfn(x_point))
        if math.isfinite(slope) and show_f:
            tangent_x = np.linspace(max(x_min, x_point-(x_max-x_min)*.15), min(x_max, x_point+(x_max-x_min)*.15), 60)
            axf.plot(tangent_x, yp+slope*(tangent_x-x_point), linestyle="--", label="Tangent line")
    axf.axhline(0, color="black", linewidth=.6); axf.axvline(0, color="black", linewidth=.6)
    axf.set_xlabel(f"{var_x} ({unit_x})"); axf.set_ylabel("Function / derivative values")
    axf.grid(True, alpha=.25); axf.legend(fontsize=8); figf.tight_layout()
    st.pyplot(figf)
    with np.errstate(all="ignore"):
        fa, fb = float(fn(x_min)), float(fn(x_max))
        fp = float(fn(x_point))
        slope_p = float(dfn(x_point))
        curvature_p = float(d2fn(x_point))
    avg_rate = (fb-fa)/(x_max-x_min) if np.isfinite(fa) and np.isfinite(fb) else float("nan")
    st.markdown("**Interpretation in context**")
    st.write(f"{desc} The function value at x = {x_point:g} is {fmt_num(fp)} {unit_y}.")
    st.write(f"The first derivative there is {fmt_num(slope_p)} {unit_y} per {unit_x}; it describes the instantaneous rate of change. The second derivative is {fmt_num(curvature_p)} {unit_y} per {unit_x}²; it describes how that rate changes.")
    st.write(f"Average rate of change on [{x_min:g}, {x_max:g}] is {fmt_num(avg_rate)} {unit_y} per {unit_x}.")
    if course == "Calculus":
        st.markdown("**Numerical and symbolic calculus analysis**")
        try:
            crit_expr = sp.solve(sp.Eq(d1, 0), x)
            crit = []
            for sol in crit_expr:
                if sol.is_real is False: continue
                val = float(sp.N(sol))
                if x_min <= val <= x_max and math.isfinite(val):
                    eps = max(1e-5, (x_max-x_min)*1e-5)
                    left_s = float(dfn(val-eps)); right_s = float(dfn(val+eps))
                    kind = "local minimum" if left_s < 0 < right_s else "local maximum" if left_s > 0 > right_s else "stationary point (not classified as an extremum)"
                    crit.append((val, float(fn(val)), kind))
            if crit:
                for xx, yy, kind in crit:
                    st.write(f"x ≈ {xx:.6g}, f(x) ≈ {yy:.6g}: {kind}, classified using a numerical sign-change check of f'.")
            else:
                st.write("No real stationary points were found in the selected interval by the symbolic solver.")
            try:
                roots2 = sp.solve(sp.Eq(d2, 0), x)
                infl = []
                for sol in roots2:
                    if sol.is_real is False: continue
                    val = float(sp.N(sol))
                    if x_min < val < x_max:
                        eps = max(1e-5, (x_max-x_min)*1e-5)
                        left_c, right_c = float(d2fn(val-eps)), float(d2fn(val+eps))
                        if np.isfinite(left_c) and np.isfinite(right_c) and left_c*right_c < 0:
                            infl.append((val, float(fn(val))))
                st.write("Verified numerical inflection candidates (concavity changes): " + (", ".join(f"x≈{xx:.5g}, f(x)≈{yy:.5g}" for xx,yy in infl) if infl else "none detected in this interval."))
            except Exception:
                st.write("Inflection analysis could not be completed for this expression.")
            st.write(f"Closed-interval endpoint values: f({x_min:g}) = {fmt_num(fa)}, f({x_max:g}) = {fmt_num(fb)}.")
            if np.isfinite(ys).any():
                finite = np.isfinite(ys)
                idx_min, idx_max = np.nanargmin(ys), np.nanargmax(ys)
                st.write(f"Numerical interval minimum ≈ {np.nanmin(ys):.6g} near x≈{grid[idx_min]:.6g}; maximum ≈ {np.nanmax(ys):.6g} near x≈{grid[idx_max]:.6g}. These grid-based estimates may miss narrow features.")
            st.write("Domain note: the plotted interval is sampled numerically. For rational/logarithmic expressions, gaps and restrictions depend on the expression; NaN values are not connected across gaps. Check symbolic denominators and logarithm arguments before making a domain claim.")
        except Exception as exc:
            st.warning(f"Some calculus classifications could not be computed reliably: {exc}")
    st.caption("Exact symbolic expressions are shown where SymPy can derive them. Critical points, concavity changes, and interval extrema reported with ≈ are numerical approximations, not proofs. A polynomial fit is descriptive, not causal.")
except Exception as exc:
    st.error(f"Function input error: {exc}. Try supported syntax such as x**2, 2*x+1, sin(x), cos(x), exp(-x), log(x), or sqrt(x). Use ** for powers (or ^, which is converted).")

st.subheader("Interactive simulation")
visual = st.selectbox("Contextual visual", ["Walking person with prosthetic leg (illustrative)", "Moving object / projectile", "Tank or concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
play = st.checkbox("Auto-play input value", value=False)
speed = st.slider("Playback speed (input units per step)", 0.1, 5.0, 1.0, 0.1)
if play:
    st.caption("Playback advances on Streamlit reruns. Adjust the current input slider to explore; this simple version does not run a background animation timer.")
sim_x = st.slider("Simulation input", float(x_min), float(x_max), float(min(max(x_point, x_min), x_max)), key="sim_x")
try:
    sim_y = float(fn(sim_x))
    sim_slope = float(dfn(sim_x))
    st.write(f"**{visual}** · input = {sim_x:.4g} {unit_x} · output = {fmt_num(sim_y)} {unit_y} · rate = {fmt_num(sim_slope)} {unit_y} per {unit_x}")
    st.progress(float(np.clip((sim_x-x_min)/(x_max-x_min), 0, 1)))
    if "prosthetic" in visual:
        st.write("Conceptual gait indicator: the bar above tracks normalized input position, not a clinical gait measurement. This is a mathematical visualization only.")
    elif "projectile" in visual:
        st.write("The moving marker represents the selected function output; a real projectile requires a physically appropriate model and units.")
    elif "Tank" in visual or "concentration" in visual:
        st.write("The indicator reflects the model output. Negative or non-finite values may not be physically meaningful.")
    elif "Population" in visual:
        st.write("The output represents a modelled population/cell count; a polynomial or custom function may not remain realistic outside its interval.")
    elif "Thermometer" in visual:
        st.write("The output is shown in the chosen units; the app does not infer weather causes.")
    else:
        st.write("The output is a model value, not a guaranteed business or economic outcome.")
except Exception:
    st.warning("The function is undefined at the current simulation input.")

st.subheader("Exports")
calc_df = st.session_state.get("calculated_results", pd.DataFrame(columns=["year","estimate","units","range_status"]))
summary = pd.DataFrame([{
    "country": country_name, "indicator": indicator_name, "indicator_code": code, "units": units,
    "definition": definition, "retrieved_utc_date": datetime.now(timezone.utc).date().isoformat(),
    "source": f"https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{code}?format=json",
    "degree": degree, "base_year": base, "equation": equation, "r_squared_in_sample": r2,
    "mae_in_sample": mae, "observation_count": len(data), "historical_min_year": historical_min,
    "historical_max_year": historical_max, "mean": mean_value, "user_modified_scenario": st.session_state.edited,
    "model_limitations": "Polynomial extrapolation is uncertain; metrics are in-sample; no causal claims."
}])
csv_bytes = data.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
xlsx_buf = io.BytesIO()
with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    working.to_excel(writer, sheet_name="Working Data", index=False)
    calc_df.to_excel(writer, sheet_name="Calculated Years", index=False)
st.download_button("Download Excel workbook (.xlsx)", xlsx_buf.getvalue(), file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
pdf_details = [
    f"Country: {country_name}; indicator: {indicator_name} ({code}); units: {units}.",
    f"Coverage: {historical_min}–{historical_max}; observations: {len(data)}; retrieved UTC date: {datetime.now(timezone.utc).date().isoformat()}.",
    f"Equation: {equation}. Base year: {base}; degree: {degree}.",
    f"In-sample R-squared: {r2:.6g}; in-sample MAE: {mae:.6g} {units}; mean: {mean_value:.6g} {units}.",
    f"Data status: {'USER-MODIFIED SCENARIO' if st.session_state.edited else 'source-matched working data'}.",
    "Limitations: polynomial extrapolation can be implausible; no validated prediction interval is supplied; association does not establish cause."
]
pdf_bytes = make_pdf("Hispanic Heritage Historical Data Explorer Report", pdf_details, chart_png, working, calc_df)
st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"{country_name} · {indicator_name} · {historical_min}–{historical_max} · n={len(data)}")
    st.write(equation)
    st.write(f"In-sample R²={r2:.4f}; MAE={mae:.5g} {units}; mean={mean_value:.5g} {units}.")
    st.write("Interpretation: the polynomial summarizes the selected historical observations. It does not identify historical causes and can behave implausibly outside the fitted range.")
    st.dataframe(data, use_container_width=True, hide_index=True)

with st.expander("How the app works, assumptions, and limitations"):
    st.markdown("""
    **Data flow:** controls select a country and indicator → the official World Bank API is requested and cached → source observations are kept separately from the editable working table → validation checks years and values → a centered-time polynomial is fitted → the chart, calculations, comparisons, and exports use the current working data.

    **Why centered time?** The model uses `t = year − base year`, reducing the scale of the year values and making polynomial fitting more numerically stable. The displayed polynomial is a least-squares fit, not a causal explanation.

    **Metrics:** R² and MAE are in-sample fit metrics. A chronological holdout MAE, when shown, tests later years using a model trained only on earlier years. Neither metric guarantees future performance.

    **Function lab:** students can select a context, variables, units, interval, and a supported expression. SymPy differentiates the expression symbolically; numerical sampling supports graphing and approximate classification. Rational/logarithmic expressions can be undefined at some inputs. The app does not draw finite values where evaluation is non-finite.

    **Limitations:** missing source observations remain missing; data can be revised by the World Bank; edited data are scenarios; polynomial extrapolation can be unstable; numerical calculus checks may miss close roots or narrow behavior; example context functions are instructional and not validated medical, engineering, weather, or economic models.
    """)
