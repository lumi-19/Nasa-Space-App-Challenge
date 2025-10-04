
from pathlib import Path
import pandas as pd
import streamlit as st
import re
import requests
from collections import Counter
import plotly.express as px
import streamlit.components.v1 as components
import io
import base64
import html
from datetime import datetime

# optional imports (wrapped)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ------------------------------
# File paths (keep your original paths)
# ------------------------------
base_path = Path(__file__).parent

# ------------------------------
# Define file paths (relative to script)
# ------------------------------
file_main = base_path / "abstracts.csv"
file_result = base_path / "Ressult.csv"
file_abstract = base_path / "abstract.csv"
file_author = base_path / "Author.csv"

# ------------------------------
# Safe CSV loader
# ------------------------------
def load_safe(path):
    try:
        df = pd.read_csv(path)
        st.success(f"✅ Loaded: {path.name} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Failed to load {path.name}: {e}")
        return pd.DataFrame()

# ------------------------------
# Load data safely
# ------------------------------
df_main = load_safe(file_main)       # Title + Link
df_result = load_safe(file_result)   # Result
df_abstract = load_safe(file_abstract)  # Abstract
df_author = load_safe(file_author)   # Author + Date
# normalize and assemble search_df
if "Date" in df_author.columns:
    # parse full date like "2014 Aug 18"
    df_author["Date"] = pd.to_datetime(df_author["Date"], errors="coerce")
    df_author["Year"] = df_author["Date"].dt.year
else:
    df_author["Date"] = pd.NaT
    df_author["Year"] = None

search_df = pd.DataFrame({
    "Title": df_main["Title"] if "Title" in df_main.columns else "",
    "Link": df_main["Link"] if "Link" in df_main.columns else "",
    "Abstract": df_abstract["Abstract"] if "Abstract" in df_abstract.columns else "",
    "Result": df_result["Result"] if "Result" in df_result.columns else "",
    "Author": df_author["Author"] if "Author" in df_author.columns else "",
    "Date": df_author["Year"] if "Year" in df_author.columns else pd.Series([None]*len(df_main))
})

# safe fill
search_df.fillna("", inplace=True)

# ------------------------------
# Utility helpers
# ------------------------------
def make_summary(text: str, n_words=40):
    if not isinstance(text, str) or text.strip()=="":
        return "No abstract available."
    words = text.split()
    if len(words) <= n_words:
        return text
    return " ".join(words[:n_words]) + " ..."

def highlight(text: str, keyword: str):
    if not keyword or not text:
        return text
    try:
        return re.sub(f"({re.escape(keyword)})", r"**\1**", text, flags=re.IGNORECASE)
    except Exception:
        return text

# detect GLDS/OSD identifiers in text
GLDS_RE = re.compile(r"\b(GLDS-?\d+|OSD-?\d+)\b", flags=re.IGNORECASE)
def extract_dataset_ids(text: str):
    if not isinstance(text, str):
        return []
    matches = GLDS_RE.findall(text)
    normalized = []
    for m in matches:
        m_clean = m.upper().replace("GLDS", "GLDS").replace("OSD", "OSD")
        normalized.append(m_clean.replace("GLDS-", "GLDS-").replace("OSD-", "OSD-"))
    return list(dict.fromkeys(normalized))

# ------------------------------
# GeneLab fallback (query by GLDS numeric id)
# ------------------------------
GENELAB_FILES_ENDPOINT = "https://genelab-data.ndc.nasa.gov/genelab/data/glds/files/{}"
GENELAB_ACCESSION_PAGE = "https://genelab-data.ndc.nasa.gov/genelab/accession/{}"  # pass GLDS-<num>

@st.cache_data(show_spinner=False)
def query_genelab_by_glds(glds_tag: str, timeout=10):
    """
    glds_tag like 'GLDS-87' or 'GLDS87' or 'GLDS-170'
    Returns: dict with keys: accession_url, hit (bool), files (list) or None
    """
    if not glds_tag:
        return None
    # extract numeric portion
    digits = re.findall(r"\d+", glds_tag)
    if not digits:
        return None
    num = digits[0]
    try:
        url = GENELAB_FILES_ENDPOINT.format(num)
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            files = []
            # j may contain 'studies'-> 'GLDS-<num>' mapping
            if isinstance(j, dict):
                # try to discover GLDS key
                if "studies" in j:
                    studies = j["studies"]
                    key = next((k for k in studies.keys() if k.upper().startswith("GLDS")), None)
                    if key:
                        entry = studies.get(key, {})
                        files = entry.get("study_files", [])
                else:
                    # fallback: store raw json as files
                    files = j.get("study_files", []) or j.get("files", []) or []
            accession = f"GLDS-{num}"
            return {
                "accession": accession,
                "accession_url": GENELAB_ACCESSION_PAGE.format(accession),
                "file_count": len(files) if isinstance(files, list) else 0,
                "study_files": files
            }
        else:
            return {"error": f"GeneLab returned status {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ------------------------------
# Helper: NSLSL / TaskBook / Google search link builders
# ------------------------------
import urllib.parse
def nslsl_search_url(query_text: str):
    # NSLSL has a public search page; we craft a link to the search page with a query param that works for many users.
    q = urllib.parse.quote_plus(query_text)
    # primary NSLSL search (opens the NSLSL interface with the query)
    return f"https://extapps.ksc.nasa.gov/NSLSL/Search?SearchString={q}"

def pubmed_search_url(query_text: str):
    return f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote_plus(query_text)}"

def taskbook_site_search_url(query_text: str):
    # Task Book does not expose a simple documented GET search param; provide a Google site: search fallback and TaskBook home.
    google_site_search = f"https://www.google.com/search?q=site:taskbook.nasaprs.com+{urllib.parse.quote_plus(query_text)}"
    taskbook_home = "https://taskbook.nasaprs.com/tbp/index.cfm"
    return {"taskbook_home": taskbook_home, "google_site_search": google_site_search}

# ------------------------------
# Recommendation engine (TF-IDF) - simple local model
# ------------------------------
@st.cache_data(show_spinner=False)
def build_tfidf_index(corpus_list):
    if not SKLEARN_AVAILABLE:
        return None, None
    vect = TfidfVectorizer(stop_words="english", max_features=10000)
    X = vect.fit_transform(corpus_list)
    return vect, X

def get_recommendations(idx, X, top_n=5):
    # idx = index int in X; returns list of (index, score)
    if X is None:
        return []
    sims = cosine_similarity(X[idx:idx+1], X).flatten()
    sims[idx] = -1  # ignore itself
    best = sims.argsort()[::-1][:top_n]
    return list(zip(best, sims[best]))

# ------------------------------
# Word cloud generator (returns image bytes)
# ------------------------------
def generate_wordcloud_image(text):
    if not WORDCLOUD_AVAILABLE or not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA").generate(text)
    img_buf = io.BytesIO()
    wc.to_image().save(img_buf, format="PNG")
    img_buf.seek(0)
    return img_buf

# ------------------------------
# Knowledge graph (pyvis) - build small network linking papers->topics->organisms
# ------------------------------
TOPIC_KEYWORDS = {
    "plants":["plant","Arabidopsis","wheat","crop","seedling","phototropism","photosynthesis"],
    "humans":["human","astronaut","crew","cardiac","bone","muscle","immune","cardiovascular"],
    "microbes":["bacteria","microbe","microbial","microbiome","fungi","yeast"],
    "radiation":["radiation","ionizing","dosimetry","HZE","gamma","protons"],
    "microgravity":["microgravity","weightless","spaceflight","micro-gravity","microgravity","spaceflight"]
}

def extract_topics(text):
    found = set()
    t = text.lower()
    for topic, keys in TOPIC_KEYWORDS.items():
        for k in keys:
            if k.lower() in t:
                found.add(topic)
                break
    return list(found)

def build_pyvis_graph(papers):
    """
    papers: list of dicts: {"id":id, "title":..., "topics":[...], "authors":[...]}
    returns HTML representation
    """
    if not PYVIS_AVAILABLE:
        return None
    net = Network(height="650px", width="100%", bgcolor="#111827", font_color="white")
    net.barnes_hut()
    for p in papers:
        pid = f"paper_{p['id']}"
        net.add_node(pid, label=p['title'][:60], title=p['title'], color="#0ea5a4", shape="dot", size=15)
        for topic in p.get("topics", []):
            tnode = f"topic_{topic}"
            net.add_node(tnode, label=topic, title=topic, color="#f97316", shape="box", size=12)
            net.add_edge(pid, tnode)
        # add author nodes (optional)
        for a in (p.get("authors") or []):
            anode = f"author_{a}"
            net.add_node(anode, label=a, title=a, color="#60a5fa", shape="triangle", size=10)
            net.add_edge(pid, anode)
    html_str = net.generate_html()
    return html_str

# ------------------------------
# BibTeX generator (very simple)
# ------------------------------
def make_bibtex(row):
    # row: Series with Title, Author, Date, Link
    authors = row.get("Author","")
    year = row.get("Date","")
    title = row.get("Title","").replace("{","").replace("}","")
    key = re.sub(r"\W+","", (authors.split()[0] if authors else "anon") + str(year))[:30]
    bib = f"@article{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{year}}},\n  url = {{{row.get('Link','')}}}\n}}"
    return bib

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="NASA Space Biology Knowledge Engine", page_icon="🚀", layout="wide")
st.markdown(
    """
    <style>
    .stApp {{
        background: radial-gradient(circle at 10% 10%, #020617 0%, #071129 35%, #000814 100%);
        color: #e6eef8;
    }}
    .badge {{
        display:inline-block;padding:6px 10px;border-radius:12px;background:#0ea5a4;color:#021124;font-weight:600;margin-right:6px;
    }}
    .smallmuted{{font-size:12px;color:#9fb0c8}}
    </style>
    """, unsafe_allow_html=True
)

# top Lottie (same as before)
def st_lottie_embed(url, height=160):
    lottie_html = f"""
    <lottie-player src="{url}" background="transparent" speed="1"
      style="height:{height}px;" loop autoplay></lottie-player>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """
    components.html(lottie_html, height=height+20)

st_lottie_embed("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json", height=140)
st.title("🚀 NASA Space Biology Knowledge Engine")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
query = st.sidebar.text_input("🔍 Search keyword (title/abstract/result):", "")
results_per_page = st.sidebar.slider("Results per page:", 5, 30, 15)
year_min = int(search_df["Date"].replace('', pd.NA).dropna().min() or datetime.now().year - 30)
year_max = int(search_df["Date"].replace('', pd.NA).dropna().max() or datetime.now().year)
year_range = st.sidebar.slider("Filter by publication year", year_min, year_max, (year_min, year_max))
author_list = ["All"] + sorted(search_df["Author"].unique().tolist())
author_filter = st.sidebar.selectbox("Filter by Author", author_list)

# quick toggles
show_wordcloud = st.sidebar.checkbox("Show word cloud", value=True)
show_knowledge_graph = st.sidebar.checkbox("Show knowledge graph (small)", value=False)
enable_recommendation = st.sidebar.checkbox("Enable recommendations (TF-IDF)", value=SKLEARN_AVAILABLE)
enable_audio = st.sidebar.checkbox("Enable audio summaries (gTTS)", value=False and GTTS_AVAILABLE)

# Apply search & filters
results = search_df.copy()

if query:
    mask = (
        results["Title"].str.contains(query, case=False, na=False) |
        results["Abstract"].str.contains(query, case=False, na=False) |
        results["Result"].str.contains(query, case=False, na=False) |
        results["Author"].str.contains(query, case=False, na=False)
    )
    results = results[mask]

# year filtering (Date column is integer year)
results["DateClean"] = pd.to_numeric(results["Date"], errors="coerce").fillna(0).astype(int)
results = results[(results["DateClean"] >= year_range[0]) & (results["DateClean"] <= year_range[1])]
if author_filter != "All":
    results = results[results["Author"] == author_filter]

st.subheader(f"📄 Results for '{query}' ({len(results)})")

if results.empty:
    st.warning("No results found. Try broadening your query or adjust filters.")
else:
    # pagination
    total_results = len(results)
    total_pages = (total_results - 1) // results_per_page + 1
    page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * results_per_page
    page_results = results.iloc[start_idx:start_idx + results_per_page]

    # Build corpus for TF-IDF (for recommendations)
    corpus = (search_df["Title"].fillna("") + " " + search_df["Abstract"].fillna("") + " " + search_df["Result"].fillna("")).tolist()
    vect, X = (None, None)
    if enable_recommendation and SKLEARN_AVAILABLE:
        vect, X = build_tfidf_index(corpus)

    # Tabs for modes
    tab1, tab2, tab3 = st.tabs(["Scientist 👩‍🔬", "Manager 💼", "Mission Architect 🚀"])

    # ------------------------------
    # SCIENTIST: detailed view + datasets + "More like this"
    # ------------------------------
    with tab1:
        st.info("Mode: Scientist → Summaries, dataset badges (GeneLab/GLDS), NSLSL / TaskBook quick links.")
        for i, row in page_results.reset_index(drop=True).iterrows():
            st.markdown(f"### {html.escape(row['Title'])}")
            st.markdown(f"<span class='smallmuted'>👩‍🔬 {html.escape(row['Author'])} • 📅 {row['Date']}</span>", unsafe_allow_html=True)
            summary = make_summary(row["Abstract"], n_words=60)
            st.markdown(f"**Abstract:** {summary}")
            st.markdown(f"**Result:** {row['Result'][:600]}")

            # Detect GLDS/OSD IDs in abstract/result
            combined_text = f"{row.get('Abstract','')} {row.get('Result','')}"
            dataset_ids = extract_dataset_ids(combined_text)
            if dataset_ids:
                st.markdown("**🔗 Datasets mentioned in paper:**")
                for ds in dataset_ids:
                    ds_link = ""
                    if ds.upper().startswith("GLDS"):
                        q = query_genelab_by_glds(ds)
                        if q and "accession_url" in q:
                            ds_link = q["accession_url"]
                            st.markdown(f"- 🔬 [{ds}]({ds_link})  — {q.get('file_count','?')} files")
                        else:
                            # fallback render
                            ds_link = f"https://genelab-data.ndc.nasa.gov/genelab/accession/{ds.upper()}"
                            st.markdown(f"- 🔬 [{ds}]({ds_link})")
                    else:
                        # OSD fallback to OSDR (link only)
                        ds_link = f"https://osdr.nasa.gov/bio/repo/data/studies/{ds}"
                        st.markdown(f"- 🔬 [{ds}]({ds_link})")

            # If no detected dataset, show a GeneLab quick-search & NSLSL + PubMed links
            else:
                nslsl = nslsl_search_url(row["Title"] or row["Abstract"][:120])
                pubmed = pubmed_search_url(row["Title"] or row["Abstract"][:120])
                st.markdown(f"[🔎 More like this (NSLSL)]({nslsl}) • [PubMed results]({pubmed})")

            # Task Book & Google site-search fallback
            tb = taskbook_site_search_url(row["Title"] or row["Author"])
            st.markdown(f"[🏷️ Search Task Book]({tb['taskbook_home']}) • [🔎 Task Book (site search)]({tb['google_site_search']})")

            # Recommendation: top 5 similar (by TF-IDF)
            if enable_recommendation and SKLEARN_AVAILABLE:
                # find index of this row in the global search_df
                global_idx = search_df.index[search_df["Title"] == row["Title"]].tolist()
                if global_idx:
                    recs = get_recommendations(global_idx[0], X, top_n=5)
                    if recs:
                        st.markdown("**🤖 Recommendations (similar papers):**")
                        for rid, score in recs:
                            title = search_df.iloc[rid]["Title"]
                            link = search_df.iloc[rid]["Link"]
                            st.markdown(f"- {title} — score {score:.2f} — [Open]({link})")

            # audio summary (optional)
            if enable_audio and GTTS_AVAILABLE:
                try:
                    txt = summary if summary else (row.get("Abstract","")[:250])
                    tts = gTTS(txt, lang="en")
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    st.audio(buf.read(), format="audio/mp3")
                except Exception as e:
                    st.write("Audio not available:", e)

            st.markdown(f"[🔗 Read Paper]({row.get('Link','')})")
            st.markdown("---")

    # ------------------------------
    # MANAGER: visualizations + topic distribution + wordcloud
    # ------------------------------
    with tab2:
        st.info("Mode: Manager → timeline, topic distribution, top keywords, export.")

        # Timeline: publications by year
        years = pd.to_numeric(results["Date"], errors="coerce").dropna().astype(int)
        if not years.empty:
            fig = px.histogram(years, x=years, nbins=(year_max-year_min+1), title="Publications by Year")
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No year data to show timeline.")

        # Topic distribution
        topic_counts = Counter()
        for _, r in results.iterrows():
            txt = f"{r['Title']} {r['Abstract']} {r['Result']}"
            tpcs = extract_topics(txt)
            for t in tpcs:
                topic_counts[t]+=1
        if topic_counts:
            tks, vals = zip(*topic_counts.items())
            fig2 = px.pie(names=tks, values=vals, title="Topic distribution")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No topics found (try enabling broader keyword matching).")

        # Top keywords bar
        text_blob = " ".join(results["Title"].fillna("") .tolist()).lower()
        words = re.findall(r'\b[a-z]{3,}\b', text_blob)
        stopwords = {"the","and","for","with","using","study","studies","effects"}
        filtered = [w for w in words if w not in stopwords]
        common = Counter(filtered).most_common(25)
        if common:
            kws, counts = zip(*common)
            fig3 = px.bar(x=counts[::-1], y=kws[::-1], orientation="h", labels={"x":"Frequency","y":"Keyword"}, title="Top keywords")
            st.plotly_chart(fig3, use_container_width=True)

        # Word cloud
        if show_wordcloud and WORDCLOUD_AVAILABLE:
            wc_img = generate_wordcloud_image(" ".join(filtered))
            if wc_img:
                st.image(wc_img)

        # Export options
        st.download_button("⬇️ Download filtered results (CSV)", data=results.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_results.csv", mime="text/csv")
        # BibTeX export (all shown)
        if st.button("Export BibTeX for results (preview)"):
            bibs = "\n\n".join(make_bibtex(r) for _, r in results.iterrows())
            st.code(bibs[:10000] + ("\n\n... (truncated)" if len(bibs)>10000 else ""))

    # ------------------------------
    # MISSION ARCHITECT: risk words, knowledge gap, graph
    # ------------------------------
    with tab3:
        st.info("Mode: Mission Architect → Risk keywords, knowledge gaps, knowledge-graph.")
        risk_words = ["radiation", "immune", "bone", "muscle", "microgravity", "health", "stress", "cardiovascular"]
        st.markdown("**Risk keywords present in current page results:**")
        for rw in risk_words:
            found = results[results.apply(lambda r: rw.lower() in (" ".join([str(r.get("Title","")), str(r.get("Abstract","")), str(r.get("Result",""))])).lower(), axis=1)]
            if not found.empty:
                st.markdown(f"- **{rw}** — {len(found)} papers")

        # Knowledge gaps: list topics with low counts (e.g., < 5)
        st.markdown("**Knowledge gaps (topics with few studies):**")
        for topic, keys in TOPIC_KEYWORDS.items():
            cnt = topic_counts.get(topic, 0)
            if cnt < 5:
                st.warning(f"Only {cnt} papers found for topic **{topic}** — possible knowledge gap.")

        # Funding tracker: attempt to find grant numbers or show TaskBook links
        st.markdown("**Funding tracker (Task Book lookup):**")
        # If your dataset has a 'Grant' column, use it; else provide search links per paper
        if "Grant" in search_df.columns and search_df["Grant"].notna().any():
            grants = search_df["Grant"].value_counts().head(20)
            st.table(grants)
        else:
            st.info("No Grant column in CSV — using Task Book site-search fallback.")
            # Show for the page results a quick Task Book site search button
            for _, r in page_results.iterrows():
                tb = taskbook_site_search_url(r["Title"][:150] or r["Author"])
                st.markdown(f"- {r['Title'][:80]} — [TaskBook home]({tb['taskbook_home']}) • [Search (site:taskbook)]({tb['google_site_search']})")

        # Knowledge graph explorer (small)
        if show_knowledge_graph and PYVIS_AVAILABLE:
            # build small set of nodes from page_results
            papers_for_graph = []
            for idx, r in page_results.iterrows():
                papers_for_graph.append({
                    "id": idx,
                    "title": r["Title"][:120],
                    "topics": extract_topics(f"{r['Title']} {r['Abstract']}"),
                    "authors": [r["Author"]] if r["Author"] else []
                })
            html_graph = build_pyvis_graph(papers_for_graph)
            if html_graph:
                components.html(html_graph, height=700)
        else:
            if not PYVIS_AVAILABLE:
                st.info("pyvis not installed — install `pyvis` to enable the interactive knowledge graph.")

    # ------------------------------
    # Bottom: Compare Mode & misc
    # ------------------------------
    st.markdown("---")
    st.subheader("🔎 Compare Mode (pick up to 2 papers)")
    choices = st.multiselect("Pick papers to compare", options=results["Title"].tolist(), max_selections=2)
    if choices:
        cols = st.columns(len(choices))
        for c, title in zip(cols, choices):
            row = results[results["Title"] == title].iloc[0]
            c.markdown(f"### {row['Title']}")
            c.markdown(f"**Author:** {row['Author']}")
            c.markdown(f"**Abstract:** {row['Abstract']}")
            c.markdown(f"[Read Paper]({row['Link']})")
            # show GLDS link if present
            ds = extract_dataset_ids(f"{row['Abstract']} {row['Result']}")
            if ds:
                for d in ds:
                    if d.upper().startswith("GLDS"):
                        q = query_genelab_by_glds(d)
                        if q and "accession_url" in q:
                            c.markdown(f"🔗 Dataset: [{d}]({q['accession_url']})")

    # Knowledge: quick "insights" panel (basic)
    # ------------------------------
# Smart Insights (fixed section)
# ------------------------------
st.markdown("---")
st.subheader("📈 Smart Insights")

# Ensure DateClean exists before using it
if "DateClean" not in results.columns:
    results["DateClean"] = pd.to_numeric(results["Date"], errors="coerce").fillna(0).astype(int)

total_by_year = results["DateClean"].value_counts().sort_index()

if not total_by_year.empty:
    slope = total_by_year.values[-1] - total_by_year.values[0] if len(total_by_year) > 1 else 0
    st.markdown(f"- Total papers in dataset: **{len(search_df)}**")
    st.markdown(f"- Growth trend (earliest → latest year): **{slope:+d}** papers")
else:
    st.info("No valid year data found to compute growth trends.")

agree = results[results["Result"].str.contains(r"\bincrease\b|\bdecrease\b|\breduced\b|\bimproved\b",
                                               case=False, na=False)]
no_effect = results[results["Result"].str.contains(r"\bno effect\b|\bnot significant\b|\bno significant\b",
                                                  case=False, na=False)]

st.markdown(f"- Papers reporting directional effects: **{len(agree)}**")
st.markdown(f"- Papers reporting no-effect results: **{len(no_effect)}**")

# Download button
csv_bytes = page_results.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download this page results (CSV)", data=csv_bytes,
                   file_name="page_results.csv", mime="text/csv")
