import streamlit as st
import networkx as nx
from pyvis.network import Network
import sqlglot
from sqlglot import exp
from typing import List, Tuple, Set, Optional

# =========================
# -------- Helpers --------
# =========================
from streamlit_agraph import agraph, Node as ANode, Edge as AEdge, Config as AConfig

def render_agraph_lr_band(G: nx.DiGraph, focus: str, sources: set, dependents: set, height_px: int = 760):
    X_LEFT, X_MID, X_RIGHT = -450, 0, 450
    Y_STEP = 80

    def grid_positions(n):
        if n == 0:
            return []
        start = -((n - 1) * Y_STEP) / 2
        return [start + i * Y_STEP for i in range(n)]

    nodes = []

    def add_node(node_id: str, kind: str, x: int, y: int, is_focus: bool = False):
        if is_focus:
            color = "#FFD59E"; shape = "box"
        else:
            if kind == "table":
                color = "#B9F6CA"; shape = "box"
            elif kind == "view":
                color = "#A3D3FF"; shape = "ellipse"
            else:
                color = "#E8EAF6"; shape = "box"

        def short_label(name: str) -> str:
            return ".".join(name.split(".")[1:]) if "." in name else name

        nodes.append(ANode(
            id=node_id,
            label=short_label(node_id),
            size=18,
            color=color,
            shape=shape,
            x=x, y=y,
            fixed=True,         
            # title=node_id, 
        ))

    # Sources (left column)
    src_list = sorted(sources)
    for y, n in zip(grid_positions(len(src_list)), src_list):
        kind = G.nodes[n].get("kind", "table")
        add_node(n, kind, X_LEFT, int(y))

    fkind = G.nodes[focus].get("kind", "table")
    add_node(focus, fkind, X_MID, 0, is_focus=True)

    dep_list = sorted(dependents)
    for y, n in zip(grid_positions(len(dep_list)), dep_list):
        kind = G.nodes[n].get("kind", "table")
        add_node(n, kind, X_RIGHT, int(y))


    edges = []
    for s in src_list:
        edges.append(AEdge(source=s, target=focus, label="feeds", smooth=False, physics=False))
    for d in dep_list:
        edges.append(AEdge(source=focus, target=d, label="feeds", smooth=False, physics=False))

    config = AConfig(
        width="100%",
        height=height_px,
        directed=True,
        nodeHighlightBehavior=True,
        highlightColor="#ffd59e",
        collapsible=False,
        physics=False,         
        hierarchical=False,    
    )

    clicked = agraph(nodes=nodes, edges=edges, config=config)
    return clicked


def fqn_from_table(t: exp.Table) -> str:
    parts = []
    for p in (t.catalog, t.db, t.name):  
        if p:
            s = str(p).replace("`", "")
            if s:
                parts.append(s)
    return ".".join(parts) if parts else (t.name or "").replace("`", "")


def complete_fqn(name: str, default_project: Optional[str], default_dataset: Optional[str]) -> str:
    if not name:
        return name
    dot = name.count(".")
    if dot == 0 and default_project and default_dataset:
        return f"{default_project}.{default_dataset}.{name}"
    if dot == 1 and default_project:
        return f"{default_project}.{name}"
    return name


def collect_cte_aliases(stmt: exp.Expression) -> Set[str]:
    names = set()
    for cte in stmt.find_all(exp.CTE):
        alias = (cte.alias or (cte.this and cte.this.name))
        if alias:
            names.add(str(alias).lower())
    return names


# =========================================================
# Parse ONLY: CREATE {TABLE|VIEW} ... AS SELECT ... statements
# =========================================================

def extract_create_targets(sql_text: str, default_project: Optional[str], default_dataset: Optional[str]) -> List[Tuple[str, Set[str], str]]:
    """Parse SQL text and return [(target, {sources}, node_type)]."""
    try:
        statements = sqlglot.parse_many(sql_text, read="bigquery")
    except Exception:
        statements = sqlglot.parse(sql_text, read="bigquery")

    results = []
    for stmt in statements:
        create = stmt.find(exp.Create)
        if not create or not isinstance(create.this, exp.Table):
            continue

        target = fqn_from_table(create.this)
        kind = (create.kind or "").lower()
        node_type = "view" if "view" in kind else "table"

        select_scope = create.find(exp.Select) or stmt
        cte_aliases = collect_cte_aliases(stmt)

        sources: Set[str] = set()
        for t in select_scope.find_all(exp.Table):
            if t is create.this:
                continue
            name = fqn_from_table(t)
            if (t.name and str(t.name).lower() in cte_aliases) and not t.db and not t.catalog:
                continue
            name = complete_fqn(name, default_project, default_dataset)
            if name:
                sources.add(name)

        target = complete_fqn(target, default_project, default_dataset)
        results.append((target, sources, node_type))
    return results


# ==========================
# --------- Graph ----------
# ==========================

def build_graph(pairs: List[Tuple[str, Set[str], str]]) -> nx.DiGraph:
    """Build a DiGraph where edges go target -> source (reads)."""
    G = nx.DiGraph()
    target_set = {t for (t, _, _) in pairs}

    for target, _, node_type in pairs:
        G.add_node(target, kind=node_type)
    for target, sources, _ in pairs:
        for s in sources:
            if s in target_set:
                if not G.has_node(s):
                    G.add_node(s, kind="table")
            else:
                if not G.has_node(s):
                    G.add_node(s, kind="external")
            G.add_edge(target, s, relation="reads")
    return G


def focus_band(G: nx.DiGraph, focus: str) -> Tuple[set, set]:
    """Return (sources feeding focus, dependents reading from focus)."""
    sources = set(G.successors(focus))     
    dependents = set(G.predecessors(focus)) 
    return sources, dependents


def _legend_html() -> str:
    return """
    <div style='display:flex; gap:12px; flex-wrap:wrap; font-size:13px'>
      <span style='background:#FFD59E; padding:3px 8px; border-radius:999px'>Focus</span>
      <span style='background:#B9F6CA; padding:3px 8px; border-radius:999px'>Table</span>
      <span style='background:#A3D3FF; padding:3px 8px; border-radius:999px'>View</span>
      <span style='background:#E8EAF6; padding:3px 8px; border-radius:999px'>External (not created in this SQL)</span>
    </div>
    """


def render_pyvis_lr_band(G: nx.DiGraph, focus: str, sources: set, dependents: set, height_px: int = 760) -> str:
    """Render LR band using PyVis and return the HTML for embedding/download."""
    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False)
    for n in sorted(sources):
        data = G.nodes[n]
        kind = data.get("kind")
        shape = "box" if kind in ("table", "external") else "ellipse"
        color = "#B9F6CA" if kind == "table" else ("#A3D3FF" if kind == "view" else "#E8EAF6")
        net.add_node(n, label=n, shape=shape, color=color, level=0)

    fkind = G.nodes[focus].get("kind", "table")
    fshape = "box" if fkind in ("table", "external") else "ellipse"
    net.add_node(focus, label=focus, shape=fshape, color="#FFD59E", level=1)

    for n in sorted(dependents):
        data = G.nodes[n]
        kind = data.get("kind")
        shape = "box" if kind in ("table", "external") else "ellipse"
        color = "#B9F6CA" if kind == "table" else ("#A3D3FF" if kind == "view" else "#E8EAF6")
        net.add_node(n, label=n, shape=shape, color=color, level=2)


    for s in sources:
        net.add_edge(s, focus, arrows="to", label="feeds")      
    for d in dependents:
        net.add_edge(focus, d, arrows="to", label="feeds")    

    net.set_options(
        """
        {
          "nodes": {"font": {"face": "Inter, Segoe UI, Roboto, Arial"}},
          "layout": { "hierarchical": {
            "enabled": true,
            "direction": "LR",
            "sortMethod": "directed",
            "nodeSpacing": 180,
            "levelSeparation": 220
          }},
          "physics": { "enabled": false },
          "interaction": { "hover": true, "dragNodes": false }
        }
        """
    )

    return net.generate_html(notebook=False)


# ==========================
# --------- UI/UX ----------
# ==========================

st.set_page_config(page_title="BigQuery Reference Viewer", layout="wide")

st.markdown(
    """
    <style>
      .metric-badge {background:#F6F7F9;padding:10px 12px;border-radius:12px;border:1px solid #eaecef}
      .section-card {background:white;padding:16px;border-radius:14px;border:1px solid #ececec}
      .stButton>button {border-radius:10px}
      .stSelectbox [data-baseweb="select"] {border-radius:10px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BigQuery Reference Viewer")
st.caption("Visualize how your CREATE TABLE/VIEW statements read from and feed other objects.")
s
with st.sidebar:
    st.header("Options")
    default_project = "" #st.text_input("Default Project (optional)")
    default_dataset =  "" #st.text_input("Default Dataset (optional)")
    graph_height = st.slider("Graph height (px)", 500, 1200, 800, 50)
    show_tables = st.toggle("Show lists of sources/dependents", value=False)
    st.markdown("---")
    st.subheader("Help")
    with st.expander("How it works"):
        st.write(
            """
            Paste SQL containing **CREATE TABLE/VIEW ... AS SELECT ...**. The app parses BigQuery SQL
            with **sqlglot**, builds a dependency graph, and shows a left→focus→right band:
            * Left: sources that feed the focus
            * Middle: the selected focus target
            * Right: dependents that read from the focus
            CTE-only references are ignored.
            """
        )
    with st.expander("Sample SQL"):
        st.code(
            """
            CREATE TABLE proj.ds.daily_sales AS
            SELECT * FROM proj.raw.orders;

            CREATE VIEW proj.ds.sales_summary AS
            SELECT order_date, SUM(amount) AS total
            FROM proj.ds.daily_sales
            GROUP BY 1;
            """,
            language="sql",
        )

left, mid, right = st.columns([1, 1, 2])

with left:
    st.subheader("SQL Input")
    uploaded = st.file_uploader("Upload .sql/.txt", type=["txt", "sql"], accept_multiple_files=False)
    if uploaded:
        raw_sql = uploaded.read().decode("utf-8", errors="ignore")
    else:
        raw_sql = st.text_area("Paste SQL here", height=260, placeholder="CREATE TABLE proj.ds.base AS SELECT ...")

if raw_sql and raw_sql.strip():
    pairs = extract_create_targets(raw_sql, default_project or None, default_dataset or None)
    if not pairs:
        st.error("No CREATE TABLE/VIEW … AS SELECT found.")
    else:
        G = build_graph(pairs)
        targets = sorted({t for (t, _, _) in pairs})
        if "focus_node" not in st.session_state:
            st.session_state["focus_node"] = targets[0]

        with mid:
            st.subheader("Focus Selection")
            default_index = targets.index(st.session_state["focus_node"]) if st.session_state.get(
                "focus_node") in targets else 0
            selected = st.selectbox("Target (only CREATE targets):", options=targets, index=default_index)
            st.session_state["focus_node"] = selected
            if selected != st.session_state["focus_node"]:
                st.session_state["focus_node"] = selected
                st.rerun()

        focus = st.session_state["focus_node"]
        sources, dependents = focus_band(G, focus)

    
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-badge'><b>Targets</b><br>{len(targets)}</div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-badge'><b>Sources (←)</b><br>{len(sources)}</div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-badge'><b>Dependents (→)</b><br>{len(dependents)}</div>", unsafe_allow_html=True)


        if show_tables:
            with left:
                st.subheader("Feeds Me (Left)")
                if sources:
                    for s in sorted(sources):
                        cols = st.columns([0.35, 0.65])
                        with cols[0]:
                            if s in targets and st.button(f"◀ Focus {s}", key=f"srcbtn_{s}"):
                                st.session_state["focus_node"] = s
                                st.rerun()
                        with cols[1]:
                            st.caption(s)
                else:
                    st.info("No sources found for this focus.")

            with right:
                st.subheader("Reads Me (Right)")
                if dependents:
                    for d in sorted(dependents):
                        cols = st.columns([0.35, 0.65])
                        with cols[0]:
                            if st.button(f"Focus {d} ▶", key=f"depbtn_{d}"):
                                st.session_state["focus_node"] = d
                                st.rerun()
                        with cols[1]:
                            st.caption(d)
                else:
                    st.info("No dependents found for this focus.")

        # st.markdown("---")
        # st.subheader(f"View: Sources → **{focus}** → Dependents")
        # st.markdown(_legend_html(), unsafe_allow_html=True)

        # # Render graph & expose download
        # html = render_pyvis_lr_band(G, focus, sources, dependents, height_px=graph_height)
        # st.components.v1.html(html, height=graph_height + 40, scrolling=True)


        st.markdown("---")
        st.subheader(f"View: Sources → **{focus}** → Dependents")
        st.markdown(_legend_html(), unsafe_allow_html=True)
        st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
        )
        renderer = st.radio(
            "Renderer",
            options=["Interactive (click to focus)", "Pretty (exportable HTML)"],
            index=0,
            horizontal=True,
            help="Interactive lets you click nodes to change focus; Pretty gives a polished PyVis export."
        )

        if renderer.startswith("Interactive"):
            clicked = render_agraph_lr_band(G, focus, sources, dependents, height_px=graph_height)
            if clicked and clicked in G.nodes:

                if clicked != st.session_state["focus_node"]:
                    st.session_state["focus_node"] = clicked
                    st.rerun()
        else:

            html = render_pyvis_lr_band(G, focus, sources, dependents, height_px=graph_height)
            st.components.v1.html(html, height=graph_height + 40, scrolling=True)
            with st.expander("Download graph as standalone HTML"):
                st.download_button(
                    label="Download .html",
                    data=html,
                    file_name=f"refviewer_{focus.replace('.', '_')}.html",
                    mime="text/html",
                )

else:
    with right:
        st.info("Paste SQL or upload a file to get started.")
