"""Project 2 page: index a repository, chat with it, and see exactly which code the answers come from."""

from __future__ import annotations

import streamlit as st

from common.errors import ChatbotError
from repo_qa.index.search import MODES
from repo_qa.service import RepoChat, index_repository, list_indexes

from . import resources
from .graph_canvas import render_interactive_graph_html
from .theme import confidence_pill


def _get_effort(raw: str | None) -> str:
    if not raw:
        return "high"
    lowered = raw.lower()
    if "high" in lowered:
        return "high"
    if "low" in lowered:
        return "low"
    return "medium"


def render() -> None:
    st.header("Project 2: Codebase Q&A")
    st.caption(
        "Prompt + RAG + knowledge graph. Ask about a repository: answers are built only from retrieved "
        "code, cite file and line numbers, and every citation is checked against what the model was shown."
    )
    cfg = resources.load_config()
    if cfg is None:
        return

    _index_section(cfg)
    indexes = list_indexes(cfg)
    if not indexes:
        st.info("No repository is indexed yet. Index one above to start chatting.")
        return

    pending = st.session_state.pop("repo_pending_select", None)
    if pending in [i.name for i in indexes]:
        st.session_state["repo_selected"] = pending
    labels = {i.name: f"{i.repo} ({i.chunks} chunks, {'with' if i.has_vectors else 'no'} vectors)" for i in indexes}
    selected = st.selectbox("Repository to chat with", list(labels), format_func=labels.get, key="repo_selected")
    chat = _open_chat(cfg, next(i for i in indexes if i.name == selected))
    if chat is None:
        return

    tab_chat, tab_graph = st.tabs(["💬 Codebase Chat", "🕸️ Knowledge Graph Structure"])
    with tab_chat:
        _options(chat)
        _graph_explorer(chat)
        _conversation(chat)
    with tab_graph:
        _render_graph_structure(chat)


# ---------------------------------------------------------------- indexing
def _index_section(cfg) -> None:
    last = st.session_state.pop("repo_last_report", None)
    if last is not None:
        st.success(f"Indexed {last.repo_name}: {last.chunks} chunks, "
                   f"{last.vectors_embedded} embedded now, {last.vectors_reused} reused.")
        for warning in last.warnings:
            st.warning(warning)
    with st.expander("Index a new repository", expanded=not list_indexes(cfg)):
        st.text_input("GitHub URL or local folder", key="repo_source",
                      placeholder="https://github.com/miguelgrinberg/microblog")
        left, right = st.columns(2)
        left.checkbox("Keyword-only (no embeddings, no Ollama needed)", key="repo_keyword_only")
        right.checkbox("Download again", key="repo_refresh")
        if st.button("Index repository", type="primary", key="repo_index_btn"):
            _run_indexing(cfg)


def _run_indexing(cfg) -> None:
    source = (st.session_state.get("repo_source") or "").strip()
    if not source:
        st.error("Enter a GitHub URL or a folder path first.")
        return
    status = st.status("Indexing ...", expanded=True)
    bar = status.progress(0.0)

    def stage(message: str) -> None:
        status.write(message)

    def progress(done: int, total: int) -> None:
        bar.progress(min(done / total, 1.0) if total else 1.0)

    try:
        report = index_repository(
            source, cfg, refresh=bool(st.session_state.get("repo_refresh")),
            keyword_only=bool(st.session_state.get("repo_keyword_only")),
            embedder_factory=resources.embedder_factory, progress=progress, stage=stage,
        )
    except (ChatbotError, ValueError) as exc:
        status.update(label="Indexing failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        return
    status.update(label="Done", state="complete", expanded=False)
    st.session_state["repo_last_report"] = report
    st.session_state["repo_pending_select"] = report.path.name
    st.rerun()


# ---------------------------------------------------------------- opening an index
def _open_chat(cfg, info) -> RepoChat | None:
    """The RepoChat for the selected index, opened once and kept for the browser session."""
    current = st.session_state.get("repo_chat")
    if current is not None and st.session_state.get("repo_chat_path") == str(info.path):
        return current
    if current is not None:
        current.close()
    for key in ("repo_chat", "repo_chat_path", "repo_messages"):
        st.session_state.pop(key, None)
    try:
        with st.spinner(f"Loading {info.repo} ..."):
            chat = RepoChat.open(info.path, cfg, chat_factory=resources.chat_factory,
                                 embedder_factory=resources.embedder_factory)
    except (ChatbotError, ValueError) as exc:
        st.error(f"Could not open this repository ({type(exc).__name__}): {exc}")
        return None
    st.session_state["repo_chat"] = chat
    st.session_state["repo_chat_path"] = str(info.path)
    st.session_state["repo_messages"] = []
    return chat


def _options(chat: RepoChat) -> None:
    counts = chat.index.graph.counts()
    st.caption(
        f"Knowledge graph: {sum(counts['nodes'].values())} nodes ({', '.join(f'{n} {t}' for t, n in counts['nodes'].items())}); "
        f"{sum(counts['edges'].values())} edges."
    )
    if not chat.index.has_vectors:
        st.info("This index has no embeddings, so only keyword search is available.")
    with st.container(border=True):
        st.markdown("**Search & Response Depth Settings**")
        top_col1, top_col2 = st.columns([1, 1])
        with top_col1:
            st.selectbox(
                "Response depth / Effort",
                ["High (Deep explanation & walkthrough)", "Medium (Balanced)", "Low (Fast & concise)"],
                index=0,
                key="repo_effort",
                help="High effort prompts the model for comprehensive architectural walkthroughs, step-by-step logic tracing, and code quotations.",
            )
        with top_col2:
            active_effort = _get_effort(st.session_state.get("repo_effort"))
            st.write("")
            st.caption(f"**Depth mode:** {active_effort.capitalize()} effort active")

        a, b, c, d = st.columns([2, 2, 2, 1])
        a.toggle("Use the knowledge graph", value=True, key="repo_use_graph",
                 help="Turn off to compare with plain search.")
        b.selectbox("Search mode", MODES, key="repo_mode")
        c.checkbox("Show what the model was given", key="repo_show_context")
        if d.button("Clear chat", key="repo_clear", use_container_width=True):
            chat.clear()
            st.session_state["repo_messages"] = []


def _graph_explorer(chat: RepoChat) -> None:
    graph = chat.index.graph
    with st.expander("Explore the knowledge graph"):
        name = st.text_input("Function or class name", key="repo_explore", placeholder="e.g. check_password")
        if not name.strip():
            return
        matches = [n for kind in ("Function", "Class") for n in graph.find(name.strip(), kind)]
        if not matches:
            st.write("No function or class with that name.")
        for node_id in matches[:5]:
            data = graph.node(node_id)
            st.markdown(f"**{graph.label(node_id)}** ({data['type'].lower()}, lines {data['start_line']}-{data['end_line']})")
            callers = graph.callers(node_id)
            callees = graph.callees(node_id)
            callers_str = ", ".join(f"`{graph.label(n)}` ({e['confidence']})" for n, e in callers[:8])
            callees_str = ", ".join(f"`{graph.label(n)}` ({e['confidence']})" for n, e in callees[:8])
            st.write("Called by: " + (callers_str or "nothing found"))
            st.write("Calls: " + (callees_str or "nothing found"))
            if data.get("decorators"):
                st.write("Decorators: " + ", ".join(f"`@{d}`" for d in data["decorators"]))


# ---------------------------------------------------------------- the conversation
def _conversation(chat: RepoChat) -> None:
    messages = st.session_state.setdefault("repo_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("answer") is not None:
                _show_sources(message["answer"], show_context=bool(st.session_state.get("repo_show_context")))
    question = st.chat_input("Ask about this repository, e.g. Where is authentication handled?", key="repo_chat_input")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        pieces: list[str] = []

        def on_token(piece: str) -> None:
            pieces.append(piece)
            placeholder.markdown("".join(pieces) + "▌")

        effort = _get_effort(st.session_state.get("repo_effort"))
        try:
            answer = chat.ask(
                question, use_graph=bool(st.session_state.get("repo_use_graph", True)),
                mode=st.session_state.get("repo_mode", "hybrid"), effort=effort, on_token=on_token,
            )
        except (ChatbotError, ValueError) as exc:
            placeholder.empty()
            message = f"{type(exc).__name__}: {exc}"
            st.error(message)
            # Keep the failed exchange on screen (the model's memory of the chat is not touched).
            messages.append({"role": "user", "content": question, "answer": None})
            messages.append({"role": "assistant", "content": f"**Error.** {message}", "answer": None, "error": True})
            return
        placeholder.markdown(answer.text)
        _show_sources(answer, show_context=bool(st.session_state.get("repo_show_context")))
    messages.append({"role": "user", "content": question, "answer": None})
    messages.append({"role": "assistant", "content": answer.text, "answer": answer})


def _show_sources(answer, *, show_context: bool) -> None:
    if answer.invalid_citations:
        st.warning("This answer cites locations that were not in its context, so treat them as unverified: "
                   + ", ".join(f"`{c}`" for c in answer.invalid_citations))
    if answer.standalone_question != answer.question:
        st.caption(f"Understood as: {answer.standalone_question}")
    retrieval = answer.retrieval
    with st.expander(f"Sources ({len(retrieval.items)})"):
        for item in retrieval.items:
            c = item.chunk
            st.markdown(f"- `{c.citation}` {c.kind} **{c.name}** _({item.origin}: {item.reason})_")
        if retrieval.dropped:
            st.caption(f"{retrieval.dropped} more related chunks did not fit the context budget.")
        if retrieval.facts:
            st.markdown("**Graph facts given to the model**")
            for fact in retrieval.facts:
                st.markdown(f"- {fact}")
        if show_context:
            from repo_qa.rag.prompts import format_context

            st.code(format_context(retrieval), language="markdown")


# ---------------------------------------------------------------- knowledge graph structure view
def _render_graph_structure(chat: RepoChat) -> None:
    graph = chat.index.graph
    counts = graph.counts()
    total_nodes = len(graph)
    total_edges = graph.g.number_of_edges()

    st.subheader("Knowledge Graph Structure")
    st.caption(
        "Tabular view of all code entities (nodes) and resolved relationships (edges) "
        "extracted during AST parsing and symbol analysis."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nodes", total_nodes)
    m2.metric("Total Edges", total_edges)
    m3.metric("Functions & Classes", counts["nodes"].get("Function", 0) + counts["nodes"].get("Class", 0))
    m4.metric("Source Files", counts["nodes"].get("File", 0))

    graph_tab_vis, graph_tab_hubs, graph_tab_path, graph_tab_nodes, graph_tab_edges, graph_tab_stats = st.tabs([
        "🕸️ Visual Graph Explorer",
        "🏛️ Architecture Hubs",
        "🔍 Call Chain & Path Finder",
        f"📦 Nodes Table ({total_nodes})",
        f"🔗 Edges Table ({total_edges})",
        "📊 Summary Statistics",
    ])

    with graph_tab_vis:
        st.markdown("**Interactive Knowledge Graph Visualization**")
        st.caption(
            "Explore nodes (functions, classes, files) and dependency relationships. "
            "Drag nodes to rearrange, scroll to zoom, click to highlight connections."
        )
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        with ctrl_col1:
            all_symbols = ["All (Top Architecture Hubs)"] + [graph.label(nid) for nid, d in graph.nodes() if d.get("type") in ("Function", "Class", "File")]
            seed_choice = ctrl_col1.selectbox("Focus / Seed Symbol", all_symbols, key="kg_vis_seed")
        with ctrl_col2:
            hops = ctrl_col2.slider("Neighborhood Hops", min_value=1, max_value=3, value=1, key="kg_vis_hops")
        with ctrl_col3:
            min_conf = ctrl_col3.selectbox("Min Confidence", ["guess", "likely", "certain"], index=0, key="kg_vis_conf")

        seeds = None
        if seed_choice != "All (Top Architecture Hubs)":
            matches = graph.find(seed_choice.split("::")[-1])
            if matches:
                seeds = matches[:2]

        subgraph_data = graph.subgraph_for_visualization(
            seeds=seeds,
            hops=hops,
            max_nodes=50,
            min_confidence=min_conf,
        )

        import streamlit.components.v1 as components
        canvas_html = render_interactive_graph_html(subgraph_data, height=540)
        components.html(canvas_html, height=560)

    with graph_tab_hubs:
        st.markdown("**Key Architecture Hubs**")
        st.caption("Symbols and files with the highest number of dependents (callers and importers).")

        hub_col1, hub_col2 = st.columns(2)
        with hub_col1:
            st.markdown("**Most Depended-On Functions**")
            fn_hubs = graph.hubs(node_types=["Function"], limit=10)
            if fn_hubs:
                fn_table = [{"Function": h["label"], "Callers (In)": h["in_degree"], "Calls (Out)": h["out_degree"], "File": h["path"]} for h in fn_hubs]
                st.dataframe(fn_table, use_container_width=True, hide_index=True)
            else:
                st.info("No functions with callers found.")

        with hub_col2:
            st.markdown("**Most Depended-On Classes & Files**")
            cls_hubs = graph.hubs(node_types=["Class", "File"], limit=10)
            if cls_hubs:
                cls_table = [{"Symbol / File": h["label"], "Type": h["type"], "Dependents (In)": h["in_degree"], "Dependencies (Out)": h["out_degree"]} for h in cls_hubs]
                st.dataframe(cls_table, use_container_width=True, hide_index=True)
            else:
                st.info("No classes or files with dependencies found.")

    with graph_tab_path:
        st.markdown("**Dependency Path Finder & Call Chain Tracing**")
        st.caption("Trace how two functions, classes, or files connect through calls, contains, and imports.")

        path_c1, path_c2, path_c3 = st.columns([2, 2, 1])
        candidate_names = sorted(list({graph.label(nid).split("::")[-1] for nid, _ in graph.nodes()}))
        with path_c1:
            src_symbol = path_c1.selectbox("Source Symbol", candidate_names, key="kg_path_src")
        with path_c2:
            default_dst_idx = min(1, len(candidate_names) - 1) if candidate_names else 0
            dst_symbol = path_c2.selectbox("Target Symbol", candidate_names, index=default_dst_idx, key="kg_path_dst")
        with path_c3:
            directed_opt = path_c3.checkbox("Directed Call Flow", value=True, key="kg_path_dir", help="Uncheck to find any connection even against call direction.")

        if st.button("Find Dependency Path", key="kg_path_btn"):
            if src_symbol == dst_symbol:
                st.info("Source and target are the same symbol.")
            else:
                trace = graph.shortest_path(src_symbol, dst_symbol, directed=directed_opt)
                if not trace and directed_opt:
                    trace = graph.shortest_path(src_symbol, dst_symbol, directed=False)
                    if trace:
                        st.info("No direct forward call flow was found, but an indirect connection exists:")

                if trace:
                    st.success(f"Path found ({trace['hops']} hops):")
                    for i, step in enumerate(trace["steps"], 1):
                        arrow = "➔" if step["direction"] == "outgoing" else "⬅️"
                        details = f" _({step['details']})_" if step["details"] else ""
                        st.markdown(
                            f"**{i}.** `{step['from_label']}` {arrow} **[{step['type']} - {step['confidence']}]** {arrow} `{step['to_label']}`{details}"
                        )
                else:
                    st.warning(f"No connection found between `{src_symbol}` and `{dst_symbol}`.")

    with graph_tab_nodes:
        col_filter1, col_filter2 = st.columns([1, 2])
        node_types = ["All"] + sorted(list(counts["nodes"].keys()))
        selected_type = col_filter1.selectbox("Filter by Node Type", node_types, key="kg_node_type_filter")
        search_query = col_filter2.text_input("Search nodes by symbol name or path", key="kg_node_search_query")

        node_rows = []
        for node_id, data in graph.nodes():
            ntype = data.get("type", "")
            if selected_type != "All" and ntype != selected_type:
                continue
            path = data.get("path") or ""
            symbol = data.get("qualname") or data.get("name") or data.get("title") or ""
            if search_query:
                q = search_query.lower()
                if q not in node_id.lower() and q not in symbol.lower() and q not in path.lower():
                    continue
            lines = ""
            if "start_line" in data and "end_line" in data:
                lines = f"{data['start_line']}-{data['end_line']}"
            details = ""
            if data.get("decorators"):
                details = ", ".join(f"@{d}" for d in data["decorators"])
            elif ntype == "External":
                details = "External library"
            node_rows.append({
                "Node ID": node_id,
                "Type": ntype,
                "Symbol / Name": symbol,
                "File Path": path,
                "Lines": lines,
                "Details": details,
            })

        st.caption(f"Showing **{len(node_rows)}** of {total_nodes} nodes")
        st.dataframe(node_rows, use_container_width=True, hide_index=True)

    with graph_tab_edges:
        col_e1, col_e2, col_e3 = st.columns([1, 1, 2])
        edge_types = ["All"] + sorted(list(counts["edges"].keys()))
        selected_edge_type = col_e1.selectbox("Filter by Edge Type", edge_types, key="kg_edge_type_filter")
        conf_levels = ["All", "certain", "likely", "guess"]
        selected_conf = col_e2.selectbox("Filter by Confidence", conf_levels, key="kg_conf_filter")
        edge_search = col_e3.text_input("Search edges by symbol or label", key="kg_edge_search_query")

        edge_rows = []
        for src, dst, data in graph.edges():
            etype = data.get("type", "")
            if selected_edge_type != "All" and etype != selected_edge_type:
                continue
            conf = data.get("confidence", "")
            if selected_conf != "All" and conf != selected_conf:
                continue
            src_label = graph.label(src)
            dst_label = graph.label(dst)
            if edge_search:
                q = edge_search.lower()
                if q not in src_label.lower() and q not in dst_label.lower() and q not in src.lower() and q not in dst.lower():
                    continue
            details = ""
            if data.get("names"):
                details = f"Calls: {', '.join(data['names'])}"
            elif data.get("via"):
                details = f"Via: {data['via']}"
            edge_rows.append({
                "Source": src_label,
                "Relationship": etype,
                "Target": dst_label,
                "Confidence": conf,
                "Details": details,
                "Source ID": src,
                "Target ID": dst,
            })

        st.caption(f"Showing **{len(edge_rows)}** of {total_edges} edges")
        st.dataframe(edge_rows, use_container_width=True, hide_index=True)

    with graph_tab_stats:
        stat_c1, stat_c2 = st.columns(2)
        with stat_c1:
            st.markdown("**Node Counts by Type**")
            node_counts_data = [{"Node Type": t, "Count": c} for t, c in counts["nodes"].items()]
            st.dataframe(node_counts_data, use_container_width=True, hide_index=True)
        with stat_c2:
            st.markdown("**Edge Counts by Type**")
            edge_counts_data = [{"Edge Type": t, "Count": c} for t, c in counts["edges"].items()]
            st.dataframe(edge_counts_data, use_container_width=True, hide_index=True)

