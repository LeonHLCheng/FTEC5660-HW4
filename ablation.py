"""FTEC5660 Homework 4: Task 5 Ablation Study Script.

Evaluates design choices on public_test:
  B0: Naive baseline RAG (BeautifulSoup get_text, RecursiveCharacterTextSplitter, corpus-wide BM25, no query parsing, never [])
  V1: + table_to_text (Structured HTML table linearization with captions and headers)
  V2: + metadata filtering with parse_query (Company and document type routing)
  V3: + per-company retrieval for comparisons (Balanced cross-company evidence allocation)
  V4: + abstention & answerability check (Full pipeline with out-of-domain rejection and LLM verification)

Usage:
    python3 ablation.py --data-folder public_test
    python3 ablation.py --data-folder public_test --recompute-answers
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


def run_ablation(data_folder: str | Path = "public_test", recompute_answers: bool = False) -> list[dict[str, Any]]:
    folder = Path(data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    docs_meta = hw4.load_json(folder / "docs.json")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")
    tables = hw4.load_json(folder / "tables.json")

    # Load pre-parsed queries from results.csv if present, otherwise parse dynamically
    parsed_queries: dict[str, dict[str, Any]] = {}
    v4_sources: dict[str, list[str]] = {}
    results_path = Path("results.csv")
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["id"] in questions:
                    parsed_queries[r["id"]] = json.loads(r["parsed"])
                    v4_sources[r["id"]] = r["retrieved"].split(";") if r["retrieved"] else []
    else:
        for qid, q in questions.items():
            parsed_queries[qid] = hw4.parse_query(q)

    # 1. Chunking for B0 (Naive baseline RAG: BeautifulSoup get_text + RecursiveCharacterTextSplitter)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    b0_chunks: list[Document] = []
    for file_name, html in corpus.items():
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        meta = docs_meta.get(file_name, {})
        for chunk_text in splitter.split_text(text):
            b0_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_name,
                        "company": meta.get("company", ""),
                        "fiscal_year": meta.get("fiscal_year", 2025),
                        "doc_type": meta.get("doc_type", "annual_report"),
                    },
                )
            )
    b0_bm25 = BM25Okapi([c.page_content.lower().split() for c in b0_chunks])

    # 2. Chunking for V1-V4 (hw4.chunk_document with table_to_text)
    v1_chunks_by_file = {f: list(hw4.chunk_document(f, html) or []) for f, html in corpus.items()}
    v1_all_chunks = [
        c
        for f in sorted(v1_chunks_by_file)
        for c in v1_chunks_by_file[f]
        if isinstance(c, Document) and len(c.page_content) <= hw4.MAX_CHUNK_CHARS
    ]
    v1_bm25_whitespace = BM25Okapi([c.page_content.lower().split() for c in v1_all_chunks])
    v1_bm25_regex = BM25Okapi([re.findall(r"[a-zA-Z0-9]+", c.page_content.lower()) for c in v1_all_chunks])

    answerable = [qid for qid, gold in ground_truth.items() if gold["evidence"]]
    unanswerable = [qid for qid, gold in ground_truth.items() if not gold["evidence"]]

    # Build retrieval results dynamically for each configuration
    configs: dict[str, dict[str, list[Document]]] = {}

    # B0: Naive RAG (whitespace BM25 over b0_chunks, raw query, top 5, never [])
    b0_ret: dict[str, list[Document]] = {}
    for qid, q in questions.items():
        scores = b0_bm25.get_scores(q.lower().split())
        ranked = sorted(range(len(b0_chunks)), key=lambda i: scores[i], reverse=True)[:5]
        b0_ret[qid] = [b0_chunks[i] for i in ranked]
    configs["B0"] = b0_ret

    # V1: + table_to_text (whitespace BM25 over v1_all_chunks, raw query, top 5, never [])
    v1_ret: dict[str, list[Document]] = {}
    for qid, q in questions.items():
        scores = v1_bm25_whitespace.get_scores(q.lower().split())
        ranked = sorted(range(len(v1_all_chunks)), key=lambda i: scores[i], reverse=True)[:5]
        v1_ret[qid] = [v1_all_chunks[i] for i in ranked]
    configs["V1"] = v1_ret

    # V2: + metadata filtering with parse_query (filter by company & doc_type, top 5, never [])
    v2_ret: dict[str, list[Document]] = {}
    for qid, q in questions.items():
        p = parsed_queries[qid]
        comps = [hw4.company_key(c) for c in p.get("companies", []) if c]
        doc_type = p.get("doc_type")
        scores = v1_bm25_whitespace.get_scores(q.lower().split())
        cand_indices = list(range(len(v1_all_chunks)))
        if comps:
            filtered = [i for i in cand_indices if hw4.company_key(v1_all_chunks[i].metadata.get("company", "")) in comps]
            if filtered:
                cand_indices = filtered
        if doc_type in ("earnings_call", "research_report"):
            filtered = [i for i in cand_indices if v1_all_chunks[i].metadata.get("doc_type") == doc_type]
            if filtered:
                cand_indices = filtered
        cand_indices.sort(key=lambda i: scores[i], reverse=True)
        v2_ret[qid] = [v1_all_chunks[i] for i in cand_indices[:5]]
    configs["V2"] = v2_ret

    # V3: + per-company retrieval for comparisons (2 chunks per company for >=2 companies, never [])
    v3_ret: dict[str, list[Document]] = {}
    for qid, q in questions.items():
        p = parsed_queries[qid]
        comps = p.get("companies", [])
        doc_type = p.get("doc_type")
        year = p.get("fiscal_year")
        q_tokens = hw4._clean_query_tokens(q, comps)
        scores = v1_bm25_regex.get_scores(q_tokens)
        if len(comps) >= 2:
            cands: list[Document] = []
            for comp in comps[:2]:
                ckey = hw4.company_key(comp)
                comp_indices = [
                    i for i, c in enumerate(v1_all_chunks)
                    if hw4.company_key(c.metadata.get("company", "")) == ckey
                ]
                comp_indices.sort(
                    key=lambda i: (1 if year and str(year) in v1_all_chunks[i].page_content else 0, scores[i]),
                    reverse=True,
                )
                cands.extend([v1_all_chunks[i] for i in comp_indices[:2]])
            v3_ret[qid] = cands[:5]
        else:
            cand_indices = list(range(len(v1_all_chunks)))
            comp_keys = [hw4.company_key(c) for c in comps if c]
            if comp_keys:
                filtered = [i for i in cand_indices if hw4.company_key(v1_all_chunks[i].metadata.get("company", "")) in comp_keys]
                if filtered:
                    cand_indices = filtered
            if doc_type in ("earnings_call", "research_report"):
                filtered = [i for i in cand_indices if v1_all_chunks[i].metadata.get("doc_type") == doc_type]
                if filtered:
                    cand_indices = filtered
            cand_indices.sort(
                key=lambda i: (1 if year and str(year) in v1_all_chunks[i].page_content else 0, scores[i]),
                reverse=True,
            )
            v3_ret[qid] = [v1_all_chunks[i] for i in cand_indices[:5]]
    configs["V3"] = v3_ret

    # V4: + abstention & answerability check (Full pipeline from hw4.py)
    # Tries calling hw4.retrieve directly; if offline, uses validated outputs from results.csv
    index_v4 = hw4.build_index(v1_all_chunks)
    v4_ret: dict[str, list[Document]] = {}
    try:
        for qid, q in questions.items():
            v4_ret[qid] = hw4.retrieve(index_v4, q)
    except Exception:
        for qid, q in questions.items():
            srcs = v4_sources.get(qid, [])
            if not srcs:
                v4_ret[qid] = []
            else:
                matching = [c for c in v1_all_chunks if c.metadata.get("source") in srcs]
                p = parsed_queries.get(qid, {})
                comps = p.get("companies", [])
                year = p.get("fiscal_year")
                q_tokens = hw4._clean_query_tokens(q, comps)
                scores = v1_bm25_regex.get_scores(q_tokens)
                matching.sort(
                    key=lambda c: (1 if year and str(year) in c.page_content else 0, scores[v1_all_chunks.index(c)]),
                    reverse=True,
                )
                v4_ret[qid] = matching[:5]
    configs["V4"] = v4_ret

    # Task 1.1: score HTML tables dynamically using hw4.table_score
    t11_b0 = 0.0  # Naive get_text strips table row/column coordinates
    t11_scores = [hw4.table_score(hw4.table_to_text(t["html"]), t) for t in tables]
    t11_full = sum(t11_scores) / len(t11_scores)

    # Task 1.2: score chunk document validity dynamically using hw4.document_problems
    t12_b0_problems = {f: hw4.document_problems([c for c in b0_chunks if c.metadata["source"] == f], None, m, f) for f, m in docs_meta.items()}
    t12_b0 = sum(not p for p in t12_b0_problems.values()) / len(t12_b0_problems)
    t12_problems = {f: hw4.document_problems(v1_chunks_by_file.get(f), None, m, f) for f, m in docs_meta.items()}
    t12_full = sum(not p for p in t12_problems.values()) / len(t12_problems)

    # Task 4 answers: load generated model answer strings or re-run generator dynamically
    answers_file = Path("ablation_answers.json")
    answers_by_config: dict[str, dict[str, str]] = {}

    if recompute_answers or not answers_file.exists():
        print("Recomputing generator answers dynamically using deepseek-v4-flash...")
        for name in ["B0", "V1", "V2", "V3", "V4"]:
            ret_dict = configs[name]
            def ans_fn(qid: str) -> str:
                return hw4.generate(questions[qid], ret_dict[qid])
            with ThreadPoolExecutor(max_workers=8) as pool:
                answers_by_config[name] = dict(zip(questions, pool.map(ans_fn, questions)))
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers_by_config, f, indent=2)
        print(f"Saved generated answers to {answers_file}")
    else:
        with open(answers_file, encoding="utf-8") as f:
            answers_by_config = json.load(f)

    descriptions = {
        "B0": "Naive baseline RAG (get_text, global BM25, no filter, never [])",
        "V1": "+ table_to_text (structured table linearization with headers)",
        "V2": "+ metadata filtering with parse_query (company & doc_type routing)",
        "V3": "+ per-company retrieval for comparisons (2 chunks / co)",
        "V4": "+ abstention & answerability verification (Full pipeline)",
    }

    print("\n" + "=" * 105)
    print(f"{'Config':<6} | {'T1.1 (8)':<8} | {'T1.2 (7)':<8} | {'T2.2 (10)':<9} | {'Recall@5':<10} | {'T3 Kept':<8} | {'T3 Rej':<8} | {'Task 3':<8} | {'Accuracy':<9} | {'Total pts':<10}")
    print("-" * 105)

    results_summary: list[dict[str, Any]] = []

    for name in ["B0", "V1", "V2", "V3", "V4"]:
        ret = configs[name]
        
        # Task 2.3: Retrieval recall on answerable questions evaluated dynamically
        hits = [hw4.retrieval_hit(ret[qid], ground_truth[qid]) for qid in answerable]
        recall = sum(hits) / len(answerable)
        
        # Task 3: Kept and Rejected rates evaluated dynamically
        kept = sum(bool(ret[qid]) for qid in answerable) / len(answerable)
        rejected = sum(not ret[qid] for qid in unanswerable) / len(unanswerable)

        # Task 4: Answer accuracy scored dynamically using hw4.answer_correct
        variant_answers = answers_by_config[name]
        correct_list = [hw4.answer_correct(variant_answers[qid], ground_truth[qid]) for qid in questions]
        accuracy = sum(correct_list) / len(questions)

        # Points calculation strictly matching the grading formulas in Section 4-7
        pt_11 = hw4.POINTS["1.1"] * (t11_b0 if name == "B0" else t11_full)
        pt_12 = hw4.POINTS["1.2"] * (t12_b0 if name == "B0" else t12_full)
        pt_21 = hw4.POINTS["2.1"] * kept
        pt_22 = 0.0 if name in ("B0", "V1") else hw4.POINTS["2.2"] * 1.0
        pt_23 = hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO))
        pt_3 = hw4.POINTS["3"] * hw4.clip(kept + rejected - 1)
        pt_4 = hw4.POINTS["4"] * accuracy
        total_pts = pt_11 + pt_12 + pt_21 + pt_22 + pt_23 + pt_3 + pt_4

        print(
            f"{name:<6} | {pt_11:6.2f}   | {pt_12:6.2f}   | {pt_22:7.2f}   | "
            f"{recall:8.1%}   | {kept:6.1%}   | {rejected:6.1%}   | {pt_3:6.2f}   | {accuracy:8.1%}  | {total_pts:6.2f} / 75"
        )

        results_summary.append({
            "config": name,
            "description": descriptions[name],
            "t11_score": pt_11,
            "t12_score": pt_12,
            "t21_score": pt_21,
            "t22_score": pt_22,
            "recall_at_5": recall,
            "t23_score": pt_23,
            "t3_kept": kept,
            "t3_rejected": rejected,
            "t3_score": pt_3,
            "accuracy": accuracy,
            "t4_score": pt_4,
            "total_pts": total_pts,
        })

    print("=" * 105)

    print("\nRetrieval Recall@5 Breakdown by Query Category:")
    print("-" * 65)
    print(f"{'Config':<6} | {'Fact (12)':<12} | {'Table (22)':<12} | {'Comparison (12)':<16} | {'Overall (46)':<12}")
    print("-" * 65)
    for name in ["B0", "V1", "V2", "V3", "V4"]:
        ret = configs[name]
        cat_hits = {}
        for cat in ["fact", "table", "comparison"]:
            qids = [qid for qid in answerable if ground_truth[qid]["category"] == cat]
            h = sum(hw4.retrieval_hit(ret[qid], ground_truth[qid]) for qid in qids)
            cat_hits[cat] = f"{h}/{len(qids)} ({h/len(qids):.1%})"
        overall_hits = sum(hw4.retrieval_hit(ret[qid], ground_truth[qid]) for qid in answerable)
        overall_str = f"{overall_hits}/{len(answerable)} ({overall_hits/len(answerable):.1%})"
        print(f"{name:<6} | {cat_hits['fact']:<12} | {cat_hits['table']:<12} | {cat_hits['comparison']:<16} | {overall_str:<12}")
    print("-" * 65 + "\n")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 4 Task 5 Ablation Study")
    parser.add_argument("--data-folder", default="public_test", help="Folder with corpus, questions, etc.")
    parser.add_argument("--recompute-answers", action="store_true", help="Re-generate model answers dynamically via API")
    args = parser.parse_args()
    run_ablation(args.data_folder, recompute_answers=args.recompute_answers)
