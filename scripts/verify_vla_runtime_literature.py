#!/usr/bin/env python3
"""Refresh and validate a bounded VLA/runtime literature metadata artifact.

The catalog is intentionally identifier-first. Formal publication status is
seeded only when an official venue or publisher record is known; an arXiv DOI
never promotes a preprint to a formal publication. Live requests verify the
frozen identities and record failures instead of silently dropping papers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
from html.parser import HTMLParser
import json
import os
import pathlib
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


SCHEMA_VERSION = "armbench.vla_runtime_literature_metadata.v1"
DEFAULT_OUTPUT = pathlib.Path("docs/research/vla_runtime_literature_metadata.json")
USER_AGENT = "armbench-vla-runtime-literature-audit/1.0"
REQUIRED_KEYS = frozenset(
    {
        "rt2",
        "openvla",
        "openvla_oft",
        "fast",
        "conrft",
        "dppo",
        "hil_serl",
        "rtc",
        "vlash",
        "future_rtc",
        "action_controlnet",
    }
)


CATALOG: Tuple[Dict[str, Any], ...] = (
    {
        "key": "rt2",
        "title": "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
        "arxiv_id": "2307.15818",
        "status": "formally_published",
        "formal": {
            "venue": "Conference on Robot Learning (CoRL)",
            "venue_year": 2023,
            "publication_year": 2023,
            "publication_date": "2023-12-02",
            "doi": None,
            "official_url": "https://proceedings.mlr.press/v229/zitkovich23a.html",
            "citation": "Proceedings of The 7th Conference on Robot Learning, PMLR 229:2165-2183",
        },
        "method_class": "generalist_vla_pretraining",
        "training_regime": "vision-language-action co-fine-tuning",
        "project_relation": "Closed-weight historical VLA reference; not a runnable ArmBench baseline.",
    },
    {
        "key": "openvla",
        "title": "OpenVLA: An Open-Source Vision-Language-Action Model",
        "arxiv_id": "2406.09246",
        "status": "formally_published",
        "formal": {
            "venue": "Conference on Robot Learning (CoRL 2024)",
            "venue_year": 2024,
            "publication_year": 2025,
            "publication_date": "2025-01-12",
            "doi": None,
            "official_url": "https://proceedings.mlr.press/v270/kim25c.html",
            "citation": "Proceedings of The 8th Conference on Robot Learning, PMLR 270:2679-2713",
        },
        "method_class": "open_generalist_vla",
        "training_regime": "supervised VLA pretraining and adaptation",
        "project_relation": "Open checkpoint family suitable for a cross-model runtime baseline when compute permits.",
    },
    {
        "key": "openvla_oft",
        "title": "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success",
        "arxiv_id": "2502.19645",
        "status": "formally_published",
        "formal": {
            "venue": "Robotics: Science and Systems XXI (RSS 2025)",
            "venue_year": 2025,
            "publication_year": 2025,
            "publication_date": "2025-06-21",
            "doi": "10.15607/RSS.2025.XXI.017",
            "official_url": "https://www.roboticsproceedings.org/rss21/p017.html",
            "citation": "Robotics: Science and Systems XXI, paper 17",
        },
        "method_class": "vla_supervised_finetuning",
        "training_regime": "supervised optimized fine-tuning with action chunking",
        "project_relation": "Strong trained OpenVLA comparator; separates policy adaptation gains from runtime-only gains.",
    },
    {
        "key": "fast",
        "title": "FAST: Efficient Action Tokenization for Vision-Language-Action Models",
        "arxiv_id": "2501.09747",
        "status": "formally_published",
        "formal": {
            "venue": "Robotics: Science and Systems XXI (RSS 2025)",
            "venue_year": 2025,
            "publication_year": 2025,
            "publication_date": "2025-06-21",
            "doi": "10.15607/RSS.2025.XXI.012",
            "official_url": "https://www.roboticsproceedings.org/rss21/p012.html",
            "citation": "Robotics: Science and Systems XXI, paper 12",
        },
        "method_class": "action_tokenization",
        "training_regime": "learned action tokenizer plus VLA training",
        "project_relation": "Training-side efficiency reference; it does not itself guarantee asynchronous runtime alignment.",
    },
    {
        "key": "conrft",
        "title": "ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy",
        "arxiv_id": "2502.05450",
        "status": "formally_published",
        "formal": {
            "venue": "Robotics: Science and Systems XXI (RSS 2025)",
            "venue_year": 2025,
            "publication_year": 2025,
            "publication_date": "2025-06-21",
            "doi": "10.15607/RSS.2025.XXI.019",
            "official_url": "https://www.roboticsproceedings.org/rss21/p019.html",
            "citation": "Robotics: Science and Systems XXI, paper 19",
        },
        "method_class": "vla_reinforcement_finetuning",
        "training_regime": "offline and online reinforced fine-tuning with interventions",
        "project_relation": "Direct answer to the RL comparison: higher training and real-robot cost than ArmBench's runtime-only intervention.",
    },
    {
        "key": "dppo",
        "title": "Diffusion Policy Policy Optimization",
        "arxiv_id": "2409.00588",
        "status": "formally_published",
        "formal": {
            "venue": "International Conference on Learning Representations (ICLR 2025)",
            "venue_year": 2025,
            "publication_year": 2025,
            "publication_date": None,
            "doi": None,
            "official_url": "https://iclr.cc/virtual/2025/poster/28475",
            "citation": "ICLR 2025 poster; OpenReview forum mEpqHvbD2h",
            "openreview_id": "mEpqHvbD2h",
        },
        "method_class": "diffusion_policy_reinforcement_learning",
        "training_regime": "policy-gradient fine-tuning of diffusion policies",
        "project_relation": "RL policy-optimization baseline, not a VLA runtime-assurance method.",
    },
    {
        "key": "hil_serl",
        "title": "Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning",
        "arxiv_id": "2410.21845",
        "status": "formally_published",
        "formal": {
            "venue": "Science Robotics",
            "venue_year": 2025,
            "publication_year": 2025,
            "publication_date": "2025-08-20",
            "doi": "10.1126/scirobotics.ads5033",
            "official_url": "https://doi.org/10.1126/scirobotics.ads5033",
            "citation": "Science Robotics 10(105): eads5033",
        },
        "method_class": "human_in_the_loop_real_world_rl",
        "training_regime": "online real-robot reinforcement learning with human interventions",
        "project_relation": "Demonstrates high-value RL evidence but requires human supervision and substantial real-robot data.",
    },
    {
        "key": "rtc",
        "title": "Real-Time Execution of Action Chunking Flow Policies",
        "arxiv_id": "2506.07339",
        "status": "formally_published",
        "formal": {
            "venue": "Conference on Neural Information Processing Systems (NeurIPS 2025)",
            "venue_year": 2025,
            "publication_year": 2025,
            "publication_date": None,
            "doi": None,
            "official_url": "https://neurips.cc/virtual/2025/poster/117747",
            "citation": "NeurIPS 2025 poster; OpenReview forum UkR2zO5uww",
            "openreview_id": "UkR2zO5uww",
        },
        "method_class": "training_free_asynchronous_runtime",
        "training_regime": "inference-time flow-policy inpainting without retraining",
        "project_relation": "Closest formal runtime comparator; ArmBench must distinguish prefix dropping from flow inpainting and reproduce latency protocols.",
    },
    {
        "key": "vlash",
        "title": "VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference",
        "arxiv_id": "2512.01031",
        "status": "preprint_only_as_of_access_date",
        "formal": None,
        "method_class": "future_state_asynchronous_vla",
        "training_regime": "future-state-aware asynchronous inference",
        "project_relation": "Direct recent comparator for stale-observation and prediction-execution misalignment.",
    },
    {
        "key": "future_rtc",
        "title": "FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking",
        "arxiv_id": "2607.24008",
        "status": "preprint_only_as_of_access_date",
        "formal": None,
        "method_class": "learned_execution_time_context_prediction",
        "training_regime": "plug-in state/visual prediction modules with policy-consistency adaptation",
        "project_relation": "Very recent learned-adapter comparator; broader than ArmBench's current deterministic temporal alignment.",
    },
    {
        "key": "action_controlnet",
        "title": "Action ControlNet: A Lightweight Delay-Aware Adapter for Smooth Asynchronous Control in Vision-Language-Action Models",
        "arxiv_id": "2606.25985",
        "status": "preprint_only_as_of_access_date",
        "formal": None,
        "method_class": "delay_aware_action_adapter",
        "training_regime": "parameter-efficient adapter training with a mostly frozen backbone",
        "project_relation": "Learned adapter comparator for action handoff smoothness; not training-free despite leaving the backbone frozen.",
    },
)


class _CitationMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.citation_titles: List[str] = []
        self.authors: List[str] = []
        self.in_title = False
        self.title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = {key.lower(): value for key, value in attrs}
        if tag.lower() == "meta":
            name = (values.get("name") or "").lower()
            content = values.get("content") or ""
            if name == "citation_title":
                self.citation_titles.append(content)
            elif name == "citation_author":
                self.authors.append(content)
        elif tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(value).lower()).strip()


def _title_matches(expected: str, observed: str) -> bool:
    left = _normalize_title(expected)
    right = _normalize_title(observed)
    return bool(left and (left == right or left in right or right in left))


class Retriever:
    def __init__(self, access_date: str, timeout_s: float) -> None:
        self.access_date = access_date
        self.timeout_s = timeout_s
        self.requests: List[Dict[str, Any]] = []

    @staticmethod
    def _redact(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        clean = [(key, "REDACTED" if key == "api_key" else value) for key, value in query]
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(clean), parsed.fragment)
        )

    def fetch(
        self, source: str, url: str, purpose: str, record_key: Optional[str] = None
    ) -> Optional[bytes]:
        attempts = 0
        last_error: Optional[BaseException] = None
        last_status: Optional[int] = None
        final_url: Optional[str] = None
        while attempts < 2:
            attempts += 1
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml, text/html;q=0.9, */*;q=0.5"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    data = response.read()
                    last_status = int(response.status)
                    final_url = response.geturl()
                self.requests.append(
                    {
                        "source": source,
                        "record_key": record_key,
                        "purpose": purpose,
                        "endpoint": self._redact(url),
                        "accessed_on": self.access_date,
                        "outcome": "success",
                        "http_status": last_status,
                        "attempts": attempts,
                        "final_url": self._redact(final_url),
                        "bytes_received": len(data),
                        "error_type": None,
                        "error_message": None,
                    }
                )
                return data
            except urllib.error.HTTPError as exc:
                last_error = exc
                last_status = exc.code
                if exc.code in (429, 503) and attempts == 1:
                    time.sleep(2.0)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                break
        self.requests.append(
            {
                "source": source,
                "record_key": record_key,
                "purpose": purpose,
                "endpoint": self._redact(url),
                "accessed_on": self.access_date,
                "outcome": "failure",
                "http_status": last_status,
                "attempts": attempts,
                "final_url": final_url,
                "bytes_received": 0,
                "error_type": type(last_error).__name__ if last_error else "UnknownError",
                "error_message": str(last_error) if last_error else "request failed without an exception",
            }
        )
        return None

    def fetch_json(
        self, source: str, url: str, purpose: str, record_key: Optional[str] = None
    ) -> Optional[Mapping[str, Any]]:
        data = self.fetch(source, url, purpose, record_key)
        if data is None:
            return None
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            leading = data.lstrip()[:1]
            explanation = (
                "expected JSON but endpoint returned HTML, likely an access challenge"
                if leading == b"<"
                else str(exc)
            )
            self.mark_parse_failure(source, url, purpose, record_key, exc, explanation)
            return None
        return value if isinstance(value, Mapping) else None

    def mark_parse_failure(
        self,
        source: str,
        url: str,
        purpose: str,
        record_key: Optional[str],
        error: BaseException,
        explanation: Optional[str] = None,
    ) -> None:
        endpoint = self._redact(url)
        if self.requests and self.requests[-1].get("endpoint") == endpoint:
            record = self.requests[-1]
            record["outcome"] = "failure"
            record["purpose"] = purpose + " response parsing"
            record["error_type"] = type(error).__name__
            record["error_message"] = explanation or str(error)
            return
        self.requests.append(
            {
                "source": source,
                "record_key": record_key,
                "purpose": purpose + " response parsing",
                "endpoint": endpoint,
                "accessed_on": self.access_date,
                "outcome": "failure",
                "http_status": None,
                "attempts": 0,
                "final_url": None,
                "bytes_received": 0,
                "error_type": type(error).__name__,
                "error_message": explanation or str(error),
            }
        )


def _arxiv_records(retriever: Retriever) -> Dict[str, Dict[str, Any]]:
    ids = [str(item["arxiv_id"]) for item in CATALOG]
    query = urllib.parse.urlencode(
        {"id_list": ",".join(ids), "start": 0, "max_results": len(ids)}
    )
    url = "https://export.arxiv.org/api/query?" + query
    data = retriever.fetch("arXiv", url, "batch identifier lookup")
    if data is None:
        return {}
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        retriever.mark_parse_failure(
            "arXiv", url, "Atom batch", None, exc, "arXiv response was not valid Atom XML"
        )
        return {}
    atom = "{http://www.w3.org/2005/Atom}"
    arxiv = "{http://arxiv.org/schemas/atom}"
    records: Dict[str, Dict[str, Any]] = {}
    for entry in root.findall(atom + "entry"):
        raw_id = entry.findtext(atom + "id", default="")
        match = re.search(r"/abs/([^v]+)(?:v[0-9]+)?$", raw_id)
        if not match:
            continue
        arxiv_id = match.group(1)
        primary = entry.find(arxiv + "primary_category")
        records[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "versioned_url": raw_id.replace("http://", "https://"),
            "title": " ".join(entry.findtext(atom + "title", default="").split()),
            "authors": [
                " ".join(author.findtext(atom + "name", default="").split())
                for author in entry.findall(atom + "author")
            ],
            "submitted_date": entry.findtext(atom + "published"),
            "updated_date": entry.findtext(atom + "updated"),
            "primary_category": primary.get("term") if primary is not None else None,
            "comment": entry.findtext(arxiv + "comment"),
            "journal_reference": entry.findtext(arxiv + "journal_ref"),
        }
    return records


def _crossref_record(retriever: Retriever, key: str, doi: str) -> Optional[Dict[str, Any]]:
    encoded = urllib.parse.quote(doi, safe="")
    query = ""
    mailto = os.environ.get("CROSSREF_MAILTO")
    if mailto:
        query = "?" + urllib.parse.urlencode({"mailto": mailto})
    url = "https://api.crossref.org/works/%s%s" % (encoded, query)
    value = retriever.fetch_json("Crossref", url, "formal DOI lookup", key)
    message = value.get("message") if isinstance(value, Mapping) else None
    if not isinstance(message, Mapping):
        return None
    title = message.get("title")
    container = message.get("container-title")
    authors = []
    for author in message.get("author", []):
        if isinstance(author, Mapping):
            authors.append(" ".join(str(author.get(field, "")) for field in ("given", "family")).strip())
    return {
        "doi": message.get("DOI"),
        "title": title[0] if isinstance(title, list) and title else None,
        "container_title": container[0] if isinstance(container, list) and container else None,
        "type": message.get("type"),
        "publisher": message.get("publisher"),
        "authors": authors,
        "url": message.get("URL"),
    }


def _openalex_record(retriever: Retriever, item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    formal = item.get("formal")
    doi = formal.get("doi") if isinstance(formal, Mapping) else None
    lookup_doi = str(doi or ("10.48550/arXiv.%s" % item["arxiv_id"]))
    identifier = urllib.parse.quote("doi:" + lookup_doi, safe=":/")
    params: Dict[str, str] = {
        "select": "id,doi,title,publication_year,publication_date,type,primary_location,ids,is_retracted"
    }
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    url = "https://api.openalex.org/works/%s?%s" % (
        identifier,
        urllib.parse.urlencode(params),
    )
    value = retriever.fetch_json("OpenAlex", url, "identifier lookup", str(item["key"]))
    if not isinstance(value, Mapping):
        return None
    location = value.get("primary_location")
    source = location.get("source") if isinstance(location, Mapping) else None
    return {
        "openalex_id": value.get("id"),
        "doi": value.get("doi"),
        "title": value.get("title"),
        "publication_year": value.get("publication_year"),
        "publication_date": value.get("publication_date"),
        "type": value.get("type"),
        "source": source.get("display_name") if isinstance(source, Mapping) else None,
        "is_retracted": value.get("is_retracted"),
    }


def _official_record(retriever: Retriever, item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    formal = item.get("formal")
    if not isinstance(formal, Mapping):
        return None
    url = str(formal["official_url"])
    data = retriever.fetch("official_venue_or_publisher", url, "formal publication lookup", str(item["key"]))
    if data is None:
        return None
    text = data.decode("utf-8", errors="replace")
    parser = _CitationMetaParser()
    parser.feed(text)
    extracted_title = parser.citation_titles[0] if parser.citation_titles else " ".join(parser.title_parts).strip()
    return {
        "requested_url": url,
        "extracted_title": extracted_title or None,
        "title_match": _title_matches(str(item["title"]), extracted_title or text),
        "citation_authors": parser.authors,
    }


def _openreview_probe(retriever: Retriever) -> Optional[Dict[str, Any]]:
    forum_id = "mEpqHvbD2h"
    url = "https://api2.openreview.net/notes?" + urllib.parse.urlencode({"id": forum_id})
    value = retriever.fetch_json(
        "OpenReview", url, "supplemental ICLR forum lookup", "dppo"
    )
    if not isinstance(value, Mapping):
        return None
    notes = value.get("notes")
    return {"forum_id": forum_id, "notes_returned": len(notes) if isinstance(notes, list) else None}


def _verification(
    item: Mapping[str, Any],
    arxiv_records: Mapping[str, Mapping[str, Any]],
    retriever: Retriever,
    offline: bool,
) -> Dict[str, Any]:
    if offline:
        return {
            "mode": "offline_catalog_only",
            "arxiv": None,
            "official": None,
            "crossref": None,
            "openalex": None,
            "title_checks": {},
        }
    arxiv_record = arxiv_records.get(str(item["arxiv_id"]))
    formal = item.get("formal")
    doi = formal.get("doi") if isinstance(formal, Mapping) else None
    official = _official_record(retriever, item)
    crossref = _crossref_record(retriever, str(item["key"]), str(doi)) if doi else None
    openalex = _openalex_record(retriever, item)
    checks = {
        "arxiv_title_match": _title_matches(str(item["title"]), str(arxiv_record.get("title", "")))
        if arxiv_record
        else None,
        "official_title_match": official.get("title_match") if official else None,
        "crossref_title_match": _title_matches(str(item["title"]), str(crossref.get("title", "")))
        if crossref
        else None,
        "openalex_title_match": _title_matches(str(item["title"]), str(openalex.get("title", "")))
        if openalex
        else None,
    }
    return {
        "mode": "live_identifier_verification",
        "arxiv": arxiv_record,
        "official": official,
        "crossref": crossref,
        "openalex": openalex,
        "title_checks": checks,
    }


def build_artifact(access_date: str, timeout_s: float, offline: bool) -> Dict[str, Any]:
    retriever = Retriever(access_date, timeout_s)
    arxiv_records = {} if offline else _arxiv_records(retriever)
    records: List[Dict[str, Any]] = []
    for seed in CATALOG:
        record = {
            "key": seed["key"],
            "canonical_title": seed["title"],
            "status_as_of_access_date": seed["status"],
            "method_class": seed["method_class"],
            "training_regime": seed["training_regime"],
            "project_relation": seed["project_relation"],
            "formal_publication": seed["formal"],
            "preprint_companion": {
                "server": "arXiv",
                "arxiv_id": seed["arxiv_id"],
                "abstract_url": "https://arxiv.org/abs/%s" % seed["arxiv_id"],
                "arxiv_doi": "10.48550/arXiv.%s" % seed["arxiv_id"],
                "note": "An arXiv DOI identifies the preprint and is not treated as a formal venue DOI.",
            },
            "verification": _verification(seed, arxiv_records, retriever, offline),
        }
        records.append(record)
    supplemental = None if offline else _openreview_probe(retriever)
    formal = [record for record in records if record["formal_publication"] is not None]
    preprints = [record for record in records if record["formal_publication"] is None]
    failures = [request for request in retriever.requests if request["outcome"] == "failure"]
    warnings: List[str] = []
    for record in records:
        checks = record["verification"]["title_checks"]
        for source, matched in checks.items():
            if matched is False:
                warnings.append("%s: %s failed" % (record["key"], source))
    observed = {str(record["key"]) for record in records}
    generator_path = pathlib.Path(__file__).resolve()
    generator_sha256 = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "access_date": access_date,
        "retrieval_contract": {
            "scope": "bounded identifier-first audit of VLA training, RL fine-tuning, and asynchronous runtime comparators",
            "classification_rule": (
                "A work is formally published only when a seeded official venue/publisher record exists. "
                "arXiv-only works stay separate even when Crossref or OpenAlex exposes the arXiv DOI."
            ),
            "network_mode": "offline" if offline else "live",
            "required_keys": sorted(REQUIRED_KEYS),
            "observed_keys": sorted(observed),
            "missing_required_keys": sorted(REQUIRED_KEYS - observed),
            "formal_publication_count": len(formal),
            "preprint_only_count": len(preprints),
        },
        "formal_publications": formal,
        "preprints_only": preprints,
        "provenance": {
            "generator": {
                "path": "scripts/verify_vla_runtime_literature.py",
                "sha256": generator_sha256,
            },
            "preferred_source_order": [
                "official venue or publisher",
                "Crossref for formal DOI metadata",
                "OpenAlex for identifier reconciliation",
                "arXiv for preprint version metadata",
            ],
            "endpoint_documentation": {
                "arxiv": "https://export.arxiv.org/api/query",
                "crossref": "https://api.crossref.org/works/{doi}",
                "openalex": "https://api.openalex.org/works/{id}",
                "openreview_supplemental": "https://api2.openreview.net/notes?id={forum_id}",
            },
            "requests": retriever.requests,
            "api_failures": failures,
            "validation_warnings": warnings,
            "supplemental_openreview_probe": supplemental,
            "environment_keys_used": {
                "OPENALEX_API_KEY": bool(os.environ.get("OPENALEX_API_KEY")),
                "CROSSREF_MAILTO": bool(os.environ.get("CROSSREF_MAILTO")),
            },
        },
        "claim_boundary": (
            "This is a targeted, reproducible metadata audit, not an exhaustive literature review. "
            "Publication status is frozen to the access date and must be refreshed before external use."
        ),
    }


def validate_artifact(value: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return ["root must be a JSON object"]
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    formal = value.get("formal_publications")
    preprints = value.get("preprints_only")
    if not isinstance(formal, list) or not isinstance(preprints, list):
        return errors + ["formal_publications and preprints_only must be arrays"]
    records = formal + preprints
    keys = [record.get("key") for record in records if isinstance(record, Mapping)]
    if len(keys) != len(set(keys)):
        errors.append("paper keys are not unique")
    missing = REQUIRED_KEYS - set(keys)
    if missing:
        errors.append("missing required papers: %s" % ", ".join(sorted(missing)))
    expected_formal = {
        str(item["key"]) for item in CATALOG if item["status"] == "formally_published"
    }
    expected_preprints = REQUIRED_KEYS - expected_formal
    observed_formal = {
        str(record.get("key")) for record in formal if isinstance(record, Mapping)
    }
    observed_preprints = {
        str(record.get("key")) for record in preprints if isinstance(record, Mapping)
    }
    if observed_formal != expected_formal:
        errors.append("formal publication classification differs from the frozen catalog")
    if observed_preprints != expected_preprints:
        errors.append("preprint-only classification differs from the frozen catalog")
    for record in formal:
        if not isinstance(record, Mapping) or not isinstance(record.get("formal_publication"), Mapping):
            errors.append("formal publication entry lacks formal metadata")
        elif record.get("status_as_of_access_date") != "formally_published":
            errors.append("formal publication entry has wrong status: %r" % record.get("key"))
    for record in preprints:
        if not isinstance(record, Mapping) or record.get("formal_publication") is not None:
            errors.append("preprint-only entry contains formal metadata")
        elif record.get("status_as_of_access_date") != "preprint_only_as_of_access_date":
            errors.append("preprint-only entry has wrong status: %r" % record.get("key"))
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        generator = provenance.get("generator")
        expected_generator_sha256 = hashlib.sha256(
            pathlib.Path(__file__).resolve().read_bytes()
        ).hexdigest()
        if not isinstance(generator, Mapping):
            errors.append("provenance.generator must be an object")
        elif generator.get("path") != "scripts/verify_vla_runtime_literature.py":
            errors.append("generator path mismatch")
        elif generator.get("sha256") != expected_generator_sha256:
            errors.append("generator SHA-256 does not match the current script")
        requests = provenance.get("requests")
        if not isinstance(requests, list):
            errors.append("provenance.requests must be an array")
        else:
            for index, request in enumerate(requests):
                required = {"source", "endpoint", "accessed_on", "outcome"}
                if not isinstance(request, Mapping) or not required <= set(request):
                    errors.append("request %d lacks reproducibility fields" % index)
    return errors


def _write_json_atomic(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=str(path.parent), prefix=".%s." % path.name, suffix=".tmp"
    )
    temporary = pathlib.Path(handle.name)
    try:
        with handle:
            handle.write(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_access_date(value: str) -> str:
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("access date must be YYYY-MM-DD") from exc
    return parsed.isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--access-date", type=_parse_access_date, default=dt.datetime.now(dt.timezone.utc).date().isoformat())
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--offline", action="store_true", help="Write the seeded catalog without network verification")
    parser.add_argument(
        "--check-only",
        "--validate-only",
        dest="check_only",
        action="store_true",
        help="Validate the existing output without network access or writes",
    )
    parser.add_argument("--strict-network", action="store_true", help="Return nonzero when any live endpoint fails or any title check mismatches")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output.resolve()
    if args.timeout_s <= 0.0:
        raise SystemExit("--timeout-s must be positive")
    if args.check_only:
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            print(json.dumps({"valid": False, "output": str(output), "errors": [str(exc)]}, indent=2))
            return 2
    else:
        value = build_artifact(args.access_date, args.timeout_s, args.offline)
        errors = validate_artifact(value)
        if errors:
            print(json.dumps({"valid": False, "output": str(output), "errors": errors}, indent=2))
            return 2
        _write_json_atomic(output, value)
    errors = validate_artifact(value)
    failures = value.get("provenance", {}).get("api_failures", []) if isinstance(value, Mapping) else []
    warnings = value.get("provenance", {}).get("validation_warnings", []) if isinstance(value, Mapping) else []
    result = {
        "valid": not errors,
        "output": str(output),
        "formal_publications": len(value.get("formal_publications", [])) if isinstance(value, Mapping) else 0,
        "preprints_only": len(value.get("preprints_only", [])) if isinstance(value, Mapping) else 0,
        "requests": len(value.get("provenance", {}).get("requests", [])) if isinstance(value, Mapping) else 0,
        "api_failures": len(failures),
        "validation_warnings": len(warnings),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors or (args.strict_network and (failures or warnings)):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
