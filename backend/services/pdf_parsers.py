from __future__ import annotations

import os
from typing import Any

import httpx
import xml.etree.ElementTree as ET


async def pdf2grobid(
    filename: str,
    grobid_url: str | None = None,
) -> str:
    grobid_url = (grobid_url or os.environ.get("GROBID_URL") or "").strip() or (
        "https://kermitt2-grobid.hf.space/api/processFulltextDocument"
    )
    timeout = httpx.Timeout(60.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(filename, "rb") as file:
            files = {"input": file}
            response = await client.post(grobid_url, files=files)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "xml" not in content_type and "text/plain" not in content_type:
        raise RuntimeError(
            f"GROBID returned unexpected content-type '{content_type}' "
            f"(HTTP {response.status_code}). "
            "The service may be unavailable or rate-limiting. "
            f"Response preview: {response.text[:200]!r}"
        )
    return response.text


async def pdf2dpt(
    filename: str,
    dpt_url: str | None = None,
) -> dict[str, Any]:
    api_key = (os.environ.get("DPT_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing DPT_API_KEY")
    dpt_url = (dpt_url or os.environ.get("DPT_URL") or "").strip() or (
        "https://api.va.eu-west-1.landing.ai/v1/ade/parse"
    )
    headers = {"Authorization": api_key}
    data = {"model": "dpt-2-latest"}
    timeout = httpx.Timeout(60.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(filename, "rb") as document:
            files = {"document": document}
            response = await client.post(
                dpt_url, headers=headers, data=data, files=files
            )
    response.raise_for_status()
    return response.json()


def extract_body_text(xml_content: str) -> str:
    namespace = {"tei": "http://www.tei-c.org/ns/1.0"}
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise RuntimeError(f"GROBID response is not valid XML: {exc}") from exc
    body = root.find(".//tei:body", namespace)
    if body is None:
        raise RuntimeError(
            "GROBID response parsed as XML but contains no <tei:body> element. "
            "The service may have returned an error document."
        )
    text = "".join(body.itertext()).strip()
    if not text:
        raise RuntimeError("GROBID returned an empty <tei:body>.")
    return text
