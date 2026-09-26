"""Privacy guard plugin — on-device PII redaction (outbound) & restore (inbound).

Bidirectional PII protection for Hermes Agent:

  OUTBOUND (tool results → model):
    transform_tool_result redacts PII into stable placeholders (EMAIL#a1b2c3),
    storing a placeholder→original mapping in a persistent local store.

  INBOUND (model reply → user):
    transform_llm_output restores placeholders back to the original values
    before the reply is shown locally, so PII always displays correctly on
    this machine while only redacted copies leave the device.

Layers:
1. Fast rule-based regex for structured PII (email, phone, IBAN, credit cards,
   IPs, Chinese IDs)
2. spaCy Chinese NER for unstructured Chinese PII (names, addresses)
3. Optional Ai4Privacy transformer model for multilingual PII

Placeholders are consistent (same input → same placeholder) so AI tools keep
working across screens and turns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── PII Category Mapping ───────────────────────────────────────────────────

PII_PLACEHOLDERS = {
    "EMAIL": "EMAIL#{}",
    "PHONE": "PHONE#{}",
    "CREDIT_CARD": "CC#{}",
    "IBAN": "IBAN#{}",
    "NAME": "NAME#{}",
    "ADDRESS": "ADDR#{}",
    "IP_ADDRESS": "IP#{}",
    "SSN": "SSN#{}",
}

# Placeholder regex: TYPE#hexhash — used for restore detection and NER skip.
# Deliberately NO leading lookbehind: a placeholder can legitimately sit flush
# against an ASCII label (e.g. "IP10.0.0.1" -> "IP" + "IP#f50473" ->
# "IPIP#f50473"), and a leading (?<![A-Za-z0-9]) would silently fail to see
# such a placeholder — breaking both idempotency (NER would re-tag it) and
# restore (the placeholder would never be extracted back to its original).
# The TYPE keywords plus the mandatory "#hex6" tail are specific enough that
# accidental matches in natural text are negligible; restore only substitutes
# placeholders that exist in the mapping, so a stray match cannot inject data.
_PLACEHOLDER_RE = re.compile(
    r"(?:EMAIL|PHONE|CC|IBAN|NAME|ADDR|IP|SSN|PII)#[a-f0-9]{6}(?![A-Za-z0-9])"
)

# Conservative skip test for NER entities: a '#' inside the span means it is
# (or contains) a placeholder, so never touch it. The full-regex check alone is
# NOT enough — spaCy spans may include adjacent CJK chars, so a loose fallback
# scan is kept.
def _is_placeholder(text: str) -> bool:
    if "#" not in text:
        return False
    if _PLACEHOLDER_RE.search(text):
        return True
    return bool(re.search(r"[A-Z]{2,5}#[a-f0-9]{4,8}", text))

# ─── Regex Patterns ─────────────────────────────────────────────────────────
# Order matters: apply more specific/longer patterns FIRST to avoid partial matches

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

_CREDIT_CARD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})(?![A-Za-z0-9])"
)

_IBAN_RE = re.compile(
    r"[A-Z]{2}\d{2}[A-Z0-9]{18,30}"
)

_CHINA_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![A-Za-z0-9])"
)

_IPV4_RE = re.compile(
    r"(?<![0-9.#])(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![0-9.])"
)

# Phone patterns - must be applied AFTER other numeric PII to avoid false positives
_PHONE_RE = re.compile(
    # Chinese mobile: optional +86 prefix, 11 digits starting with 1[3-9]
    # lookbehind prevents matching a suffix of a longer digit string (e.g. China ID)
    r"(?<![\d.-])(?:\+?86[-\s.]?)?1[3-9][-\s.]?\d[-\s.]?\d{4}[-\s.]?\d{4}(?![\d.-])"
    # US/CA: 11 digits with optional +1 prefix
    r"|(?<![\d.-])\+?1[-\s.]?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}(?![\d.-])"
    # Generic international: +country_code followed by 4-14 digits
    r"|(?<![\d.-])\+\d{1,3}[-\s.]?\d{4,14}(?![\d.-])"
)


def _hash_short(value: str, length: int = 6) -> str:
    """Create a short stable hash for placeholder consistency."""
    h = hashlib.sha256(value.encode()).hexdigest()[:length]
    return h


def _store_path() -> Path:
    """Persistent store for the placeholder ↔ original mapping.

    Survives plugin reloads / session restarts so placeholders sent to the
    model in an earlier turn can be restored in a later reply.
    """
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir("privacy-guard") / "mapping.json"
    except Exception:
        # Fallback: keep in plugin dir if plugin_storage is unavailable
        return Path(__file__).parent / ".privacy-guard-mapping.json"


class PrivacyGuard:
    """Main PII detection, redaction, and restore engine."""

    def __init__(self, use_model: bool = True):
        self._model = None
        self._use_model = use_model
        self._model_lock = threading.Lock()

        # spaCy Chinese NER
        self._chinese_nlp = None
        self._nlp_lock = threading.Lock()

        # Bidirectional mapping: placeholder -> (pii_type, original_value)
        self._ph_to_value: Dict[str, Tuple[str, str]] = {}
        # (pii_type, value) -> placeholder (reverse lookup for consistency)
        self._value_to_ph: Dict[Tuple[str, str], str] = {}
        self._map_lock = threading.Lock()

        # Persistent store
        self._store_file = _store_path()
        self._load_mapping()

    # ── Mapping persistence ─────────────────────────────────────────────

    def _load_mapping(self) -> None:
        try:
            if self._store_file.exists():
                data = json.loads(self._store_file.read_text(encoding="utf-8"))
                for ph, (ptype, value) in data.items():
                    self._ph_to_value[ph] = (ptype, value)
                    self._value_to_ph[(ptype, value)] = ph
                logger.info("Privacy guard: loaded %d mapping entries", len(self._ph_to_value))
        except Exception as e:
            logger.warning(f"Privacy guard: failed to load mapping: {e}")

    def _save_mapping(self) -> None:
        try:
            self._store_file.parent.mkdir(parents=True, exist_ok=True)
            with self._map_lock:
                data = {ph: [ptype, value] for ph, (ptype, value) in self._ph_to_value.items()}
            self._store_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Privacy guard: failed to save mapping: {e}")

    # ── Placeholder management ──────────────────────────────────────────

    def _get_placeholder(self, pii_type: str, value: str) -> str:
        """Get or create a consistent placeholder, persisting both directions."""
        key = (pii_type, value)
        with self._map_lock:
            if key in self._value_to_ph:
                return self._value_to_ph[key]
            short_hash = _hash_short(value)
            placeholder = PII_PLACEHOLDERS.get(pii_type, f"PII#{short_hash}").format(short_hash)
            # Guard against accidental collision (different value, same type+hash):
            # extremely unlikely with sha256-6, but keep mapping consistent anyway.
            self._ph_to_value[placeholder] = (pii_type, value)
            self._value_to_ph[key] = placeholder
        self._save_mapping()
        return placeholder

    def restore(self, text: str) -> str:
        """Restore original PII from placeholders in model output.

        Only known placeholders (sent outbound earlier) are replaced —
        unknown ones stay as-is, so a hallucinated placeholder never
        injects anything.
        """
        if not text or not isinstance(text, str):
            return text
        if not _PLACEHOLDER_RE.search(text):
            return text

        with self._map_lock:
            snapshot = dict(self._ph_to_value)

        restored = text
        # Longest placeholder first to avoid prefix collisions
        for ph in sorted(snapshot, key=len, reverse=True):
            if ph in restored:
                restored = restored.replace(ph, snapshot[ph][1])
        return restored

    # ── Redaction layers ────────────────────────────────────────────────

    def _redact_regex(self, text: str) -> str:
        """Apply regex-based redaction for structured PII."""

        def repl(ptype):
            return lambda m: self._get_placeholder(ptype, m.group(0))

        # Apply in order: most specific first
        text = _EMAIL_RE.sub(repl("EMAIL"), text)
        text = _CREDIT_CARD_RE.sub(repl("CREDIT_CARD"), text)
        text = _IBAN_RE.sub(repl("IBAN"), text)
        text = _CHINA_ID_RE.sub(repl("SSN"), text)
        text = _IPV4_RE.sub(repl("IP_ADDRESS"), text)
        text = _PHONE_RE.sub(repl("PHONE"), text)
        return text

    def _find_chinese_model(self, spacy) -> Optional[str]:
        """Locate a loadable zh_core_web_sm data directory.

        Handles both the pip-installed layout (resolved by name) and manual
        copies where the data lives in a version subdirectory.
        """
        candidates = []
        spacy_dir = Path(spacy.__file__).parent
        # spacy.__file__ is site-packages/spacy/__init__.py — the model package
        # sits next to spacy/, so check site-packages first, then inside spacy/.
        for base in (spacy_dir.parent, spacy_dir):
            pkg_dir = base / "zh_core_web_sm"
            if not pkg_dir.is_dir():
                continue
            for cand in sorted(pkg_dir.glob("zh_core_web_sm-*")):
                if (cand / "config.cfg").exists() or (cand / "config.json").exists():
                    candidates.append(str(cand))
            if (pkg_dir / "config.cfg").exists() or (pkg_dir / "config.json").exists():
                candidates.append(str(pkg_dir))
        return candidates[0] if candidates else None

    def _get_chinese_nlp(self):
        if self._chinese_nlp is not None:
            return self._chinese_nlp
        with self._nlp_lock:
            if self._chinese_nlp is None:
                import warnings
                with warnings.catch_warnings():
                    # Suppress spacy's model-version-compat warning (3.7 model on 3.8 spacy)
                    warnings.simplefilter("ignore")
                    try:
                        import spacy
                        paths = [
                            "zh_core_web_sm",  # pip-installed, resolved by name
                            self._find_chinese_model(spacy) or "",
                        ]
                        for path in paths:
                            if not path:
                                continue
                            try:
                                self._chinese_nlp = spacy.load(path)
                                logger.info(f"Privacy guard: Chinese NER model loaded from {path}")
                                break
                            except Exception as e:
                                logger.debug(f"Privacy guard: could not load {path}: {e}")
                        if self._chinese_nlp is None:
                            logger.warning(
                                "Privacy guard: Chinese NER model not found; "
                                "install with `pip install zh-core-web-sm`"
                            )
                    except Exception as e:
                        logger.warning(f"Privacy guard: Chinese NER not available: {e}")
            return self._chinese_nlp

    def _redact_chinese_ner(self, text: str) -> str:
        nlp = self._get_chinese_nlp()
        if nlp is None:
            return text
        # Never run NER on text that already contains placeholders. spaCy splits
        # placeholders into small tokens ("IP#2a39f1" -> "IP" + "2a39") and
        # re-tags them as PII, and the boundary Chinese segments it creates can
        # be double-annotated (corrupting the text). First pass has no
        # placeholders so NER runs normally there; a second pass is a no-op,
        # which makes redact() idempotent.
        if _PLACEHOLDER_RE.search(text):
            return text
        return self._ner_on_segment(nlp, text)

    def _ner_on_segment(self, nlp, segment: str) -> str:
        """Run NER redaction on a single text segment.

        Invariant: output is never shorter than input. A placeholder is always
        longer than a single CJK char, and we skip spans under 2 chars, so a
        length shrink signals silent corruption — fail safe to the original.
        """
        if not segment:
            return segment
        doc = nlp(segment)
        replacements = []
        for ent in doc.ents:
            if _is_placeholder(ent.text):
                continue
            if len(ent.text) < 2:
                continue
            if ent.label_ == "PERSON":
                replacements.append(("NAME", ent.text, ent.start_char, ent.end_char))
            elif ent.label_ in ("GPE", "FAC", "ORG"):
                if any(c in ent.text for c in ["路", "街", "区", "省", "市", "县", "镇", "村", "号"]):
                    replacements.append(("ADDRESS", ent.text, ent.start_char, ent.end_char))

        if not replacements:
            return segment
        out = segment
        for pii_type, value, start, end in reversed(replacements):
            if not (0 <= start <= end <= len(out)):
                continue  # out-of-range span: skip rather than corrupt output
            placeholder = self._get_placeholder(pii_type, value)
            out = out[:start] + placeholder + out[end:]
        if len(out) < len(segment):
            logger.warning("Privacy guard: NER replacement shrank text; restoring original")
            return segment
        return out

    def _redact_model(self, text: str) -> str:
        if not self._use_model or self._model is None:
            return text
        try:
            result = self._model(text, multilingual=True, classify_pii=True)
            if isinstance(result, dict):
                return result.get('text', text)
            elif isinstance(result, str):
                return result
            return text
        except Exception as e:
            logger.debug(f"Privacy guard: model redaction failed: {e}")
            return text

    def _load_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is None:
                try:
                    from ai4privacy import protect
                    self._model = protect
                    logger.info("Privacy guard: Ai4Privacy model loaded")
                except ImportError:
                    logger.warning("Privacy guard: ai4privacy not available, using regex + NER only")
                    self._model = None

    def redact(self, text: str) -> str:
        """Redact PII into placeholders (outbound path)."""
        if not text or not isinstance(text, str):
            return text

        # Quick skip for non-PII text
        has_ascii_pii = any(p in text for p in ['@', '+', '4', '5', '3', '6', 'D', 'E', 'F', 'G'])
        has_chinese_pii_markers = any(c in text for c in ['路', '街', '区', '省', '市', '张', '李', '王', '赵'])
        if not has_ascii_pii and not has_chinese_pii_markers:
            if not _CHINA_ID_RE.search(text):
                return text

        # NER first (unstructured Chinese), then regex (structured), then ML
        text = self._redact_chinese_ner(text)
        text = self._redact_regex(text)
        if self._use_model:
            self._load_model()
            text = self._redact_model(text)
        return text


# Global singleton instance
_guard: Optional[PrivacyGuard] = None
_guard_lock = threading.Lock()


def get_guard() -> PrivacyGuard:
    """Get or create the global privacy guard instance."""
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = PrivacyGuard(use_model=True)
        return _guard


# ─── Plugin Hooks ───────────────────────────────────────────────────────────

def _on_transform_tool_result(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """Outbound: redact PII from tool results before they reach the model."""
    if not isinstance(result, str) or not result:
        return None
    redacted = get_guard().redact(result)
    if redacted != result:
        logger.debug(f"Privacy guard: redacted PII in {tool_name} result")
    return redacted


def _on_transform_llm_output(
    response_text: str = "",
    session_id: str = "",
    model: str = "",
    **_: Any,
) -> Optional[str]:
    """Inbound: restore original PII from placeholders in the model's reply."""
    if not response_text:
        return None
    restored = get_guard().restore(response_text)
    if restored != response_text:
        logger.debug(f"Privacy guard: restored PII in model reply (session {session_id})")
    return restored


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    **_: Any,
) -> None:
    """Observer: log PII detection statistics."""
    if isinstance(result, str) and result:
        redacted = get_guard().redact(result)
        if redacted != result:
            logger.info(f"Privacy guard: detected PII in {tool_name} output")


def register(ctx) -> None:
    """Register privacy guard hooks with Hermes."""
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    logger.info("Privacy guard plugin registered (bidirectional)")