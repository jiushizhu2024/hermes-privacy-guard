"""Privacy guard plugin — PII detection and redaction for all Hermes tool outputs.

Uses a hybrid approach:
1. Fast rule-based regex for structured PII (email, phone, IBAN, credit cards, IPs, Chinese IDs)
2. spaCy Chinese NER for unstructured Chinese PII (names, addresses)
3. Optional Ai4Privacy transformer model for multilingual PII

Redacted values use consistent placeholders so AI tools keep working:
- EMAIL → EMAIL#<hash>
- PHONE → PHONE#<hash>
- CREDIT_CARD → CC#<hash>
- IBAN → IBAN#<hash>
- NAME → NAME#<hash>
- ADDRESS → ADDR#<hash>
- IP_ADDRESS → IP#<hash>
- SSN → SSN#<hash>
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any, Dict, Optional

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

# ─── Regex Patterns ─────────────────────────────────────────────────────────
# Order matters: apply more specific/longer patterns FIRST to avoid partial matches

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

_CREDIT_CARD_RE = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
)

_IBAN_RE = re.compile(
    r"[A-Z]{2}\d{2}[A-Z0-9]{18,30}"
)

_CHINA_ID_RE = re.compile(
    r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"
)

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# Phone patterns - must be applied AFTER other numeric PII to avoid false positives
_PHONE_RE = re.compile(
    # Chinese mobile: optional +86 prefix, 11 digits starting with 1[3-9]
    r"(?:\+?86[-\s.]?)?1[3-9][-\s.]?\d[-\s.]?\d{4}[-\s.]?\d{4}"
    # US/CA: 11 digits with optional +1 prefix
    r"|(?<![\d.-])\+?1[-\s.]?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}(?![\d.-])"
    # Generic international: +country_code followed by 4-14 digits
    r"|(?<![\d.-])\+\d{1,3}[-\s.]?\d{4,14}(?![\d.-])"
)

def _hash_short(value: str, length: int = 6) -> str:
    """Create a short stable hash for placeholder consistency."""
    h = hashlib.sha256(value.encode()).hexdigest()[:length]
    return h


class PrivacyGuard:
    """Main PII detection and redaction engine."""
    
    def __init__(self, use_model: bool = True):
        self._model = None
        self._use_model = use_model
        self._model_lock = threading.Lock()
        
        # spaCy Chinese NER
        self._chinese_nlp = None
        self._nlp_lock = threading.Lock()
        
        # Local cache for observed PII values → consistent placeholders
        self._pii_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
    
    def _get_placeholder(self, pii_type: str, value: str) -> str:
        """Get or create a consistent placeholder for a PII value."""
        key = f"{pii_type}:{value}"
        with self._cache_lock:
            if key not in self._pii_cache:
                short_hash = _hash_short(value)
                placeholder = PII_PLACEHOLDERS.get(pii_type, f"PII#{short_hash}")
                self._pii_cache[key] = placeholder.format(short_hash)
            return self._pii_cache[key]
    
    def _redact_regex(self, text: str) -> str:
        """Apply regex-based redaction for structured PII.
        
        Order matters: apply longer/more specific patterns first to avoid
        partial matches (e.g., Chinese ID shouldn't be matched as phone).
        """
        
        def replace_email(m):
            return self._get_placeholder("EMAIL", m.group(0))
        
        def replace_cc(m):
            return self._get_placeholder("CREDIT_CARD", m.group(0))
        
        def replace_iban(m):
            return self._get_placeholder("IBAN", m.group(0))
        
        def replace_china_id(m):
            return self._get_placeholder("SSN", m.group(0))
        
        def replace_ip(m):
            return self._get_placeholder("IP_ADDRESS", m.group(0))
        
        def replace_phone(m):
            return self._get_placeholder("PHONE", m.group(0))
        
        # Apply in order: most specific first
        text = _EMAIL_RE.sub(replace_email, text)
        text = _CREDIT_CARD_RE.sub(replace_cc, text)
        text = _IBAN_RE.sub(replace_iban, text)
        text = _CHINA_ID_RE.sub(replace_china_id, text)
        text = _IPV4_RE.sub(replace_ip, text)
        text = _PHONE_RE.sub(replace_phone, text)
        
        return text
    
    def _get_chinese_nlp(self):
        """Lazy-load Chinese NER model."""
        if self._chinese_nlp is not None:
            return self._chinese_nlp
        
        with self._nlp_lock:
            if self._chinese_nlp is not None:
                return self._chinese_nlp
            
            try:
                import spacy
                self._chinese_nlp = spacy.load("zh_core_web_sm")
                logger.info("Privacy guard: Chinese NER model loaded")
            except Exception as e:
                logger.warning(f"Privacy guard: Chinese NER not available: {e}")
                self._chinese_nlp = None
            
            return self._chinese_nlp
    
    def _redact_chinese_ner(self, text: str) -> str:
        """Use Chinese NER to redact names and addresses."""
        nlp = self._get_chinese_nlp()
        if nlp is None:
            return text
        
        try:
            doc = nlp(text)
            
            # Collect entities to redact with their positions
            replacements = []
            for ent in doc.ents:
                # Skip if already redacted (contains placeholder pattern)
                if "NAME#" in ent.text or "ADDR#" in ent.text or "EMAIL#" in ent.text:
                    continue
                
                # Skip if looks like structured PII (contains only digits/dashes)
                if all(c.isdigit() or c in "-." for c in ent.text):
                    continue
                
                # Skip if entity contains placeholder hash (e.g., "SSN#1d841b")
                if re.search(r'[A-Z]+#[a-f0-9]{6}', ent.text):
                    continue
                
                # Skip if entity is too short (likely false positive)
                if len(ent.text) < 2:
                    continue
                
                if ent.label_ == "PERSON":
                    replacements.append(("NAME", ent.text, ent.start_char, ent.end_char))
                elif ent.label_ in ("GPE", "FAC", "ORG"):
                    # Check if the entity looks like an address (contains 路/街/区/省/市 etc.)
                    # ORG is included because spaCy sometimes mislabels Chinese addresses as ORG
                    if any(c in ent.text for c in ["路", "街", "区", "省", "市", "县", "镇", "村", "号"]):
                        replacements.append(("ADDRESS", ent.text, ent.start_char, ent.end_char))
            
            # Apply replacements in reverse order to preserve positions
            for pii_type, value, start, end in reversed(replacements):
                placeholder = self._get_placeholder(pii_type, value)
                text = text[:start] + placeholder + text[end:]
            
            return text
        except Exception as e:
            logger.debug(f"Privacy guard: Chinese NER failed: {e}")
            return text
    
    def _load_model(self):
        """Lazy-load Ai4Privacy model."""
        if self._model is not None:
            return
        
        with self._model_lock:
            if self._model is not None:
                return
            
            try:
                from ai4privacy import protect
                self._model = protect
                logger.info("Privacy guard: Ai4Privacy model loaded")
            except ImportError:
                logger.warning("Privacy guard: ai4privacy not available, using regex + NER only")
                self._model = None
    
    def _redact_model(self, text: str) -> str:
        """Use Ai4Privacy model for multilingual PII detection."""
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
    
    def redact(self, text: str) -> str:
        """Apply all redaction layers to text.
        
        Order: NER first (for Chinese names/addresses), then regex (for structured PII),
        then ML model (for edge cases). This avoids NER false positives on already-redacted text.
        """
        if not text or not isinstance(text, str):
            return text
        
        # Quick skip for non-PII text (only if no obvious PII patterns)
        # Note: We still run NER on Chinese text even without ASCII markers
        has_ascii_pii = any(p in text for p in ['@', '+', '4', '5', '3', '6', 'D', 'E', 'F', 'G'])
        has_chinese_pii_markers = any(c in text for c in ['路', '街', '区', '省', '市', '张', '李', '王', '赵'])
        
        if not has_ascii_pii and not has_chinese_pii_markers:
            # Still check for obvious numeric PII patterns
            if not re.search(r'\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b', text):
                if not re.search(r'\b[1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b', text):
                    return text
        
        # First pass: Chinese NER (covers unstructured Chinese names/addresses)
        text = self._redact_chinese_ner(text)
        
        # Second pass: regex-based (fast, covers structured PII)
        text = self._redact_regex(text)
        
        # Third pass: ML model (covers edge cases)
        if self._use_model:
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
    """Redact PII from tool results before they reach the model."""
    if not isinstance(result, str) or not result:
        return None
    
    guard = get_guard()
    redacted = guard.redact(result)
    
    if redacted != result:
        logger.debug(f"Privacy guard: redacted PII in {tool_name} result")
    
    return redacted


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    **_: Any,
) -> None:
    """Log PII detection statistics (observer only)."""
    if isinstance(result, str) and result:
        guard = get_guard()
        redacted = guard.redact(result)
        if redacted != result:
            logger.info(f"Privacy guard: detected PII in {tool_name} output")


def register(ctx) -> None:
    """Register privacy guard hooks with Hermes."""
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    logger.info("Privacy guard plugin registered")
