from __future__ import annotations

from collections import Counter
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from .schema import FeatureSchema, build_feature_schema
from .url_normalizer import NormalizedURL, normalize_url


class URLFeatureExtractor:
    def __init__(
        self,
        features_config: Any,
        feature_weights: Optional[Dict[str, float]] = None,
        enabled_groups: Optional[Set[str]] = None,
        reference_root: Optional[Path] = None,
    ):
        self.features_config = features_config
        self.schema: FeatureSchema = build_feature_schema(features_config)
        self.features_list = self.schema.feature_names
        self.feature_groups_map = features_config.get_feature_groups_map()
        self.feature_groups_config = features_config.feature_groups
        self.enabled_groups = enabled_groups or set(self.feature_groups_map.keys())
        self.feature_weights = feature_weights or {}

        if reference_root is None:
            from core import settings

            reference_root = Path(settings.dataset_path).parent / "feature" / "raw"
        self._feature_raw_path = reference_root

        self._brand_keywords = self._load_feature_values(
            "brand_keyword",
            [
                "paypal",
                "amazon",
                "google",
                "facebook",
                "microsoft",
                "apple",
                "bank",
                "secure",
                "login",
                "account",
                "verify",
                "update",
            ],
        )
        self._suspicious_tlds = self._load_feature_values(
            "suspicious_tld",
            ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link"],
            transform=lambda value: value.lstrip("*.").lower(),
        )
        self._free_hosts = self._load_feature_values(
            "free_hosting",
            ["000webhost", "freenom", "freehosting", "byethost", "awardspace", "x10hosting"],
        )
        self._non_standard_ports = self._load_feature_values(
            "non_standard_port",
            ["4444", "1337", "8080", "8888", "3000", "5000", "7000", "9000"],
        )
        self._url_shorteners = self._load_feature_values(
            "sorted_url",
            ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly"],
        )
        self._auto_download_params = self._load_feature_values(
            "auto_download_params",
            ["download=", "file=", "get=", "attachment="],
        )

    def extract(self, url: object, apply_weights: bool = False, target_class: Optional[str] = None) -> Dict[str, float]:
        normalized = normalize_url(url)
        parsed = urlsplit(normalized.normalized_url or "http://invalid.local")
        lexical_source = normalized.normalized_url or ("" if url is None else str(url).strip())

        extracted: Dict[str, float] = {}
        extracted.update(self._extract_url_level(lexical_source))
        extracted.update(self._extract_domain_level(parsed, normalized))
        extracted.update(self._extract_path_level(parsed))
        extracted.update(self._extract_query_level(parsed))
        extracted.update(self._extract_sequence_patterns(lexical_source))

        if self.enabled_groups and "all_features" not in self.enabled_groups:
            allowed = set()
            for group_name in self.enabled_groups:
                allowed.update(self.feature_groups_map.get(group_name, []))
            extracted = {name: value for name, value in extracted.items() if name in allowed}

        aligned = self.schema.align_record(extracted)

        if apply_weights and self.feature_weights:
            aligned = {
                name: value * self.feature_weights.get(name, 1.0)
                for name, value in aligned.items()
            }

        return aligned

    def get_enabled_features(self) -> Iterable[str]:
        if "all_features" in self.enabled_groups:
            return self.schema.feature_names
        features = []
        for group in self.enabled_groups:
            features.extend(self.feature_groups_map.get(group, []))
        return features

    def enable_feature_groups(self, groups):
        self.enabled_groups = set(groups)

    def disable_feature_groups(self, groups):
        for group in groups:
            self.enabled_groups.discard(group)

    def set_feature_weights(self, weights: Dict[str, float]):
        self.feature_weights = weights

    def _load_feature_values(self, feature_name: str, default: list, transform=None) -> set:
        csv_path = self._feature_raw_path / f"{feature_name}.csv"
        if csv_path.exists():
            try:
                frame = pd.read_csv(csv_path)
                if "value" in frame.columns:
                    values = frame["value"].dropna().astype(str).str.strip().tolist()
                    if transform:
                        values = [transform(value) for value in values]
                    return set(value.lower() for value in values if value)
            except Exception:
                pass
        return set((transform(value) if transform else value).lower() for value in default)

    def _extract_url_level(self, url: str) -> Dict[str, float]:
        features: Dict[str, float] = {}
        length = len(url)
        digits = sum(char.isdigit() for char in url)
        letters = sum(char.isalpha() for char in url)
        special_chars = length - digits - letters

        features["http_token"] = 1.0 if url.startswith("http://") else 0.0
        features["url_length"] = float(length)
        features["dot_count"] = float(url.count("."))
        features["hyphen_count"] = float(url.count("-"))
        features["slash_count"] = float(url.count("/"))
        features["at_symbol_count"] = float(url.count("@"))
        features["percent_count"] = float(url.count("%"))
        features["special_char_count"] = float(special_chars)
        features["digit_ratio"] = digits / length if length else 0.0
        features["letter_ratio"] = letters / length if length else 0.0
        features["special_char_ratio"] = special_chars / length if length else 0.0
        features["longest_digit_sequence"] = float(self._longest_sequence_extraction(url, str.isdigit))
        features["longest_letter_sequence"] = float(self._longest_sequence_extraction(url, str.isalpha))
        features["longest_special_sequence"] = float(
            self._longest_sequence_extraction(url, lambda char: not char.isalnum())
        )
        features["character_continuity_rate"] = self._calculate_continuity(url)
        features["url_entropy"] = self._calculate_entropy(url)
        features["low_entropy"] = 1.0 if features["url_entropy"] < 3.0 else 0.0
        features["high_entropy"] = 1.0 if features["url_entropy"] > 4.0 else 0.0
        features["long_url_length"] = 1.0 if features["url_length"] > 100 else 0.0
        features["high_digit_ratio"] = 1.0 if features["digit_ratio"] > 0.3 else 0.0
        features["low_special_char_ratio"] = 1.0 if features["special_char_ratio"] < 0.05 else 0.0
        return features

    def _extract_domain_level(self, parsed, normalized: NormalizedURL) -> Dict[str, float]:
        features: Dict[str, float] = {}
        domain = normalized.hostname or ""
        domain_parts = [part for part in domain.split(".") if part]
        token_lengths = [len(part) for part in domain_parts]
        tld = domain_parts[-1] if domain_parts else ""
        explicit_port = normalized.had_explicit_port
        risky_port = explicit_port and (
            str(normalized.port) in self._non_standard_ports
            or (parsed.scheme == "http" and normalized.port != 80)
            or (parsed.scheme == "https" and normalized.port != 443)
        )

        features["domain_length"] = float(len(domain))
        features["domain_token_count"] = float(len(domain_parts))
        features["subdomain_count"] = float(max(0, len(domain_parts) - 2))
        digits = sum(char.isdigit() for char in domain)
        features["digit_ratio_domain"] = digits / len(domain) if domain else 0.0
        features["hyphen_count_domain"] = float(domain.count("-"))
        features["longest_domain_token"] = float(max(token_lengths) if token_lengths else 0)
        features["avg_domain_token_length"] = float(sum(token_lengths) / len(token_lengths) if token_lengths else 0)
        features["domain_entropy"] = self._calculate_entropy(domain)
        features["ip_address_flag"] = 1.0 if self._is_ip_address(domain) else 0.0
        features["multiple_subdomain_flag"] = 1.0 if features["subdomain_count"] > 1 else 0.0
        features["brand_keyword_flag"] = 1.0 if self._has_brand_keywords(domain) else 0.0
        features["port_in_url_flag"] = 1.0 if explicit_port else 0.0
        features["https_in_domain"] = 1.0 if "https" in domain else 0.0
        features["tld_suspicious_flag"] = 1.0 if self._has_suspicious_tld(tld) else 0.0
        features["shortening_service_flag"] = 1.0 if self._is_url_shortener(domain) else 0.0
        features["double_slash_redirecting"] = 1.0 if "//" in parsed.path else 0.0
        features["punycode_domain_flag"] = 1.0 if any(part.startswith("xn--") for part in domain_parts) else 0.0
        features["unicode_domain_flag"] = 1.0 if any(ord(char) > 127 for char in normalized.raw_url) else 0.0
        features["homograph_suspicious_flag"] = 1.0 if self._has_homograph_chars(normalized.raw_url) else 0.0
        features["excessive_subdomain_depth"] = float(max(0, int(features["subdomain_count"]) - 4))
        features["random_string_domain_flag"] = 1.0 if self._is_random_domain(domain) else 0.0
        features["free_hosting_domain_flag"] = 1.0 if self._is_free_hosting(domain) else 0.0
        features["dga_domain_flag"] = 1.0 if self._is_dga_domain(domain) else 0.0
        features["non_standard_port_flag"] = 1.0 if risky_port else 0.0
        features["high_domain_entropy"] = 1.0 if features["domain_entropy"] > 3.8 else 0.0
        return features

    def _extract_path_level(self, parsed) -> Dict[str, float]:
        features: Dict[str, float] = {}
        path = parsed.path or ""
        path_tokens = [token for token in path.split("/") if token]
        token_lengths = [len(token) for token in path_tokens]
        filename = path_tokens[-1] if path_tokens else ""
        lower_filename = filename.lower()
        suspicious_extensions = [".exe", ".bat", ".cmd", ".scr", ".vbs", ".js", ".jar", ".apk", ".sh"]

        features["path_length"] = float(len(path))
        features["path_token_count"] = float(len(path_tokens))
        digits = sum(char.isdigit() for char in path)
        features["digit_ratio_path"] = digits / len(path) if path else 0.0
        features["dot_count_path"] = float(path.count("."))
        features["longest_path_token"] = float(max(token_lengths) if token_lengths else 0)
        features["avg_path_token_length"] = float(sum(token_lengths) / len(token_lengths) if token_lengths else 0)
        features["path_entropy"] = self._calculate_entropy(path)
        features["filename_length"] = float(len(filename))
        features["suspicious_extension_flag"] = 1.0 if any(lower_filename.endswith(ext) for ext in suspicious_extensions) else 0.0
        features["executable_extension_flag"] = 1.0 if any(
            lower_filename.endswith(ext) for ext in [".exe", ".bat", ".cmd", ".scr"]
        ) else 0.0
        features["suspicious_js_extension_flag"] = 1.0 if (
            lower_filename.endswith(".js") or "javascript:" in parsed.geturl().lower()
        ) else 0.0
        return features

    def _extract_query_level(self, parsed) -> Dict[str, float]:
        features: Dict[str, float] = {}
        query = parsed.query or ""
        params = parse_qs(query, keep_blank_values=True)
        lower_query = query.lower()
        param_lengths = [len(key) + len("".join(values)) for key, values in params.items()]

        features["query_length"] = float(len(query))
        features["parameter_count"] = float(len(params))
        digits = sum(char.isdigit() for char in query)
        features["digit_ratio_query"] = digits / len(query) if query else 0.0
        features["equal_count_query"] = float(query.count("="))
        features["ampersand_count_query"] = float(query.count("&"))
        features["avg_parameter_length"] = float(sum(param_lengths) / len(param_lengths) if param_lengths else 0)
        features["max_parameter_length"] = float(max(param_lengths) if param_lengths else 0)
        features["query_entropy"] = self._calculate_entropy(query)
        features["encoded_url_flag"] = 1.0 if "%" in query else 0.0
        features["redirect_parameter_flag"] = 1.0 if any(
            token in lower_query
            for token in [
                "url=",
                "redirect=",
                "next=",
                "return=",
                "goto=",
                "redirect_url=",
                "return_url=",
                "next_url=",
                "goto_url=",
                "dest=",
                "destination=",
                "target=",
                "rurl=",
                "ru=",
                "back=",
                "callback=",
                "continue=",
                "link=",
                "path=",
                "ref=",
                "referrer=",
                "site=",
                "to=",
                "uri=",
                "u=",
                "redirecturl=",
            ]
        ) else 0.0
        full_url = parsed.geturl().lower()
        features["script_in_url_flag"] = 1.0 if "script" in full_url else 0.0
        features["auto_download_param_flag"] = 1.0 if any(
            token in lower_query for token in self._auto_download_params
        ) else 0.0
        features["base64_in_url_flag"] = 1.0 if self._has_base64_encoding(full_url) else 0.0
        return features

    def _extract_sequence_patterns(self, url: str) -> Dict[str, float]:
        return {
            "mixed_token_flag": 1.0 if re.search(r"[a-zA-Z]+\d+|[0-9]+[a-zA-Z]+", url) else 0.0,
            "hex_encoding_flag": 1.0 if re.search(r"%[0-9a-fA-F]{2}", url) else 0.0,
            "obfuscation_pattern_flag": 1.0 if re.search(r"[a-zA-Z0-9]{20,}", url) else 0.0,
        }

    def _longest_sequence_extraction(self, text: str, condition) -> int:
        max_len = 0
        current = 0
        for char in text:
            if condition(char):
                current += 1
                max_len = max(max_len, current)
            else:
                current = 0
        return max_len

    def _calculate_continuity(self, text: str) -> float:
        if not text:
            return 0.0
        longest_alpha = self._longest_sequence_extraction(text, str.isalpha)
        longest_digit = self._longest_sequence_extraction(text, str.isdigit)
        longest_special = self._longest_sequence_extraction(text, lambda char: not char.isalnum())
        return (longest_alpha + longest_digit + longest_special) / len(text)

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        entropy = 0.0
        for count in counts.values():
            probability = count / len(text)
            entropy -= probability * math.log2(probability)
        return float(entropy)

    def _is_ip_address(self, domain: str) -> bool:
        if not domain:
            return False
        parts = domain.split(".")
        if len(parts) != 4:
            return False
        for part in parts:
            if not part.isdigit():
                return False
            value = int(part)
            if value < 0 or value > 255:
                return False
        return True

    def _has_brand_keywords(self, domain: str) -> bool:
        return any(keyword in domain for keyword in self._brand_keywords)

    def _has_suspicious_tld(self, tld: str) -> bool:
        return tld.lower() in self._suspicious_tlds

    def _is_url_shortener(self, domain: str) -> bool:
        return any(shortener in domain for shortener in self._url_shorteners)

    def _has_homograph_chars(self, text: str) -> bool:
        cyrillic_chars = set("авсԁеһіјӏорԛѕԝхуᴢ")
        greek_chars = set("αβγδεζηθικλμνξοπρστυφχψω")
        return any(char.lower() in cyrillic_chars or char.lower() in greek_chars for char in text)

    def _is_random_domain(self, domain: str) -> bool:
        domain_name = domain.split(".")[0] if "." in domain else domain
        return len(domain_name) >= 8 and self._calculate_entropy(domain_name) > 3.5

    def _is_free_hosting(self, domain: str) -> bool:
        return any(host in domain for host in self._free_hosts)

    def _is_dga_domain(self, domain: str) -> bool:
        domain_name = domain.split(".")[0] if "." in domain else domain
        if len(domain_name) < 10:
            return False
        vowels = sum(1 for char in domain_name if char in "aeiou")
        consonants = sum(1 for char in domain_name if char.isalpha() and char not in "aeiou")
        if not consonants:
            return False
        ratio = vowels / (vowels + consonants)
        return self._calculate_entropy(domain_name) > 3.8 and (ratio < 0.2 or ratio > 0.6)

    def _has_base64_encoding(self, url: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9+/]{20,}={0,2}", url))
