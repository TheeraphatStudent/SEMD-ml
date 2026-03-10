from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlparse, parse_qs
import re
import math
from collections import Counter
import numpy as np
import sys
from pathlib import Path

import pandas as pd

from core import features_config, settings


class FeatureExtractor:

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None, enabled_groups: Optional[Set[str]] = None):
        self.features_list = features_config.get_all_features()
        self.feature_groups_map = features_config.get_feature_groups_map()
        self.feature_groups_config = features_config.feature_groups
        self.enabled_groups = enabled_groups or set(
            self.feature_groups_map.keys())
        self.use_flat_features = bool(features_config.features)
        self._build_feature_to_group_map()

        self.valid_feature_names = self._build_valid_feature_names()
        self.feature_metadata = self._build_feature_metadata()

        self.feature_weights = self._build_feature_weights()
        if feature_weights:
            self.feature_weights.update(feature_weights)

        self._feature_raw_path = Path(
            settings.dataset_path).parent / 'feature' / 'raw'
        self._brand_keywords = self._load_feature_values('brand_keyword', [
            'paypal', 'amazon', 'google', 'facebook', 'microsoft', 'apple',
            'bank', 'secure', 'login', 'account', 'verify', 'update'
        ])
        self._suspicious_tlds = self._load_feature_values('suspicious_tld', [
            'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click', 'link'
        ], transform=lambda v: v.lstrip('*.').lower())
        self._free_hosts = self._load_feature_values('free_hosting', [
            '000webhost', 'freenom', 'freehosting', 'byethost', 'awardspace', 'x10hosting'
        ])
        self._non_standard_ports = self._load_feature_values('non_standard_port', [
            '4444', '1337', '8080', '8888', '3000', '5000', '7000', '9000'
        ])
        self._url_shorteners = self._load_feature_values('sorted_url', [
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly'
        ])
        self._auto_download_params = self._load_feature_values('auto_download_params', [
            'download=', 'file=', 'get=', 'attachment='
        ])

    def _load_feature_values(self, feature_name: str, default: list, transform=None) -> set:
        csv_path = self._feature_raw_path / f"{feature_name}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'value' in df.columns:
                    values = df['value'].dropna().astype(
                        str).str.strip().tolist()
                    if transform:
                        values = [transform(v) for v in values]
                    return set(v.lower() for v in values if v)
            except Exception:
                pass
        return set(v.lower() if not transform else transform(v).lower() for v in default)

    def _build_feature_to_group_map(self):
        self.feature_to_group = {}
        for group_name, features in self.feature_groups_map.items():
            for feature in features:
                self.feature_to_group[feature] = group_name

    def _build_valid_feature_names(self) -> Set[str]:
        valid_names = set()
        for feature in self.features_list:
            if isinstance(feature, dict) and 'name' in feature:
                valid_names.add(feature['name'])
            elif isinstance(feature, str):
                valid_names.add(feature)
        return valid_names

    def _build_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        metadata = {}
        for group_name, group_data in self.feature_groups_config.items():
            if 'features' in group_data:
                for feature in group_data['features']:
                    if isinstance(feature, dict) and 'name' in feature:
                        metadata[feature['name']] = {
                            'type': feature.get('type', 'numeric'),
                            'description': feature.get('description', ''),
                            'group': group_name
                        }
        return metadata

    def _build_feature_weights(self) -> Dict[str, float]:
        weights = {}
        class_emphasis = features_config.class_feature_emphasis

        for class_name, emphasis_config in class_emphasis.items():
            if isinstance(emphasis_config, dict):
                weight_multiplier = emphasis_config.get('weight', 1.0)
                strong_features = emphasis_config.get('strong_features', [])

                for feature_name in strong_features:
                    if feature_name in weights:
                        weights[feature_name] = max(
                            weights[feature_name], weight_multiplier)
                    else:
                        weights[feature_name] = weight_multiplier

        return weights

    def extract(self, url: str, apply_weights: bool = True, target_class: Optional[str] = None) -> Dict[str, float]:
        original_url = url
        if '://' not in url:
            url = 'http://' + url
        parsed = urlparse(url)
        features = {}

        all_extracted = {}
        all_extracted.update(self._extract_url_level(original_url))
        all_extracted.update(self._extract_domain_level(parsed))
        all_extracted.update(self._extract_path_level(parsed))
        all_extracted.update(self._extract_query_level(parsed))
        # all_extracted.update(self._extract_structural_ratios(url, parsed))
        all_extracted.update(self._extract_sequence_patterns(original_url))

        if self.use_flat_features or 'all_features' in self.enabled_groups:
            features = all_extracted
        else:
            for group_name in self.enabled_groups:
                if group_name in self.feature_groups_map:
                    for feature_name in self.feature_groups_map[group_name]:
                        if feature_name in all_extracted:
                            features[feature_name] = all_extracted[feature_name]

        self._validate_features(features)

        if apply_weights and self.feature_weights:
            features = self._apply_feature_weights(features)

        return features

    def _validate_features(self, features: Dict[str, float]) -> None:
        invalid_features = []
        for feature_name in features.keys():
            if feature_name not in self.valid_feature_names:
                invalid_features.append(feature_name)

        if invalid_features:
            raise ValueError(
                f"Feature miss match in features.yaml:\n"
                f"{', '.join(invalid_features)}\n"
            )

            sys.exit(1)

    def _apply_feature_weights(self, features: Dict[str, float]) -> Dict[str, float]:
        weighted_features = {}
        for feature_name, value in features.items():
            weight = self.feature_weights.get(feature_name, 1.0)
            weighted_features[feature_name] = value * weight
        return weighted_features

    def _apply_class_emphasis(self, features: Dict[str, float], target_class: str) -> Dict[str, float]:
        emphasis_features = features_config.get_class_emphasis_features(
            target_class)
        if not emphasis_features or settings.class_weight_mode != 'soft':
            return features

        emphasized_features = features.copy()
        emphasis_weight = 1.2

        for feature_name in emphasis_features:
            if feature_name in emphasized_features:
                emphasized_features[feature_name] *= emphasis_weight

        return emphasized_features

    def set_feature_weights(self, weights: Dict[str, float]):
        self.feature_weights = weights

    def enable_feature_groups(self, groups: List[str]):
        self.enabled_groups = set(groups)

    def disable_feature_groups(self, groups: List[str]):
        for group in groups:
            self.enabled_groups.discard(group)

    def get_enabled_features(self) -> List[str]:
        enabled_features = []
        for group in self.enabled_groups:
            enabled_features.extend(self.feature_groups_map.get(group, []))
        return enabled_features

    # -------------------------------
    # ---------   Feature  ----------
    # -------------------------------

    def _extract_url_level(self, url: str) -> Dict[str, float]:
        features = {}

        features['http_token'] = 1.0 if url.startswith('http://') else 0.0
        features['url_length'] = len(url)
        features['dot_count'] = url.count('.')
        features['hyphen_count'] = url.count('-')
        features['slash_count'] = url.count('/')
        features['at_symbol_count'] = url.count('@')
        features['percent_count'] = url.count('%')

        digits = sum(c.isdigit() for c in url)
        letters = sum(c.isalpha() for c in url)
        special_chars = len(url) - digits - letters

        features['special_char_count'] = special_chars

        features['digit_ratio'] = digits / len(url) if len(url) > 0 else 0
        features['letter_ratio'] = letters / len(url) if len(url) > 0 else 0
        features['special_char_ratio'] = special_chars / \
            len(url) if len(url) > 0 else 0

        features['longest_digit_sequence'] = self._longest_sequence_extraction(
            url, str.isdigit)
        features['longest_letter_sequence'] = self._longest_sequence_extraction(
            url, str.isalpha)
        features['longest_special_sequence'] = self._longest_sequence_extraction(
            url, lambda c: not c.isalnum())

        features['character_continuity_rate'] = self._calculate_continuity(url)
        features['url_entropy'] = self._calculate_entropy(url)

        # Threshold-based features (5 features)
        features['low_entropy'] = 1.0 if features['url_entropy'] < 3.0 else 0.0
        features['high_entropy'] = 1.0 if features['url_entropy'] > 4.0 else 0.0
        features['long_url_length'] = 1.0 if features['url_length'] > 100 else 0.0
        features['high_digit_ratio'] = 1.0 if features['digit_ratio'] > 0.3 else 0.0
        features['low_special_char_ratio'] = 1.0 if features['special_char_ratio'] < 0.05 else 0.0

        return features

    def _extract_domain_level(self, parsed) -> Dict[str, float]:
        features = {}
        domain = parsed.netloc

        features['domain_length'] = len(domain)

        domain_parts = domain.split('.')
        features['domain_token_count'] = len(domain_parts)
        features['subdomain_count'] = max(0, len(domain_parts) - 2)

        tld = domain_parts[-1] if domain_parts else ''

        digits = sum(c.isdigit() for c in domain)
        features['digit_ratio_domain'] = digits / \
            len(domain) if len(domain) > 0 else 0

        features['hyphen_count_domain'] = domain.count('-')

        if domain_parts:
            token_lengths = [len(part) for part in domain_parts]
            features['longest_domain_token'] = max(token_lengths)
            features['avg_domain_token_length'] = sum(
                token_lengths) / len(token_lengths)
        else:
            features['longest_domain_token'] = 0
            features['avg_domain_token_length'] = 0

        features['domain_entropy'] = self._calculate_entropy(domain)

        # Binary flags
        features['ip_address_flag'] = 1.0 if self._is_ip_address(
            domain) else 0.0
        features['multiple_subdomain_flag'] = 1.0 if features['subdomain_count'] > 1 else 0.0
        features['brand_keyword_flag'] = 1.0 if self._has_brand_keywords(
            domain) else 0.0

        features['port_in_url_flag'] = 1.0 if ':' in parsed.netloc and not self._is_ip_address(
            domain) else 0.0
        features['https_in_domain'] = 1.0 if 'https' in domain.lower() else 0.0
        features['tld_suspicious_flag'] = 1.0 if self._has_suspicious_tld(
            tld) else 0.0
        features['shortening_service_flag'] = 1.0 if self._is_url_shortener(
            domain) else 0.0
        features['double_slash_redirecting'] = 1.0 if '//' in parsed.path else 0.0

        features['punycode_domain_flag'] = 1.0 if domain.startswith(
            'xn--') or 'xn--' in domain else 0.0
        features['unicode_domain_flag'] = 1.0 if any(
            ord(c) > 127 for c in domain) else 0.0
        features['homograph_suspicious_flag'] = 1.0 if self._has_homograph_chars(
            domain) else 0.0

        features['excessive_subdomain_depth'] = max(
            0, features['subdomain_count'] - 4)
        features['random_string_domain_flag'] = 1.0 if self._is_random_domain(
            domain) else 0.0
        features['free_hosting_domain_flag'] = 1.0 if self._is_free_hosting(
            domain) else 0.0
        features['dga_domain_flag'] = 1.0 if self._is_dga_domain(
            domain) else 0.0
        features['non_standard_port_flag'] = 1.0 if self._has_non_standard_port(
            parsed.netloc) else 0.0

        features['high_domain_entropy'] = 1.0 if features['domain_entropy'] > 3.8 else 0.0

        return features

    def _extract_path_level(self, parsed) -> Dict[str, float]:
        features = {}
        path = parsed.path

        features['path_length'] = len(path)

        path_tokens = [p for p in path.split('/') if p]
        features['path_token_count'] = len(path_tokens)

        digits = sum(c.isdigit() for c in path)
        features['digit_ratio_path'] = digits / \
            len(path) if len(path) > 0 else 0

        features['dot_count_path'] = path.count('.')

        if path_tokens:
            token_lengths = [len(token) for token in path_tokens]
            features['longest_path_token'] = max(
                token_lengths) if token_lengths else 0
            features['avg_path_token_length'] = sum(
                token_lengths) / len(token_lengths) if token_lengths else 0
        else:
            features['longest_path_token'] = 0
            features['avg_path_token_length'] = 0

        features['path_entropy'] = self._calculate_entropy(path)

        filename = path_tokens[-1] if path_tokens else ''
        features['filename_length'] = len(filename)

        suspicious_extensions = ['.exe', '.bat',
                                 '.cmd', '.scr', '.vbs', '.js', '.jar', '.apk', '.sh']
        features['suspicious_extension_flag'] = 1.0 if any(
            filename.endswith(ext) for ext in suspicious_extensions) else 0.0
        features['executable_extension_flag'] = 1.0 if any(filename.endswith(
            ext) for ext in ['.exe', '.bat', '.cmd', '.scr']) else 0.0
        features['suspicious_js_extension_flag'] = 1.0 if (path.endswith(
            '.js') or 'javascript:' in parsed.geturl().lower()) else 0.0

        return features

    def _extract_query_level(self, parsed) -> Dict[str, float]:
        features = {}
        query = parsed.query

        features['query_length'] = len(query)

        params = parse_qs(query)
        features['parameter_count'] = len(params)

        digits = sum(c.isdigit() for c in query)
        features['digit_ratio_query'] = digits / \
            len(query) if len(query) > 0 else 0

        features['equal_count_query'] = query.count('=')
        features['ampersand_count_query'] = query.count('&')

        if params:
            param_lengths = [len(k) + len(''.join(v))
                             for k, v in params.items()]
            features['avg_parameter_length'] = sum(
                param_lengths) / len(param_lengths) if param_lengths else 0
            features['max_parameter_length'] = max(
                param_lengths) if param_lengths else 0
        else:
            features['avg_parameter_length'] = 0
            features['max_parameter_length'] = 0

        features['query_entropy'] = self._calculate_entropy(query)

        # Binary flags
        features['encoded_url_flag'] = 1.0 if '%' in query else 0.0
        features['redirect_parameter_flag'] = 1.0 if any(param in query.lower(
        ) for param in ['url=', 'redirect=', 'next=', 'return=', 'goto=', 'redirect_url=', 'return_url=', 'next_url=', 'goto_url=', 'dest=', 'destination=', 'target=', 'rurl=', 'ru=', 'back=', 'callback=', 'continue=', 'link=', 'path=', 'ref=', 'referrer=', 'site=', 'to=', 'uri=', 'u=', 'redirecturl=']) else 0.0

        full_url = parsed.geturl().lower()
        features['script_in_url_flag'] = 1.0 if 'script' in full_url else 0.0
        features['auto_download_param_flag'] = 1.0 if any(param in query.lower(
        ) for param in self._auto_download_params) else 0.0
        features['base64_in_url_flag'] = 1.0 if self._has_base64_encoding(
            full_url) else 0.0

        return features

    def _extract_structural_ratios(self, url: str, parsed) -> Dict[str, float]:
        features = {}
        url_length = len(url)

        features['domain_url_ratio'] = len(
            parsed.netloc) / url_length if url_length > 0 else 0
        features['path_url_ratio'] = len(
            parsed.path) / url_length if url_length > 0 else 0

        # digits = sum(c.isdigit() for c in url)
        # features['overall_digit_ratio'] = digits / \
        #     url_length if url_length > 0 else 0

        # special_chars = sum(not c.isalnum() for c in url)
        # features['overall_special_char_ratio'] = special_chars / \
        #     url_length if url_length > 0 else 0

        return features

    def _extract_sequence_patterns(self, url: str) -> Dict[str, float]:
        features = {}

        features['mixed_token_flag'] = 1.0 if re.search(
            r'[a-zA-Z]+\d+|[0-9]+[a-zA-Z]+', url) else 0.0

        features['hex_encoding_flag'] = 1.0 if re.search(
            r'%[0-9a-fA-F]{2}', url) else 0.0
        features['obfuscation_pattern_flag'] = 1.0 if re.search(
            r'[a-zA-Z0-9]{20,}', url) else 0.0

        return features

    def _longest_sequence_extraction(self, text: str, condition) -> int:
        max_len = 0
        current_len = 0
        for char in text:
            if condition(char):
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len

    # https://medium.com/@ashetty.undef/ml-for-shortlisting-malicious-url-bf6f96456a5a

    def _calculate_continuity(self, text: str) -> float:
        if len(text) == 0:
            return 0.0

        longest_alpha = self._longest_sequence_extraction(text, str.isalpha)
        longest_digit = self._longest_sequence_extraction(text, str.isdigit)
        longest_special = self._longest_sequence_extraction(
            text, lambda c: not c.isalnum())

        continuity_rate = (longest_alpha + longest_digit +
                           longest_special) / len(text)
        return continuity_rate

    def _calculate_entropy(self, text: str) -> float:
        if not text or len(text) == 0:
            return 0.0

        counter = Counter(text)
        length = len(text)

        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        return float(entropy)

    def _is_ip_address(self, domain: str) -> bool:
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, domain))

    def _has_brand_keywords(self, domain: str) -> bool:
        domain_lower = domain.lower()
        return any(keyword in domain_lower for keyword in self._brand_keywords)

    def _has_suspicious_tld(self, tld: str) -> bool:
        return tld.lower() in self._suspicious_tlds

    def _is_url_shortener(self, domain: str) -> bool:
        domain_lower = domain.lower()
        return any(shortener in domain_lower for shortener in self._url_shorteners)

    def _has_homograph_chars(self, domain: str) -> bool:
        cyrillic_chars = set('авсԁеһіјӏорԛѕԝхуᴢ')
        greek_chars = set('αβγδεζηθικλμνξοπρστυφχψω')
        return any(c.lower() in cyrillic_chars or c.lower() in greek_chars for c in domain)

    def _is_random_domain(self, domain: str) -> bool:
        domain_name = domain.split('.')[0] if '.' in domain else domain
        if len(domain_name) < 8:
            return False
        entropy = self._calculate_entropy(domain_name)
        return entropy > 3.5

    def _is_free_hosting(self, domain: str) -> bool:
        domain_lower = domain.lower()
        return any(host in domain_lower for host in self._free_hosts)

    def _is_dga_domain(self, domain: str) -> bool:
        domain_name = domain.split('.')[0] if '.' in domain else domain
        if len(domain_name) < 10:
            return False
        vowels = sum(1 for c in domain_name.lower() if c in 'aeiou')
        consonants = sum(1 for c in domain_name.lower()
                         if c.isalpha() and c not in 'aeiou')
        if consonants == 0:
            return False
        vowel_ratio = vowels / (vowels + consonants)
        entropy = self._calculate_entropy(domain_name)
        return entropy > 3.8 and (vowel_ratio < 0.2 or vowel_ratio > 0.6)

    def _has_non_standard_port(self, netloc: str) -> bool:
        if ':' not in netloc:
            return False
        try:
            port = netloc.split(':')[-1]
            return port in self._non_standard_ports
        except ValueError:
            return False

    def _has_base64_encoding(self, url: str) -> bool:
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        return bool(re.search(base64_pattern, url))

    def _has_suspicious_keywords(self, url: str) -> bool:
        keywords = ['admin', 'login', 'verify', 'update',
                    'secure', 'account', 'confirm', 'password', 'signin']
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in keywords)


feature_extractor = FeatureExtractor()
