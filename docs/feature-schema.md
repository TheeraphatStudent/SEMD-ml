# Feature Schema

## Version

- Schema version: `1.0.0`
- Canonical feature count: `73`

## Contract

Training and inference now share one canonical feature schema:

- `src/semd_ml/features/schema.py`
- `src/semd_ml/features/extractor.py`
- `src/ml/ml_pipeline.py`

The schema defines:

- Feature name
- Data type
- Description
- Expected range
- Default value
- Schema version

Inference behavior:

- If a model artifact includes `feature_schema_<run_id>.json`, inference uses that schema directly.
- Legacy artifacts fall back to scaler column order.

## Feature groups

### URL level

- `http_token`
- `url_length`
- `dot_count`
- `hyphen_count`
- `slash_count`
- `at_symbol_count`
- `percent_count`
- `special_char_count`
- `digit_ratio`
- `letter_ratio`
- `special_char_ratio`
- `longest_digit_sequence`
- `longest_letter_sequence`
- `longest_special_sequence`
- `character_continuity_rate`
- `url_entropy`
- `low_entropy`
- `high_entropy`
- `long_url_length`
- `high_digit_ratio`
- `low_special_char_ratio`

### Domain level

- `domain_length`
- `domain_token_count`
- `subdomain_count`
- `digit_ratio_domain`
- `hyphen_count_domain`
- `longest_domain_token`
- `avg_domain_token_length`
- `domain_entropy`
- `ip_address_flag`
- `multiple_subdomain_flag`
- `brand_keyword_flag`
- `port_in_url_flag`
- `https_in_domain`
- `tld_suspicious_flag`
- `shortening_service_flag`
- `double_slash_redirecting`
- `punycode_domain_flag`
- `unicode_domain_flag`
- `homograph_suspicious_flag`
- `excessive_subdomain_depth`
- `random_string_domain_flag`
- `free_hosting_domain_flag`
- `dga_domain_flag`
- `non_standard_port_flag`
- `high_domain_entropy`

### Path level

- `path_length`
- `path_token_count`
- `digit_ratio_path`
- `dot_count_path`
- `longest_path_token`
- `avg_path_token_length`
- `path_entropy`
- `filename_length`
- `suspicious_extension_flag`
- `executable_extension_flag`
- `suspicious_js_extension_flag`

### Query level

- `query_length`
- `parameter_count`
- `digit_ratio_query`
- `equal_count_query`
- `ampersand_count_query`
- `avg_parameter_length`
- `max_parameter_length`
- `query_entropy`
- `encoded_url_flag`
- `redirect_parameter_flag`
- `script_in_url_flag`
- `auto_download_param_flag`
- `base64_in_url_flag`

### Sequence patterns

- `mixed_token_flag`
- `hex_encoding_flag`
- `obfuscation_pattern_flag`

## Audited behavior notes

- `port_in_url_flag` means an explicit port was present, including default ports.
- `non_standard_port_flag` is the risk signal for non-default or suspicious ports.
- `punycode_domain_flag` is derived from normalized hostname labels beginning with `xn--`.
- `unicode_domain_flag` indicates non-ASCII characters in the raw URL input.
- Entropy features use Shannon entropy.
- Feature extraction is pure string parsing and has no network dependency.
- The schema excludes labels and any derived target information, preventing label leakage.
