# Dataset Quality Report

Generated: `2026-07-14`

Source dataset:

- `src/dataset/raw/malicious_url_train2.csv`
- Sample evaluated: first `1000` rows

## Results

- Validation status: `failed`
- Reason: `13` invalid URLs were detected in the sampled rows
- Duplicate normalized URLs: `0`
- Conflicting labels: `0`
- Missing labels: `0`
- Invalid labels: `0`
- Empty `type` values: `1000`
- Valid records after cleaning: `987`
- Unique registered domains: `770`
- Class distribution:
  - `benign`: `751`
  - `malicious`: `236`
- Imbalance severity: `mild`
- Imbalance ratio: `3.1822`
- Dataset hash: `acafc6d42709098035af1930ab8390d34918b337fba1d7888c4215a946ce6b9f`

## Invalid URL examples

- `cnhedge.cn/js/index.htm?http://us.battle.net/login/en/?ref=http://utjrrbhus.battle.net/d3/en/index&amp;app=com-d3`
- `lidanhang.com/img/?https://secure.runescape.com/m=weblogin/loginform.ws?mod=www&amp;nzeozmwamp;`
- `two.hfdodiopr.biz/40d7vetq42\ndkoawit6.com/ik/redir.php?url=http://two.hfdodiopr.biz/40d7vetq42`
- `ysgrp.com.cn/js/?ref=http://us.battle.net/d3/en`

## Notes

- The report was generated with the new validator and normalizer.
- URLs with invalid host/path structure are counted before cleaning and excluded from the cleaned dataset metadata.
