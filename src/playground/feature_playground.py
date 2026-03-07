from urllib.parse import urlparse
import re

features = {}

url = 'http://7x24v9n.ru/login?token=12345&sT=aaa111222555555'
domain = urlparse(url).netloc

count = sum(c.isdigit() for c in domain)
features['digit_count_domain'] = count
features['digit_ratio_domain'] = count / len(domain) if len(domain) > 0 else 0
features['repeated_character_flag'] = 1.0 if re.search(
    r'(.)\1{2,}', url) else 0.0

print(features)
