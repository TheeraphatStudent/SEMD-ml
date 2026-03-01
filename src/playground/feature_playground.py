from urllib.parse import urlparse

features = {}

url = 'http://7x24v9n.ru/login?token=12345'
domain = urlparse(url).netloc

count = sum(c.isdigit() for c in domain)
features['digit_count_domain'] = count
features['digit_ratio_domain'] = count / len(domain) if len(domain) > 0 else 0

print(features)
