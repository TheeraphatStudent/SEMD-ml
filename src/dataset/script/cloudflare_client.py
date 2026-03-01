src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, src_dir)
os.chdir(src_dir)

import sys
import os
import json

import datetime
import pandas as pd
import csv
import gzip
from dateutil.relativedelta import relativedelta

from cloudflare import Cloudflare
from core import settings

client = Cloudflare(
    api_token=settings.cloudflare_api_token,
)

# ------- Get malicious scan url
# - https://developers.cloudflare.com/api/resources/url_scanner/subresources/responses/methods/get/

# malicious_scans = client.url_scanner.scans.list(
#     account_id=settings.cloudflare_account_id,
#     q="verdicts.malicious:true AND date:[2025-02 TO 2025-03]", # ต้องทำ Bulk query 2025-02 TO ปัจจุบัน
#     size=5
# )

# print(json.dumps(malicious_scans, default=lambda o: o.__dict__, indent=2))

# + Map to url (task.url | page.url), label (verdicts.malicious)
# url, label
# https://gceearthworks.com.au/, malicious

def cloudflare_malicious_scans(start_date_str="2025-01", output_dir="./dataset/store"):
    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m")
    end_date = datetime.datetime.now()

    current_date = start_date
    all_results = pd.DataFrame()
    
    print(f"Fetching malicious scans from {start_date_str} to {end_date.strftime('%Y-%m')}")
    
    while current_date <= end_date:
        next_month = current_date + relativedelta(months=1)
        
        date_start = current_date.strftime("%Y-%m")
        date_end = next_month.strftime("%Y-%m")
        
        query = f"verdicts.malicious:true AND date:[{date_start} TO {date_end}]"
        
        try:
            scans = client.url_scanner.scans.list(
                account_id=settings.cloudflare_account_id,
                q=query,
                size=10000 
            )
            
            if hasattr(scans, 'results'):
                for scan in scans.results:
                    url = None
                    if hasattr(scan, 'task') and hasattr(scan.task, 'url'):
                        url = scan.task.url
                    elif hasattr(scan, 'page') and hasattr(scan.page, 'url'):
                        url = scan.page.url
                    
                    label = 'malicious'
                    if hasattr(scan, 'verdicts') and hasattr(scan.verdicts, 'malicious'):
                        label = 'malicious' if scan.verdicts.malicious else 'benign'
                    
                    if url:
                        all_results = all_results.append({'url': url, 'label': label}, ignore_index=True)
            
        except Exception as e:
            print(f"Error fetching data for {date_start}: {e}")
        
        current_date = next_month
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cloudflare_malicious_scans_{timestamp}.csv.gz")
    
    with gzip.open(output_file, 'wt', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'label'])
        writer.writeheader()
        writer.writerows(all_results.values)
    
    print(f"Exported {len(all_results)} records to {output_file}")
    return output_file

cloudflare_malicious_scans()

# ------ List datasets

# datasets = client.radar.datasets.list(
#     format='JSON'
# )

# print(json.dumps(datasets.datasets, default=lambda o: o.__dict__, indent=2))

# ------ Summarize overview

# def custom_serializer(obj):
#     if hasattr(obj, '__dict__'):
#         return obj.__dict__
#     elif isinstance(obj, (datetime.datetime, datetime.date)):
#         return obj.isoformat()
#     else:
#         return str(obj)

# summary = client.radar.email.security.summary.malicious(
#     date_range="14d",
# )

# print(json.dumps(summary, default=custom_serializer, indent=2))