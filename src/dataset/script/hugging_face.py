from datasets import load_dataset

'''
# Login with hugging face

hf auth login
'''

ds2 = load_dataset("stanpony/phishing_urls")
print(ds2)

for split_name, split_data in ds2.items():
    split_name = split_name + "2"

    split_data.to_csv(f'malicious_url_{split_name}.csv', index=False)
    print(f"Exported ds2 {split_name} split to malicious_url_{split_name}.csv")
