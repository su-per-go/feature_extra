import json
import math
import re
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from auto_select import auto_select_embedding_len


class URLFeatureExtra:
    def __init__(self, url, label):
        self.url = url
        self.pre_url, self.suf_url = self.split_url(self.url)
        self.label = label
        self.feature_name = (
            "url_split_len", "url_url_entropy", "url_pre_suf_entropy", "url_https_count", "url_http_count",
            "url_count_special_characters", "url_numbers_in_domain"
        )
        # self.best_pre_len = best_pre_len
        # self.best_suf_len = best_suf_len

    def url_split_len(self):
        parsed_url = urlparse(self.url)
        return {
            "url_url": self.url,
            "url_url_pre": self.pre_url,
            "url_url_suf": self.suf_url,
            "url_len_url": len(self.url),
            "url_len_pre": len(self.pre_url),
            "url_len_suf": len(self.suf_url),
            "url_len_params": len(parsed_url.params),
            "url_len_query": len(parsed_url.query),
            "url_len_fragment": len(parsed_url.fragment)
        }

    def url_url_entropy(self):
        return {
            "url_pre_entropy": self.calculate_entropy(self.pre_url),
            "url_suf_entropy": self.calculate_entropy(self.suf_url),
        }

    def url_pre_suf_entropy(self):
        return {
            "url_pre_entropy": self.calculate_entropy(self.pre_url),
            "url_suf_entropy": self.calculate_entropy(self.suf_url)
        }

    def url_https_count(self):
        return {
            "url_pre_https_count": self.pre_url.count("https"),
            "url_suf_https_count": self.suf_url.count("https"),
        }

    def url_http_count(self):
        pattern = r'http(?![sS])'
        return {
            "url_pre_http_count": len(re.findall(pattern, self.pre_url)),
            "url_suf_http_count": len(re.findall(pattern, self.suf_url)),
        }

    def url_www_count(self):
        return {
            "url_pre_www_count": self.pre_url.count("www."),
            "url_suf_www_count": self.suf_url.count("www.")
        }

    def url_count_special_characters(self):
        special_characters = ["/", "%", "=", "-", "@", "."]
        counts = {"url_pre_count" + char: self.pre_url.count(char) for char in special_characters}
        counts.update({"url_suf_count" + char: self.suf_url.count(char) for char in special_characters})

        return counts

    def url_numbers_in_domain(self):
        number_domain = {
            "url_pre_numbers_in_domain": sum(1 for char in urlparse(self.pre_url).netloc if char.isnumeric())}
        number_domain.update(
            {"url_suf_numbers_in_domain": sum(1 for char in urlparse(self.suf_url).netloc if char.isnumeric())})
        return number_domain

    @staticmethod
    def calculate_entropy(text):
        char_frequency = {}
        for char in text:
            char_frequency[char] = char_frequency.get(char, 0) + 1

        total_chars = len(text)
        entropy = -sum((freq / total_chars) * math.log2(freq / total_chars) for freq in char_frequency.values())
        return entropy

    @staticmethod
    def split_url(url):
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        # 提取路径
        path = parsed_url.path
        if len(domain_name) == 0:
            path = path.lstrip("/")
            sp_url = path.split("/", 1)
            prefix = sp_url[0]
            suffix = sp_url[1] if len(sp_url) > 1 else ""
        else:
            prefix = parsed_url.scheme + "://" + parsed_url.netloc
            suffix = parsed_url.path + parsed_url.params + parsed_url.query + parsed_url.fragment
        return prefix, suffix

    def handle(self):
        feature_dict = {}
        for func_name in self.feature_name:
            function = getattr(self, func_name, None)
            feature_dict.update(function())
        return {**feature_dict, "label": self.label}


def url_feature_extra(phishing_url_info_path, legal_url_info_path, graph_info_path, save_path):
    with open(graph_info_path, "r") as f:
        json_file = json.load(f)
    phishing_df = pd.read_csv(phishing_url_info_path)
    legal_df = pd.read_csv(legal_url_info_path)
    legal_url_features = []
    for index, url in tqdm(enumerate(json_file["after_legal_url_ls"]), total=(len(json_file["after_legal_url_ls"])),
                           desc="legal_URL_feat_extra"):
        feature_result = URLFeatureExtra(url, 0, ).handle()
        feature_result["index"] = legal_df.loc[index, "num"]
        legal_url_features.append(feature_result)

    phishing_url_features = []
    for index, row in tqdm(enumerate(phishing_df.itertuples()), total=(len(phishing_df)),
                           desc="phishing_URL_feat_extra"):
        feature_result = URLFeatureExtra(row.request_url, 1).handle()
        feature_result["index"] = phishing_df.loc[index, "num"]
        phishing_url_features.append(feature_result)
    url_features = pd.DataFrame(legal_url_features + phishing_url_features)

    url_features.to_csv(save_path, index=False)


if __name__ == "__main__":
    url_feature_extra("E:/code/dataset/phishing/new_url_info.csv", "E:/code/dataset/legal/new_url_info.csv",
                      "pilot_process/graph_info.json", "pilot_process/url_features.csv")
