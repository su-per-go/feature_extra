import json
import pickle
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm


class GraphInfo:
    def __init__(self, legal_path, phishing_path, save_feature_path):
        self.legal_path = legal_path
        self.phishing_path = phishing_path
        self.save_feature_path = save_feature_path
        self.before_constitute = [[], []]  # legal   phishing
        self.after_constitute = [[], []]  # legal   phishing
        self.before_legal_domain_num = {}
        self.after_legal_domain_num = {}
        self.before_legal_url_ls = []
        self.after_legal_url_ls = []

    def handel(self):
        legal_url_info = pd.read_csv(self.legal_path + "/new_url_info.csv")
        phishing_url_info = pd.read_csv(self.phishing_path + "/new_url_info.csv")
        self.get_constitute_domain_info(legal_url_info, phishing_url_info, before=True,
                                        desc="before_constitute")
        self.get_constitute_domain_info(legal_url_info, phishing_url_info, before=False,
                                        desc="after_constitute")
        self.save_graph_info()

    def get_constitute_domain_info(self, legal_info, phishing_info, before=True, desc=""):
        for row in tqdm(legal_info.itertuples(), desc=desc + " legal"):
            if row.state_code == 200 and row.success:
                if before:
                    constitute = self.read_pickle(self.legal_path + "/" + str(row.num) + "-" + str(
                        row.state_code) + "/before_response.pickle")
                else:
                    constitute = self.read_pickle(self.legal_path + "/" + str(row.num) + "-" + str(
                        row.state_code) + "/after_response.pickle")
                url_ls = []
                random_select_url = []
                link_set = set()
                for link in constitute:
                    parse_link = urlparse(link.url)
                    parse_request_url = urlparse(row.request_url)
                    if parse_request_url.netloc == parse_link.netloc or "www." + parse_request_url.netloc == parse_link.netloc:
                        random_select_url.append(link.url)
                    url_ls.append(parse_link.netloc)
                    if before:  # 计算每个域名出现次数
                        if parse_link.netloc in link_set:
                            continue
                        link_set.add(parse_link.netloc)
                        self.before_legal_domain_num[parse_link.netloc] = self.before_legal_domain_num.get(
                            parse_link.netloc, 0) + 1
                    else:
                        if parse_link.netloc in link_set:
                            continue
                        link_set.add(parse_link.netloc)
                        self.after_legal_domain_num[parse_link.netloc] = self.after_legal_domain_num.get(
                            parse_link.netloc, 0) + 1
                if before:
                    self.before_constitute[0].append(url_ls)
                else:
                    self.after_constitute[0].append(url_ls)
                if before:
                    if len(random_select_url) and not row.login_page:
                        import random
                        self.before_legal_url_ls.append(random.choice(random_select_url))
                    else:
                        self.before_legal_url_ls.append(row.request_url)
                else:
                    if len(random_select_url) and not row.login_page:
                        import random
                        self.after_legal_url_ls.append(random.choice(random_select_url))
                    else:
                        self.after_legal_url_ls.append(row.request_url)
            else:
                if before:
                    self.before_constitute[0].append([])
                    self.before_legal_url_ls.append(row.request_url)
                else:
                    self.after_constitute[0].append([])
                    self.after_legal_url_ls.append(row.request_url)

        for row in tqdm(phishing_info.itertuples(), desc=desc + " phishing"):
            if row.state_code == 200 and row.success:
                if before:
                    constitute = self.read_pickle(
                        self.phishing_path + "/" + str(row.num) + "-" + str(
                            row.state_code) + "/before_response.pickle")
                else:
                    constitute = self.read_pickle(
                        self.phishing_path + "/" + str(row.num) + "-" + str(
                            row.state_code) + "/after_response.pickle")
                url_ls = []
                for link in constitute:
                    parse_link = urlparse(link.url)
                    url_ls.append(parse_link.netloc)
                if before:
                    self.before_constitute[1].append(url_ls)
                else:
                    self.after_constitute[1].append(url_ls)
            else:
                if before:
                    self.before_constitute[1].append([])
                else:
                    self.after_constitute[1].append([])

    def save_graph_info(self):
        with open(self.save_feature_path, "w") as f:
            json.dump({
                "before_constitute": self.before_constitute,
                "after_constitute": self.after_constitute,
                "before_legal_domain_num": self.before_legal_domain_num,
                "after_legal_domain_num": self.after_legal_domain_num,
                "before_legal_url_ls": self.before_legal_url_ls,
                "after_legal_url_ls": self.after_legal_url_ls,
            }, f)

    @staticmethod
    def read_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    GraphInfo("E:/code/dataset/legal", "E:/code/dataset/phishing", "pilot_process/graph_info.json").handel()
