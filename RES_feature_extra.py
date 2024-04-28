import json
import pickle
from multiprocessing import Process
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm


class ResFeatureExtra:
    def __init__(self, res_pickle_path, request_url, current_url, eliminate_ls=None):
        with open(res_pickle_path, "rb") as f:
            self.res_pickle = pickle.load(f)
        self.request_url = request_url
        self.current_url = current_url
        self.eliminate_ls = eliminate_ls
        self.feature_name = (
            "res_is_redirect", "res_headers_type_num", "res_constitute_url_num",
            "res_constitute_domain_type"
        )

    def res_is_redirect(self):
        parse_res = urlparse(self.current_url)
        parse_req = urlparse(self.request_url)
        if parse_res.netloc.startswith("www."):
            res_domain = parse_res.netloc[len("www."):]
        else:
            res_domain = parse_res.netloc
        if parse_req.netloc.startswith("www."):
            req_domain = parse_req.netloc[len("www."):]
        else:
            req_domain = parse_req.netloc
        if res_domain == req_domain:
            return {"res_is_redirect": 0}
        return {"res_is_redirect": 1}

    def res_headers_type_num(self):
        return {"res_headers_type_num": len(self.res_pickle[0].headers.keys())}

    def res_constitute_url_num(self):
        return {"res_constitute_url_num": len(self.res_pickle)}

    def res_constitute_domain_type(self):
        type_set = set()
        for link in self.res_pickle:
            parse_link = urlparse(link.url)
            if self.eliminate_ls and parse_link.netloc in self.eliminate_ls:
                continue
            type_set.add(parse_link.netloc)
        return {"res_constitute_domain_type": len(type_set)}

    def handle(self):
        feature_dict = {}
        for func_name in self.feature_name:
            function = getattr(self, func_name, None)
            feature_dict.update(function())
        return feature_dict

    @classmethod
    def get_default_feature(cls):
        return {
            'res_is_redirect': 0,
            'res_headers_type_num': 0,
            'res_constitute_url_num': 0,
            'res_constitute_domain_type': 0
        }


def get_lim_set(eli_num=2, before=True):
    with open("pilot_process/graph_info.json") as f:
        json_file = json.load(f)
    if before:
        legal_set = set(sum(json_file["before_constitute"][0], []))
        phishing_set = set(sum(json_file["before_constitute"][1], []))
        legal_domain_num = json_file["before_legal_domain_num"]
    else:
        legal_set = set(sum(json_file["after_constitute"][0], []))
        phishing_set = set(sum(json_file["after_constitute"][1], []))
        legal_domain_num = json_file["after_legal_domain_num"]
    intersection_set = legal_set.intersection(phishing_set)
    for key, value in legal_domain_num.items():
        if value >= eli_num:
            intersection_set.add(key)
    return intersection_set


def res_feature_extra(info_path, legal=True, before=True, eli_set=None):
    df = pd.read_csv(info_path + "/new_url_info.csv")
    feature_ls = []
    for row in tqdm(df.itertuples(), total=(len(df)), desc=str(legal) + "-" + str(before)):
        if row.state_code == 200 and row.success:
            if before:
                row_res_features = ResFeatureExtra(
                    info_path + "/" + str(row.num) + "-" + "200/before_response.pickle", row.request_url,
                    row.response_url, eli_set).handle()
            else:
                row_res_features = ResFeatureExtra(
                    info_path + "/" + str(row.num) + "-" + "200/after_response.pickle",
                    row.request_url, row.response_url, eli_set).handle()
        else:
            row_res_features = ResFeatureExtra.get_default_feature()
        row_res_features["res_state_code"] = row.state_code
        feature_ls.append(row_res_features)
    features = pd.DataFrame(feature_ls)
    if legal:
        if before:
            save_path = "before_legal_res_features.csv"
        else:
            save_path = "after_legal_res_features.csv"
    else:
        if before:
            save_path = "before_phishing_res_features.csv"
        else:
            save_path = "after_phishing_res_features.csv"
    features.to_csv("no_eli_set_pilot_process/" + save_path, index=False)


def run_res_feature_extra(info_path, legal, before, ):
    # lim_set = get_lim_set(before=before)
    res_feature_extra(info_path, legal, before)


if __name__ == "__main__":
    processes = [Process(target=run_res_feature_extra, args=("E:/code/dataset/phishing/", False, False)),
                 Process(target=run_res_feature_extra, args=("E:/code/dataset/phishing/", False, True)),
                 Process(target=run_res_feature_extra, args=("E:/code/dataset/legal/", True, True)),
                 Process(target=run_res_feature_extra, args=("E:/code/dataset/legal/", True, False))]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
