import os

import pandas as pd


def main(pre_select):
    success_info = pd.concat([
        pd.read_csv('e:/code/dataset/legal/new_url_info.csv')["success"],
        pd.read_csv('e:/code/dataset/phishing/new_url_info.csv')["success"]
    ], ignore_index=True)
    legal_page_res_features = pd.concat([
        pd.read_csv(f"{pre_select}pilot_process/after_legal_page_features.csv"),
        pd.read_csv(f"{pre_select}pilot_process/after_legal_res_features.csv"),
    ], axis=1)
    phishing_page_res_features = pd.concat([
        pd.read_csv(f"{pre_select}pilot_process/after_phishing_page_features.csv"),
        pd.read_csv(f"{pre_select}pilot_process/after_phishing_res_features.csv"),
    ], axis=1)
    features = pd.concat([
        pd.read_csv(f"{pre_select}pilot_process/url_features.csv"),
        pd.concat([legal_page_res_features, phishing_page_res_features], ignore_index=True),
        pd.read_csv(f"{pre_select}pilot_process/dyn_features.csv"),
        success_info
    ], axis=1)
    features.to_csv(f"output_feature/{pre_select}features.csv", index=False)


if __name__ == "__main__":
    pre = "no_eli_set_"
    main(pre)
