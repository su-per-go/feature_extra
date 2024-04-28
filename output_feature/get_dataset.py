import pandas as pd
from URL_feature_extra import URLFeatureExtra
import numpy as np


def calculate_condition(path, save_name, data_num):
    features_df = pd.read_csv(path)
    legal_num = (features_df["label"] == 0).sum()
    phishing_num = (features_df["label"] == 1).sum()
    #  检查合法或非法是否满足数目要求
    if legal_num < data_num * 0.5:
        raise ValueError(f"legal_num {legal_num} is less than need num {int(data_num * 0.5)}")
    if phishing_num < data_num * 0.5:
        raise ValueError(f"legal_num {phishing_num} is less than need num {int(data_num * 0.5)}")

    # 合法 访问成功和访问失败的包含前后缀数据个数
    legal_access_success_pre = features_df[(features_df["label"] == 0) & (features_df["url_len_suf"] < 2) &
                                           (features_df["res_state_code"] < 300) & (
                                                   200 <= features_df["res_state_code"]) & (
                                                   1 == features_df["success"])]

    legal_access_success_pre_suf = features_df[(features_df["label"] == 0) & (features_df["url_len_suf"] >= 2) &
                                               (features_df["res_state_code"] < 300) & (
                                                       200 <= features_df["res_state_code"]) & (
                                                       1 == features_df["success"])]

    legal_access_fail_pre = features_df[((features_df["label"] == 0) & (features_df["url_len_suf"] < 2) &
                                         (features_df["res_state_code"] >= 300)) | (
                                                (features_df["label"] == 0) & (features_df["url_len_suf"] < 2) &
                                                (features_df["res_state_code"] == 200) & (
                                                        0 == features_df["success"]))]

    legal_access_fail_pre_suf = features_df[(features_df["label"] == 0) & (features_df["url_len_suf"] >= 2) &
                                            (features_df["res_state_code"] >= 300) | (
                                                    (features_df["label"] == 0) & (features_df["url_len_suf"] >= 2) &
                                                    (features_df["res_state_code"] == 200) & (
                                                            0 == features_df["success"]))]
    # 钓鱼  访问成功和访问失败的包含前后缀数据个数
    phishing_access_success_pre = features_df[(features_df["label"] == 1) & (features_df["url_len_suf"] < 2) &
                                              (features_df["res_state_code"] < 300) & (
                                                      200 <= features_df["res_state_code"]) & (
                                                      1 == features_df["success"])]
    phishing_access_success_pre_suf = features_df[(features_df["label"] == 1) & (features_df["url_len_suf"] >= 2) &
                                                  (features_df["res_state_code"] < 300) & (
                                                          200 <= features_df["res_state_code"]) & (
                                                          1 == features_df["success"])]

    phishing_access_fail_pre = features_df[((features_df["label"] == 1) & (features_df["url_len_suf"] < 2) &
                                            (features_df["res_state_code"] >= 300)) | (
                                                   (features_df["label"] == 1) & (features_df["url_len_suf"] < 2) &
                                                   (features_df["res_state_code"] == 200) & (
                                                           0 == features_df["success"]))]

    phishing_access_fail_pre_suf = features_df[((features_df["label"] == 1) & (features_df["url_len_suf"] >= 2) &
                                                (features_df["res_state_code"] >= 300)) | (
                                                       (features_df["label"] == 1) & (
                                                       features_df["url_len_suf"] >= 2) &
                                                       (features_df["res_state_code"] == 200) & (
                                                               0 == features_df["success"]))]
    # 数据集生成
    need = pd.DataFrame()
    num_atom = int(data_num * 6.25 / 100)
    need = update_need_pre(need, legal_access_success_pre, num_atom,
                           legal_access_success_pre_suf,
                           "legal_access_success_pre")
    need = update_need_pre(need,
                           legal_access_fail_pre, num_atom,
                           legal_access_fail_pre_suf,
                           "legal_access_fail_pre"
                           )
    need = update_need_pre_suf(need, legal_access_success_pre_suf, num_atom, None,
                               "legal_access_success_pre_suf")

    need = update_need_pre_suf(need, legal_access_fail_pre_suf, num_atom,
                               legal_access_success_pre_suf,
                               "legal_access_fail_pre_suf", generate=True)

    legal_columns = {
        "legal_access_success_pre": legal_access_success_pre,
        "legal_access_fail_pre": legal_access_fail_pre,
        "legal_access_success_pre_suf": legal_access_success_pre_suf,
        "legal_access_fail_pre_suf": legal_access_fail_pre_suf
    }
    need = update_need_not_access(need, legal_columns, num_atom * 2, "legal_not_access_pre")
    legal_columns.pop("legal_access_success_pre")
    legal_columns.pop("legal_access_fail_pre")
    need = update_need_not_access(need, legal_columns, num_atom * 2, "legal_not_access_pre_suf")

    # # 钓鱼
    need = update_need_pre(need, phishing_access_success_pre,
                           num_atom,
                           phishing_access_success_pre_suf,
                           "phishing_access_success_pre")
    need = update_need_pre(need, phishing_access_fail_pre,
                           num_atom,
                           phishing_access_fail_pre_suf,
                           "phishing_access_fail_pre")
    need = update_need_pre_suf(need, phishing_access_success_pre_suf, num_atom, None,
                               "phishing_access_success_pre_suf")
    need = update_need_pre_suf(need, phishing_access_fail_pre_suf, num_atom,
                               "phishing_access_fail_pre_suf", phishing_access_success_pre_suf, generate=True)

    phishing_columns = {
        "phishing_access_success_pre": phishing_access_success_pre,
        "phishing_access_fail_pre": phishing_access_fail_pre,
        "phishing_access_success_pre_suf": phishing_access_success_pre_suf,
        "phishing_access_fail_pre_suf": phishing_access_fail_pre_suf
    }
    need = update_need_not_access(need, phishing_columns, num_atom * 2, "phishing_not_access_pre")
    phishing_columns.pop("phishing_access_success_pre")
    phishing_columns.pop("phishing_access_fail_pre")
    need = update_need_not_access(need, phishing_columns, num_atom * 2, "phishing_not_access_pre_suf")
    need.drop("success", axis=1, inplace=True)
    need.to_csv(f"{save_name}.csv", index=False)
    return need


def update_need_pre(need, pre_data, num_atom, pre_suf_data, pre_name, generate=False):
    if len(pre_data) - num_atom > 0:
        random_selection = pre_data.sample(num_atom, random_state=SEED)
        pre_data.drop(random_selection.index, inplace=True)
        need = need.append(random_selection, ignore_index=True)
    elif generate:
        raise NotImplementedError("未实现该方法")
    else:
        pre_data_len = len(pre_data)
        pre_suf_data_len = len(pre_suf_data)
        if pre_suf_data_len - (num_atom - pre_data_len) > 0:
            random_selection = pre_data.sample(pre_data_len, random_state=SEED)
            pre_data.drop(random_selection.index, inplace=True)
            need = need.append(random_selection)
            random_selection = pre_suf_data.sample((num_atom - pre_data_len), random_state=SEED)
            pre_suf_data.drop(random_selection.index, inplace=True)
            random_selection.loc[:, "url_url_suf"] = "/"
            for index, row in random_selection.iterrows():
                for column_name in row.index:
                    if column_name == "url_url":
                        url_features = URLFeatureExtra(URLFeatureExtra.split_url(row["url_url"])[0] + "/",
                                                       "null").handle()
                        for key, value in url_features.items():
                            if key != "label":
                                random_selection.at[index, key] = value
            need = need.append(random_selection, ignore_index=True)
        else:
            raise ValueError(
                f"需要 {(num_atom - pre_data_len) - pre_suf_data_len} 个{pre_name}_suf或{(num_atom - pre_data_len)}个{pre_name}构成数据集")
    return need


def update_need_pre_suf(need, pre_suf_data, num_atom, contrary_data, suf_pre_name, generate=False):
    len_pre_suf_data = len(pre_suf_data)
    if len_pre_suf_data - num_atom > 0:
        random_selection = pre_suf_data.sample(num_atom, random_state=SEED)
        pre_suf_data.drop(random_selection.index, inplace=True)
        need = need.append(random_selection, ignore_index=True)
    elif generate:
        need_num = num_atom - len_pre_suf_data
        if len(contrary_data) >= need_num:
            random_selection = pre_suf_data.sample(len_pre_suf_data, random_state=SEED)
            pre_suf_data.drop(random_selection.index, inplace=True)
            filtered_columns = random_selection[random_selection.filter(like='page').columns | random_selection.filter(
                like='res').columns | random_selection.filter(like='dyn').columns]
            need = need.append(random_selection)
            random_selection = contrary_data.sample(need_num, random_state=SEED)
            contrary_data.drop(random_selection.index, inplace=True)
            for index, row in random_selection.iterrows():
                for column_name in row.index:
                    if column_name.startswith("page") or column_name.startswith("res") or column_name.startswith("dyn"):
                        random_selection.at[index, column_name] = np.random.choice(filtered_columns[column_name])
            need = need.append(random_selection, ignore_index=True)
        else:
            raise ValueError(f"需要{need_num - len(contrary_data)}个反向数据进行生成")
    else:
        raise ValueError(
            f"需要 {(num_atom - len_pre_suf_data)}个{suf_pre_name}构成数据集")
    return need


def update_need_not_access(need, columns, num_atom, pre_name, generate=False):
    def adjust(the_need, the_random_select, the_pre_name):
        for index, row in the_random_select.iterrows():
            for column_name in row.index:
                if column_name.startswith("page") or column_name.startswith("res") or column_name.startswith("dyn"):
                    the_random_select.at[index, column_name] = 0
                if "pre_suf" not in the_pre_name and column_name == "url_url" and "pre_suf" in key:
                    url_features = URLFeatureExtra(URLFeatureExtra.split_url(row["url_url"])[0] + "/",
                                                   "null").handle()
                    for url_key, url_value in url_features.items():
                        if url_key != "label":
                            the_random_select.at[index, url_key] = url_value
        return the_need.append(the_random_select, ignore_index=True)

    for key, value in columns.items():
        the_len = len(value)
        if the_len - num_atom >= 0:
            random_selection = value.sample(num_atom, random_state=SEED)
            value.drop(random_selection.index, inplace=True)
            need = adjust(need, random_selection, pre_name)
            num_atom = 0
            break
        else:

            random_selection = value.sample(the_len, random_state=SEED)
            value.drop(random_selection.index, inplace=True)
            need = adjust(need, random_selection, pre_name)
            num_atom -= the_len
        if num_atom == 0:
            break
    if num_atom > 0:
        result_string = ''
        for key in columns.keys():
            result_string += key + '或'
        raise ValueError(f"需要{num_atom}个 {result_string.rstrip('或')}")

    return need


SEED = 40

calculate_condition("no_eli_set_features.csv", "no_eli_set_dataset", 50000)
