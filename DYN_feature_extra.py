import pandas as pd


def get_before_after_features(state="before"):
    legal_features = pd.concat([
        pd.read_csv("no_eli_set_pilot_process/{}_legal_page_features.csv".format(state)),
        pd.read_csv("no_eli_set_pilot_process/{}_legal_res_features.csv".format(state)),
    ], axis=1)
    phishing_features = pd.concat([
        pd.read_csv("no_eli_set_pilot_process/{}_phishing_page_features.csv".format(state)),
        pd.read_csv("no_eli_set_pilot_process/{}_phishing_res_features.csv".format(state))
    ], axis=1)
    return pd.concat([legal_features, phishing_features])


def dyn_feature_extra():
    before_features = get_before_after_features("before")
    after_features = get_before_after_features("after")
    column_to_ignore = 'res_state_code'
    before_features = before_features.drop(column_to_ignore, axis=1)
    after_features = after_features.drop(column_to_ignore, axis=1)

    diff = after_features.sub(before_features)
    diff = diff.fillna(0)
    new_column_names = ["dyn_" + i for i in diff.columns]
    diff = diff.rename(columns=dict(zip(diff.columns, new_column_names)))
    diff.to_csv("no_eli_set_pilot_process/dyn_features.csv",index=False)


if __name__ == "__main__":
    dyn_feature_extra()
