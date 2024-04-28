# import pandas as pd
# import numpy as np
#
# # # 假设filtered_columns是你的筛选后的列
# # # 请用你实际使用的DataFrame替换df
# #
# # # 示例DataFrame
# # data = {'page_1': [1, 2, 3],
# #         'page_2': [4, 5, 6],
# #         'res_1': [7, 8, 9],
# #         'res_2': [10, 11, 12],
# #         'dyn_1': [13, 14, 15],
# #         'dyn_2': [16, 17, 18]}
# # df = pd.DataFrame(data)
# #
# # # 筛选列
# # filtered_columns = df.filter(like='page').columns | df.filter(like='res').columns | df.filter(like='dyn').columns
# #
# # # 从筛选后的列中随机选择一个列名
# # selected_column_name = np.random.choice(filtered_columns)
# #
# # # 获取选定列的所有值
# # selected_column_values = df[selected_column_name]
# #
# # # 显示结果
# # print(f"Selected Column Name: {selected_column_name}")
# # print("Selected Column Values:")
# # print(selected_column_values)
#
# # 读取 CSV 文件
# data = pd.read_csv('output_feature/features.csv')
#
# # 提取满足条件的行
# filtered_data = data[
#     (data['page_len'] >400) & (data['page_len'] <= 3000) & (data['label'] == 0) & (data['res_state_code'] == 200)]
#
# # 按照 page_len 从大到小排序
# sorted_data = filtered_data.sort_values(by='page_len', ascending=False)
#
# # 显示结果
# print(sorted_data)

for i in range(10603,10993):
    print(i)