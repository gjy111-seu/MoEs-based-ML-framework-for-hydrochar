import pandas as pd
import numpy as np

# 读取Excel数据（假设列名为 HHV, EY, T, RT, SL）
df = pd.read_excel("Data file path", sheet_name='Pareto Front')


target_columns = ["HHV", "EY"]
data = df[target_columns].values

# Customized ideal solution and worst solution
ideal_point = np.array([31.3, 92.7])
worst_point = np.array([15.96, 48.14])


norms = np.sqrt(np.sum(data**2, axis=0))
normalized_data = data / norms
normalized_ideal = ideal_point / norms
normalized_worst = worst_point / norms


weights = np.array([0.5, 0.5])


d_plus = np.sqrt(np.sum(weights * (normalized_data - normalized_ideal)**2, axis=1))
d_minus = np.sqrt(np.sum(weights * (normalized_data - normalized_worst)**2, axis=1))
closeness = d_minus / (d_plus + d_minus)


df["Closeness"] = closeness

df_sorted = df.sort_values(by="Closeness", ascending=False)

df_sorted.to_excel("save path", index=False)


optimal_row = df_sorted.iloc[0]
print("Detailed information of the optimal solution：")
print(f"HHV: {optimal_row['HHV']}, EY: {optimal_row['EY']}")
print(f"operating parameters T: {optimal_row['T']}, RT: {optimal_row['RT']}, SL: {optimal_row['SL']}")
print(f"Closeness: {optimal_row['Closeness'].round(4)}")