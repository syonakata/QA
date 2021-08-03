# coding: UTF-8

from gurobipy import *
import numpy as np

np.random.seed(seed=0)

# 問題を設定
model = Model(name="Knapsack")

# 定数設定
N_BAG = 10 # 荷物の個数
W_MAX = 50 # 最大荷重

# 変数を設定
x = {}
for i in range(N_BAG):
    x[i] = model.addVar(vtype="B", name="x[%s]"%(i))

# 価値と荷重の設定
c = np.random.randint(1, 15+1, 10) # 1〜25の値を取る1x10の配列
w = np.random.randint(1, 25+1, 10) # 1〜15の値を取る1x10の配列

model.update()

# 目的関数を設定
model.setObjective(quicksum(c[i]*x[i] for i in range(N_BAG)), sense=GRB.MAXIMIZE)

# 制約を設定
model.addConstr(quicksum(w[i]*x[i] for i in range(N_BAG)) <= W_MAX, name="const")

# 解を求める計算
model.optimize()

print("\n[Gurobi Optimizerログ]\n")

print("[解]")
b = np.empty(N_BAG, dtype='int8')
if model.Status == GRB.OPTIMAL:
    for i in range(N_BAG):
        b[i] = 1 if x[i].X > 0.98 else 0

# 価値、荷重、荷物の選択を表示
print("c: {}".format(c))
print("w: {}".format(w))
print("b: {}".format(b))
print()

print("価値合計: {}".format(np.dot(c, b)))
print("荷重合計: {}".format(np.dot(w, b)))
print("計算時間: %f sec." % model.Runtime)
