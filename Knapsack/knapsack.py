# coding: UTF-8

######################## PHASE 0 ########################
## パッケージのインポート

from pyqubo import Array, LogEncInteger, Constraint, solve_qubo
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import numpy as np
import math

np.random.seed(seed=0)

######################## PHASE 1 ########################
## 使用する定数や変数の設定

# 定数設定
N_BAG = 10 # 荷物の個数
W_MAX = 50 # 最大荷重

# 変数の初期化
x = Array.create('x', shape = N_BAG, vartype='BINARY')
y = LogEncInteger('y', (0, W_MAX)) #Logエンコードしたスラック変数

# 価値と荷重の設定
c = np.random.randint(1, 15+1, 10) # 1〜25の値を取る1x10の配列
w = np.random.randint(1, 25+1, 10) # 1〜15の値を取る1x10の配列

######################## PHASE 2 ########################
## ハミルトニアンの記述

# パラメータ設定
B = 1
A = 10 * B *  np.max(c)

# 制約項
HA = Constraint( (W_MAX - sum(w[i]*x[i] for i in range(N_BAG)) - y)**2, label = 'HA')

# 目的項
HB = sum(c[i]*x[i] for i in range(N_BAG))

# ハミルトニアン
H = -B*HB + A*HA

######################## PHASE 2 ########################
## Leapを使って解く

# モデルをコンパイル
model = H.compile()

# QUBOを作成
qubo, offset = model.to_qubo()

# 使用するアカウント、マシン、サンプラーの設定
token = 'DEV-a73514868ffb1b8a8d6dbb6ebecbc6583dbd0016'
endpoint = 'https://cloud.dwavesys.com/sapi'
solver = 'Advantage_system1.1'

sampler = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token, solver=solver))

# 解を求める
raw_solution = sampler.sample_qubo(qubo, num_reads=700)

######################## PHASE 3 ########################
## 解を可視化する

# 求めた解をデコード
decoded_sample = model.decode_sample(raw_solution.first.sample, vartype="BINARY")

# 荷物の選択を格納する配列
#b = np.empty(N_BAG, dtype='int8')
b = np.array([decoded_sample.array('x', k) for k in range(N_BAG)])
#for k in range(N_BAG):
#  b[k] = decoded_sample.array('x', k)

# 価値、荷重、荷物の選択を表示
print("c: {}".format(c))
print("w: {}".format(w))
print("b: {}".format(b))
print()

# 価値の合計、荷重の合計を表示
print("価値の合計: {}".format(np.dot(c, b)))
print("荷重の合計: {}".format(np.dot(w, b)))
print()

# スラック変数の値を表示
slack = sum(2**k * v for k, v in [(elem, decoded_sample.array('y', elem)) for elem in range(math.ceil(math.log2(W_MAX)))])
print("スラック変数y: {}".format(slack))
print()

# 制約を満たしているかチェック
print(decoded_sample.constraints(only_broken=True))
