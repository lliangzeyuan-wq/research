# 介绍
- 融合方式 ： 早期融合
- 语言 ： cpp
- 特点 ： 早期融合：把音频、环境所有特征在输入端直接拼成长向量，再用一个单层分类器做预测。
# 编译运行

g++ -O2 -std=c++17 -Wall -Wextra -pedantic early_fusion.cpp -o early_fusion
./early_fusion