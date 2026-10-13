# 介绍
融合方式 ： 混合融合
编程语言 ： cpp
特征 ： 把早融合、特征融合、晚融合三种方式全部**并行**跑一遍，最后用 combiner 综合所有路径的结果。
# 编译运行
g++ -O2 -std=c++17 -Wall -Wextra -pedantic hybrid_fusion.cpp -o hybrid_fusion
./hybrid_fusion

