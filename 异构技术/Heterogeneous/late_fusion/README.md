# 介绍

- 融合方式 ：晚期融合
- 代码语言 ： cpp
- 特征 ：晚融合：音频、环境两个分支各自独立算出预测结果，再由门控网络根据模态缺失 / 质量状态动态加权融合。
# 编译运行
g++ -O2 -std=c++17 -Wall -Wextra -pedantic late_fusion.cpp -o late_fusion
./late_fusion