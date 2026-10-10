# 介绍
- 特征级融合示例代码
- 语言 ： cpp
- 特征 ：特征级融合：音频、环境各自经过独立编码器提取隐特征，再将两组特征拼接后送入共享MLP分类头做预测。
# 编译运行
g++ -O2 -std=c++17 -Wall -Wextra -pedantic feature_fusion.cpp -o feature_fusion
./feature_fusion
