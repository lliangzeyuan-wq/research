#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// 简化命名，不用每次写std::
using std::cout;
using std::string;
using std::vector;

/**
 * @brief sigmoid激活函数，把任意实数映射到(0,1)，用来表示概率
 *        做了分支处理防止数值溢出
 * @param x 输入的浮点数
 * @return 0~1之间的概率值 
 对应的公式 ：(x>=0) 1  / (1+e^-x)
              (x<0)  e^x / (1+e^x)
 */
static float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    // x是负数时，改用这个写法，避免exp(-x)爆炸
    const float z = std::exp(x);
    return z / (1.0f + z);
}

/**
 * @brief 向量点积 w · x，就是对应位置相乘再全部加起来
 * @param w 权重向量
 * @param x 特征向量
 * @return 点积结果
 */
static float dot(const vector<float>& w, const vector<float>& x) {
    float sum = 0.0f;
    for (size_t i = 0; i < w.size(); ++i) {
        sum += w[i] * x[i];
    }
    return sum;
}

/**
 * @brief Box-Muller算法，生成标准正态分布随机数（均值0，方差1）
 * @param rng 随机数引擎
 * @return 正态随机数
 */
static float randn(std::mt19937& rng) {
    // 下限1e-7，避免log(0)报错
    //`uniform_real_distribution`：均匀分布，在区间内每个数字机会均等
    //- 区间：`1e-7f ~ 1.0f`（0.0000001 到 1.0）
    std::uniform_real_distribution<float> u(1e-7f, 1.0f);
    const float a = u(rng);
    const float b = u(rng);
    return std::sqrt(-2.0f * std::log(a)) * std::cos(2.0f * 3.14159265358979323846f * b);
}

/**
 * @brief 生成[lo, hi]之间均匀分布的随机小数
 * @param rng 随机引擎
 * @param lo 最小值
 * @param hi 最大值
 * @return 区间内随机浮点数
 */
static float runif(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    return u(rng);
}

/**
 * @brief 一条样本数据结构
 * 两个模态：audio音频，env环境
 */
struct Sample {
    float a0 = 0.0f;   // 观测到的音频特征0
    float a1 = 0.0f;   // 观测到的音频特征1
    float e0 = 0.0f;   // 观测到的环境特征0
    float e1 = 0.0f;   // 观测到的环境特征1

    float qa = 0.0f;   // 音频模态质量：0最差，1最好
    float qe = 0.0f;   // 环境模态质量：0最差，1最好

    int ma = 1;        // ma=1：音频存在；ma=0：音频缺失
    int me = 1;        // me=1：环境存在；me=0：环境缺失

    //有风险（y=1）：听到异响、摔倒、异常噪音。
    //没有风险（y=0）：正常安静环境、正常说话，一切平安。
    int y = 0;         // 真实标签：1=有风险，0=无风险
};

/**
 * @brief 生成一条人造样本
 * @param rng 随机数发生器
 * @param miss_a 音频缺失概率
 * @param miss_e 环境缺失概率
 * @param qa_override 强制指定音频质量，-1代表不强制，随机生成
 * @param qe_override 强制指定环境质量，-1代表不强制，随机生成
 * @return 生成好的Sample样本
 */
static Sample make_sample(
    std::mt19937& rng,
    float miss_a = 0.20f,  // 音频丢失概率，默认20%
    float miss_e = 0.20f,   // 环境丢失概率，默认20%
    float qa_override = -1.0f,  // 强制指定音频质量，-1代表不强制，随机生成
    float qe_override = -1.0f  // 强制指定环境质量，-1代表不强制，随机生成
) {
    Sample s;

    // la0 la1 le0 le1：底层真实隐特征（模型永远看不到！）
    // clamp限制范围-3~3，防止数值爆炸
    //std::clamp(value, low, high) :
    //   如果 `value < low` → 返回 low
    //   如果 `value > high` → 返回 high
    //   如果在中间，直接返回 value
    //`randn(rng)`：调用刚才 Box-Muller 函数，生成**标准正态随机数**（均值 0 方差 1）
    const float la0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float la1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le1 = std::clamp(randn(rng), -3.0f, 3.0f);

    // =================核心公式================
    // 真实风险由隐变量算出，la0*le0是跨模态交互项（最重要）
    // 风险不是音频、环境简单相加，两者相乘共同决定风险
    //     算出来的值可以是任意实数，正数代表风险高，负数风险低
    const float risk_logit =
        0.8f * la0 + 0.8f * le0 + 2.2f * la0 * le0 +
        0.15f * la1 - 0.10f * le1 - 0.25f;

    // 根据logit得到标签y，按sigmoid概率随机取0或1
    // runif(rng, 0.0f, 1.0f : 随机生成 0~1 之间的小数
    //按 sigmoid 输出的概率，抛一次有偏硬币，随机生成标签 y
    s.y = (runif(rng, 0.0f, 1.0f) < sigmoid(risk_logit)) ? 1 : 0;

    // 随机决定当前样本的音频/环境是否缺失
    s.ma = (runif(rng, 0.0f, 1.0f) > miss_a) ? 1 : 0;
    s.me = (runif(rng, 0.0f, 1.0f) > miss_e) ? 1 : 0;

    // 设置模态质量，可以强制覆盖，不覆盖就在0.4~1之间随机
    s.qa = (qa_override >= 0.0f) ? qa_override : runif(rng, 0.40f, 1.0f);
    s.qe = (qe_override >= 0.0f) ? qe_override : runif(rng, 0.40f, 1.0f);

    // 如果模态缺失，质量直接置0
    if (!s.ma) {
        s.qa = 0.0f;
    }
    if (!s.me) {
        s.qe = 0.0f;
    }

    // 噪声系数：质量越差，噪声越大
    const float na = 0.10f + 0.70f * (1.0f - s.qa);
    const float ne = 0.10f + 0.70f * (1.0f - s.qe);

    // 能观测到的特征 = 真实隐特征 + 噪声；模态缺失则直接填0
    //     观测特征 = 真实隐特征 + 噪声 × 正态随机数
    s.a0 = s.ma ? (la0 + na * randn(rng)) : 0.0f;
    s.a1 = s.ma ? (la1 + na * randn(rng)) : 0.0f;
    s.e0 = s.me ? (le0 + ne * randn(rng)) : 0.0f;
    s.e1 = s.me ? (le1 + ne * randn(rng)) : 0.0f;

    return s;
}

/**
 * @brief Early Fusion早期融合：把多模态所有特征拼接成一维向量
 * @param s 一条样本
 * @return 拼接后的9维特征向量
 */
static vector<float> early_features(const Sample& s) {
    // 9个特征：a0,a1,e0,e1,交互乘积项,ma,qa,me,qe
    return {
        s.a0,
        s.a1,
        s.e0,
        s.e1,
        s.a0 * s.e0,  // 手动构造跨模态交互特征
        static_cast<float>(s.ma),
        s.qa,
        static_cast<float>(s.me),
        s.qe,
    };
}

/**
 * @brief 手写逻辑回归（线性分类器）
 */
struct Logistic {
    vector<float> w;   // 权重参数
    vector<float> dw;  // 权重的梯度，用来存反向求导结果
    float b = 0.0f;    // 偏置
    float db = 0.0f;   // 偏置的梯度

    // 构造函数：n个特征，初始化权重和梯度数组为0
    //explicit : 禁止隐式转换
    explicit Logistic(int n) : w(n, 0.0f), dw(n, 0.0f) {}

    /**
     * @brief 前向计算：w·x + b，得到logit
     */
    float forward(const vector<float>& x) const {
        return dot(w, x) + b;
    }

    /**
     * @brief 反向传播，累加梯度
     * @param grad dLoss/dLogit = p - y
     * @param x 当前样本特征
     */
    void backward(float grad, const vector<float>& x) {
        for (size_t i = 0; i < w.size(); ++i) {
            dw[i] += grad * x[i];
        }
        db += grad;
    }

    // 梯度清零，每个batch开始前调用
    void zero_grad() {
        std::fill(dw.begin(), dw.end(), 0.0f);
        db = 0.0f;
    }

    /**
     * @brief 参数更新：SGD梯度下降
     * @param lr 学习率
     * @param batch_size 当前批次样本数量，梯度要除以样本数求平均
     */
    void step(float lr, int batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] -= scale * dw[i];
        }
        b -= scale * db;
    }
};

/**
 * @brief 早期融合模型包装类
 */
struct EarlyFusionModel {
    Logistic head{9};   // 逻辑回归头，输入9维特征
    vector<float> x;    // 缓存当前样本提取后的特征

    // 前向：提取特征，交给逻辑回归计算logit
    float forward(const Sample& s) {
        x = early_features(s);
        return head.forward(x);
    }

    // 反向传播，把梯度传给底层逻辑回归
    void backward(float grad) {
        head.backward(grad, x);
    }

    void zero_grad() {
        head.zero_grad();
    }

    void step(float lr, int batch_size) {
        head.step(lr, batch_size);
    }
};

/**
 * @brief 模型评估指标结构体
 */
struct Metrics {
    float accuracy = 0.0f;   // 准确率
    float precision = 0.0f;  // 精确率
    float recall = 0.0f;      // 召回率
    float f1 = 0.0f;         // F1分数，精确率召回率调和平均
    float brier = 0.0f;      // Brier分数，评估概率好坏，越小越好
};

/**
 * @brief 在测试数据集上跑模型，计算各项指标
 * @param model 训练好的模型
 * @param data 测试样本集合
 * @return 指标结果Metrics
 */
static Metrics evaluate(EarlyFusionModel& model, const vector<Sample>& data) {
    int tp = 0; // 真阳性：预测1，真实1 （猜对的危险样本）
    int fp = 0; // 假阳性：预测1，真实0
    int tn = 0; // 真阴性：预测0，真实0  （猜对的安全样本）
    int fn = 0; // 假阴性：预测0，真实1
    float brier = 0.0f;

    for (const Sample& s : data) {
        const float p = sigmoid(model.forward(s));
        const int pred = (p >= 0.5f) ? 1 : 0; // 0.5作为分类阈值

        tp += (pred == 1 && s.y == 1); //真，真 ： 模型预测为1 并且 有风险的真实性为1
        fp += (pred == 1 && s.y == 0);
        tn += (pred == 0 && s.y == 0);
        fn += (pred == 0 && s.y == 1);

        // Brier平方误差累加   ：模型输出的概率值准不准。
        //公式 :   Brier =  (求和符号 （pi - yi）^2)
        brier += (p - static_cast<float>(s.y)) * (p - static_cast<float>(s.y));
    }

    const float n = static_cast<float>(data.size());
    Metrics m;
    // 准确率 ： 猜对的样本/总样本
    m.accuracy = static_cast<float>(tp + tn) / n;  
    //精确率 ： 猜对的有风险 / （猜对的有风险+猜错的有风险）
    m.precision = (tp + fp > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    //召回率 ： 有风险且猜对 / （有风险但没有猜对 +  有风险且猜对）
    m.recall = (tp + fn > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    // F1分数，精确率召回率调和平均
    m.f1 = (m.precision + m.recall > 0.0f)
        ? 2.0f * m.precision * m.recall / (m.precision + m.recall)
        : 0.0f;
    // 前面算的Brier平方误差累加 / n :  衡量的是模型输出的概率准不准
    m.brier = brier / n;

    return m;
}

/**
 * @brief 模型训练函数，小批量SGD
 * @param model 模型
 * @param data 训练数据集 ：4000个
 * @param epochs 迭代轮数 : 200
 * @param lr 学习率 : 0.04f
 * @param batch_size 批次大小 : 64(每一次读64个)
 */
static void train(EarlyFusionModel& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
    std::mt19937 rng(12345); // 训练打乱用的随机种子
    //200轮，每一轮都打乱顺序
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 每一轮epoch，打乱训练样本顺序
        std::shuffle(data.begin(), data.end(), rng);

        // 按batch_size切分样本
        for (size_t start = 0; start < data.size(); start += batch_size) {
            const size_t end = std::min(data.size(), start + batch_size);
            model.zero_grad(); // 每个批次先清空梯度

            // 遍历本batch所有样本，前向+累加梯度
            for (size_t i = start; i < end; ++i) {
                const float p = sigmoid(model.forward(data[i]));
                // 交叉熵损失对logit的导数就是 p-y
                model.backward(p - static_cast<float>(data[i].y));
            }
            // 使用这个批次的梯度更新参数
            model.step(lr, static_cast<int>(end - start));
        }
    }
}

/**
 * @brief 生成测试样本，不同regime代表不同测试场景
 * @param rng 随机引擎
 * @param regime 场景编号
 * @return 样本
 */
static Sample make_test_sample(std::mt19937& rng, int regime) {
    switch (regime) {
        case 0: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f); // 模态齐全高质量
        case 1: return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f); // 音频全部丢失
        case 2: return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f); // 环境全部丢失
        case 3: return make_sample(rng, 0.0f, 0.0f, 0.2f, 0.9f); // 音频噪声很大
        default: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.2f); // 环境噪声很大
    }
}

int main() {


    // 超参数设置

    // constexpr : 编译的时候这个数字就固定死了，运行的时候不能修改

    
    //用来训练 AI 的数据一共有 4000 条。
    //AI 反复看这 4000 条，学会怎么判断有没有风险。
    constexpr int train_size = 4000;    // 训练集样本数量
    //后面我们要测 5 种场景（音频丢了、环境噪声大等等）
    //，**每一种场景单独准备 1200 条新数据**，用来检验 AI 学得好不好。
    //这些数据 AI 训练的时候没见过。
    constexpr int test_size = 1200;     // 每个测试场景样本数量
    //一轮 = 把全部 4000 条训练数据从头到尾完整看一遍。
    //AI 把这 4000 条数据反复完整读 200 遍，慢慢调整自己的判断规则。
    constexpr int epochs = 200;         // 训练轮数
    //AI 改错时，每次调整参数的幅度。0.04 就是每次小幅度修改。
    constexpr float lr = 0.04f;         // 学习率
    //AI 不是一次性拿 4000 条一起学，而是分批：一次拿 64 条，算完误差，再更新自己。
    //4000 ÷ 64，每一轮里面会分成很多小批次。
    constexpr int batch_size = 64;      // 批次大小

    //固定随机数种子，每次生成的随机数一样
    std::mt19937 rng(20261009); // 固定随机种子，保证每次运行结果一样

    // 1. 生成训练数据集
    vector<Sample> train_data;
    //reserve(n) : 提前向内存申请一块足够大的空间
    train_data.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        // 训练集：音频20%丢失，环境20%丢失
        train_data.push_back(make_sample(rng, 0.20f, 0.20f));
    }

    // 2. 生成5组测试数据集，对应5种场景
    vector<vector<Sample>> test_data(5);
    for (int regime = 0; regime < 5; ++regime) {
        test_data[regime].reserve(test_size);
        //5种情况 ， 每种情况有1200个数据 ， 并且都是同类的。
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    // 3. 创建模型并开始训练
    EarlyFusionModel model;
    train(model, train_data, epochs, lr, batch_size);

    // 场景名字
    const vector<string> names = {
        "complete",  //模态齐全高质量
        "missing-audio", // 音频全部丢失
        "missing-env", //环境全部丢失
        "noisy-audio", // 音频噪声很大
        "noisy-env", // //环境噪声很大
    };

    // 打印表头
    cout << "Early fusion: concatenate audio and environment features first\n";
    cout << "Regime          Acc     Prec    Recall  F1      Brier\n";
    cout << "-------------------------------------------------------\n";
    cout << std::fixed << std::setprecision(4); // 设置输出小数保留4位

    // 在5个测试集上评估，输出结果
    for (int regime = 0; regime < 5; ++regime) {
        const Metrics m = evaluate(model, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.precision
             << std::setw(8) << m.recall
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier << '\n';
    }

    return 0;
}

