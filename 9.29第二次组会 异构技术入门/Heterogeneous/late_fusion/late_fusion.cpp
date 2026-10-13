#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using std::cout;
using std::string;
using std::vector;

static float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static float dot(const vector<float>& w, const vector<float>& x) {
    float sum = 0.0f;
    for (size_t i = 0; i < w.size(); ++i) {
        sum += w[i] * x[i];
    }
    return sum;
}

static float randn(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(1e-7f, 1.0f);
    const float a = u(rng);
    const float b = u(rng);
    return std::sqrt(-2.0f * std::log(a)) * std::cos(2.0f * 3.14159265358979323846f * b);
}

static float runif(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    return u(rng);
}

struct Sample {
    float a0 = 0.0f;
    float a1 = 0.0f;
    float e0 = 0.0f;
    float e1 = 0.0f;
    float qa = 0.0f;
    float qe = 0.0f;
    int ma = 1;
    int me = 1;
    int y = 0;
};

static Sample make_sample(
    std::mt19937& rng,
    float miss_a = 0.20f,
    float miss_e = 0.20f,
    float qa_override = -1.0f,
    float qe_override = -1.0f
) {
    Sample s;
    const float la0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float la1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float risk_logit =
        0.8f * la0 + 0.8f * le0 + 2.2f * la0 * le0 +
        0.15f * la1 - 0.10f * le1 - 0.25f;
    s.y = (runif(rng, 0.0f, 1.0f) < sigmoid(risk_logit)) ? 1 : 0;
    s.ma = (runif(rng, 0.0f, 1.0f) > miss_a) ? 1 : 0;
    s.me = (runif(rng, 0.0f, 1.0f) > miss_e) ? 1 : 0;
    s.qa = (qa_override >= 0.0f) ? qa_override : runif(rng, 0.40f, 1.0f);
    s.qe = (qe_override >= 0.0f) ? qe_override : runif(rng, 0.40f, 1.0f);
    if (!s.ma) {
        s.qa = 0.0f;
    }
    if (!s.me) {
        s.qe = 0.0f;
    }
    const float na = 0.10f + 0.70f * (1.0f - s.qa);
    const float ne = 0.10f + 0.70f * (1.0f - s.qe);
    s.a0 = s.ma ? (la0 + na * randn(rng)) : 0.0f;
    s.a1 = s.ma ? (la1 + na * randn(rng)) : 0.0f;
    s.e0 = s.me ? (le0 + ne * randn(rng)) : 0.0f;
    s.e1 = s.me ? (le1 + ne * randn(rng)) : 0.0f;
    return s;
}

//从一个sample中 
//  提取音频侧的 4 个原始特征
//  ，组装成长度为 4 的 float 向量返回
static vector <float> audio_input(const Sample & s)
{
    return
    {
        s.a0, s.a1, static_cast <float> (s.ma), s.qa
    };
}


// 构造环境分支输入特征
static vector<float> env_input(const Sample& s) {
    return {s.e0, s.e1, static_cast<float>(s.me), s.qe};
}

//【与之前代码差异】构造门控网络输入，只用缺失标记和质量分，不使用原始观测特征
static vector<float> gate_input(const Sample& s) {
    return {
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


//【与之前代码差异】晚融合模型：双分支逻辑回归头 + 门控网络，动态计算融合权重
struct LateFusionModel {
    Logistic audio_head{4};
    Logistic env_head{4};
    Logistic gate{4}; //【与之前代码差异】新增门控网络，用来预测动态alpha权重
    float b = 0.0f;  // 融合输出层偏置参数
    float db = 0.0f;  // 融合输出层偏置对应的梯度
    float alpha = 0.5f; //【与之前代码差异】动态融合权重，不再是固定常数
    vector<float> ax;  // 保存音频分支的输入特征，反向传播需要复用
    vector<float> ex;  // 保存环境分支的输入特征，反向传播需要复用
    vector<float> gx;  // 保存门控网络的输入特征，反向传播需要复用
    float z1 = 0.0f; //【与之前代码差异】audio分支输出logit
    float z2 = 0.0f; //【与之前代码差异】env分支输出logit

    //音频、环境两个模态，有独立的处理分支，互不干扰
    //音频先自己提取特征，自己输出预测 。环境也是独立一套流程
    //融合 ： 用gate网络，读取两个模态的状态，动态算出一个aplha权重。
    float forward(const Sample& s) {
        ax = audio_input(s);
        ex = env_input(s);
        gx = gate_input(s);
        z1 = audio_head.forward(ax);
        z2 = env_head.forward(ex);
        //算出模型对音频分支结果的信任程度
        alpha = sigmoid(gate.forward(gx));
        //如果alpha的值比较大，代表对音频更加的信任
        //  ，用音频就更加多一点，权重更大
        return alpha * z1 + (1.0f - alpha) * z2 + b;
    }

    void backward(float grad) {
        //【与之前代码差异】梯度按alpha权重分配给两个模态分支
        audio_head.backward(grad * alpha, ax);
        env_head.backward(grad * (1.0f - alpha), ex);
        //【与之前代码差异】计算gate网络的梯度，更新门控参数
        const float dalpha = grad * (z1 - z2);
        const float dgate_logit = dalpha * alpha * (1.0f - alpha);
        gate.backward(dgate_logit, gx);
        db += grad;
    }

    void zero_grad() {
        audio_head.zero_grad();
        env_head.zero_grad();
        gate.zero_grad();
        db = 0.0f;
    }
    void step(float lr, int batch_size) {
        audio_head.step(lr, batch_size);
        env_head.step(lr, batch_size);
        gate.step(lr, batch_size);
        b -= (lr / static_cast<float>(batch_size)) * db;
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
    float mean_alpha = 0.0f; //【与之前代码差异】新增指标：记录平均动态alpha
};

static Metrics evaluate(LateFusionModel& model, const vector<Sample>& data) {
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    float brier = 0.0f;
    float alpha_sum = 0.0f;
    for (const Sample& s : data) {
        const float p = sigmoid(model.forward(s));
        const int pred = (p >= 0.5f) ? 1 : 0;
        tp += (pred == 1 && s.y == 1);
        fp += (pred == 1 && s.y == 0);
        tn += (pred == 0 && s.y == 0);
        fn += (pred == 0 && s.y == 1);
        brier += (p - static_cast<float>(s.y)) * (p - static_cast<float>(s.y));
        alpha_sum += model.alpha; //【与之前代码差异】累加每个样本的alpha，用于求均值
    }
    const float n = static_cast<float>(data.size());
    Metrics m;
    m.accuracy = static_cast<float>(tp + tn) / n;
    m.precision = (tp + fp > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    m.recall = (tp + fn > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    m.f1 = (m.precision + m.recall > 0.0f)
        ? 2.0f * m.precision * m.recall / (m.precision + m.recall)
        : 0.0f;
    m.brier = brier / n;
    m.mean_alpha = alpha_sum / n;
    return m;
}

static void train(LateFusionModel& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
    std::mt19937 rng(12345);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng);
        for (size_t start = 0; start < data.size(); start += batch_size) {
            const size_t end = std::min(data.size(), start + batch_size);
            model.zero_grad();
            for (size_t i = start; i < end; ++i) {
                const float p = sigmoid(model.forward(data[i]));
                model.backward(p - static_cast<float>(data[i].y));
            }
            model.step(lr, static_cast<int>(end - start));
        }
    }
}


static Sample make_test_sample(std::mt19937& rng, int regime) {
    switch (regime) {
        case 0: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f);
        case 1: return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f);
        case 2: return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f);
        case 3: return make_sample(rng, 0.0f, 0.0f, 0.2f, 0.9f);
        default: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.2f);
    }
}

int main() {
    constexpr int train_size = 4000;
    constexpr int test_size = 1200;
    constexpr int epochs = 200;
    constexpr float lr = 0.04f;
    constexpr int batch_size = 64;
    std::mt19937 rng(20261009);
    vector<Sample> train_data;
    train_data.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        train_data.push_back(make_sample(rng, 0.20f, 0.20f));
    }

    
    vector<vector<Sample>> test_data(5);
    for (int regime = 0; regime < 5; ++regime) {
        test_data[regime].reserve(test_size);
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    
    LateFusionModel model;
    train(model, train_data, epochs, lr, batch_size);
    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };
    cout << "Late fusion: p1=f1(audio), p2=f2(env), p=g(p1,p2)\n";
    cout << "Regime          Acc     Prec    Recall  F1      Brier   alpha\n";
    cout << "-----------------------------------------------------------------\n";
    cout << std::fixed << std::setprecision(4);
    for (int regime = 0; regime < 5; ++regime) {
        const Metrics m = evaluate(model, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.precision
             << std::setw(8) << m.recall
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier
             << std::setw(8) << m.mean_alpha << '\n';
    }
    return 0;
}

