// Gated fusion demo: learn a gate that decides how much each modality contributes.
//
// Reference: Gated Multimodal Units for Information Fusion, Arevalo et al., 2017
//            https://arxiv.org/abs/1702.01992
//
// Fusion rule (audio a, environment e, per dimension j):
//   h_a = tanh(W_a x_a)                   audio representation
//   h_e = tanh(W_e x_e)                   environment representation
//   z   = sigmoid(W_g [x_a; x_e])         gate in (0, 1), learned from the inputs
//   h   = z (*) h_a + (1 - z) (*) h_e     gated fusion
//   p   = sigmoid(W_o h + b)              risk probability
//
// Because the gate reads the modal values, the missing flags, and the quality
// scores, it can lower the weight of a modality that is absent or unreliable.
// The same encoders also feed a fixed equal-weight fusion (z = 0.5) so the two
// fusion rules can be compared with everything else held constant.
//
// Build: g++ -O2 -std=c++17 -Wall -Wextra -pedantic gate_fusion.cpp -o gate_fusion

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

static constexpr int H = 8;         // 输出纬度
static constexpr int DA = 4;        // 音频输入：{a0, a1, mask_a, quality_a}
static constexpr int DE = 4;        // 环境输入：{e0, e1, mask_e, quality_e}
static constexpr int DG = DA + DE;  // 门控模块输入：两个模态的全部特征 + 各自的质量信息

static float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
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

// Same synthetic task as the other demos: the risk is driven by a cross-modal
// interaction, so neither modality alone is sufficient.
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

static vector<float> audio_input(const Sample& s) {
    return {s.a0, s.a1, static_cast<float>(s.ma), s.qa};
}

static vector<float> env_input(const Sample& s) {
    return {s.e0, s.e1, static_cast<float>(s.me), s.qe};
}




// =====================================================================
// 门控融合网络（核心模型）
// 结构：两个模态各自独立编码器 + 一个门控网络 + 门控融合 + 输出层
//   h_a = tanh(W_a x_a)                 音频编码
//   h_e = tanh(W_e x_e)                 环境编码
//   z   = sigmoid(W_g [x_a; x_e])       门控值 ∈ (0,1)，决定每个模态贡献多少
//   h   = z·h_a + (1-z)·h_e             门控融合（核心公式）
//   logit = W_o·h + b,  p = sigmoid(logit)
//
// gated 布尔值是两组实验的唯一变量：
//   true  → 用可学习门控 z = sigmoid(W_g [x_a;x_e])
//   false → 固定等权 z = 0.5（两个模态各占一半）
// 其余编码器、输出层、初始化方式完全相同，纯粹对比"融合规则"的差别。
// =====================================================================
struct FusionNet {
    bool gated = true;                    // 是否启用可学习门控（false=固定等权）

    // ---- 权重参数（4组）----
    vector<float> wa, we, wg, wo;       // weights
    // wa: 音频编码器权重  H×DA
    // we: 环境编码器权重  H×DE
    // wg: 门控网络权重    H×DG（读两模态拼接输入）
    // wo: 输出层权重      H

    // ---- 梯度（与权重一一对应，反向传播时累加）----
    vector<float> dwa, dwe, dwg, dwo;   // gradients

    // ---- 偏置（4组，每组H个，输出层偏置是标量）----
    float ba[H]{}, be[H]{}, bg[H]{}, bo = 0.0f;
    // ba: 音频编码器偏置  be: 环境编码器偏置
    // bg: 门控网络偏置    bo: 输出层偏置

    // ---- 偏置的梯度 ----
    float dba[H]{}, dbe[H]{}, dbg[H]{}, dbo = 0.0f;

    // ---- 前向传播时缓存的中间激活值，反向传播要用 ----
    vector<float> ax, ex, gx, ha, he, z, hh;
    // ax: 音频输入  ex: 环境输入  gx: 拼接输入(门控用)
    // ha: 音频编码  he: 环境编码  z: 门控值  hh: 融合结果
    float logit = 0.0f;                  // 最终线性输出

    // =================================================================
    // 构造函数：分配所有数组空间，权重做小随机初始化
    // use_gate: 是否用可学习门控
    // rng: 随机数生成器（固定种子 → 两组模型初始化完全一致）
    // =================================================================
    explicit FusionNet(bool use_gate, std::mt19937& rng) : gated(use_gate) {
        // 分配权重数组空间并初始化为0
        wa.assign(H * DA, 0.0f);
        we.assign(H * DE, 0.0f);
        wg.assign(H * DG, 0.0f);
        wo.assign(H, 0.0f);
        // 分配梯度数组空间并初始化为0
        dwa.assign(H * DA, 0.0f);
        dwe.assign(H * DE, 0.0f);
        dwg.assign(H * DG, 0.0f);
        dwo.assign(H, 0.0f);
        // 分配中间激活值数组空间并初始化为0
        ha.assign(H, 0.0f);
        he.assign(H, 0.0f);
        z.assign(H, 0.0f);
        hh.assign(H, 0.0f);
        // 所有权重做 [-0.5, 0.5] 小随机初始化，打破对称性（全0初始化会导致梯度对称，学不动）
        std::uniform_real_distribution<float> small(-0.5f, 0.5f);
        for (float& w : wa) w = small(rng);
        for (float& w : we) w = small(rng);
        for (float& w : wg) w = small(rng);
        for (float& w : wo) w = small(rng);
    }

    // =================================================================
    // 前向传播：输入一条样本，输出 logit（未过sigmoid）
    // 流程：提取两模态输入 → 各自tanh编码 → 算门控z → 门控融合 → 输出层
    // =================================================================
    float forward(const Sample& s) {
        ax = audio_input(s);      // 音频输入：{a0, a1, 丢失标记ma, 质量分qa}
        ex = env_input(s);        // 环境输入：{e0, e1, 丢失标记me, 质量分qe}
        gx = ax;                  // 门控输入 = 两模态拼接 [x_a; x_e]
        gx.insert(gx.end(), ex.begin(), ex.end());
        // 逐维度（共H维）计算编码、门控、融合
        for (int j = 0; j < H; ++j) {
            // --- 音频编码器：线性组合 + tanh ---
            float za = ba[j];
            for (int i = 0; i < DA; ++i) za += wa[j * DA + i] * ax[i];
            ha[j] = std::tanh(za);
            // --- 环境编码器：线性组合 + tanh ---
            float ze = be[j];
            for (int i = 0; i < DE; ++i) ze += we[j * DE + i] * ex[i];
            he[j] = std::tanh(ze);
            // --- 门控：gated=true时可学习，false时固定0.5 ---
            if (gated) {
                float zg = bg[j];
                for (int i = 0; i < DG; ++i) zg += wg[j * DG + i] * gx[i];
                z[j] = sigmoid(zg);          // 门控值 ∈ (0,1)
            } else {
                z[j] = 0.5f;  // fixed equal-weight fusion
            }
            // --- 门控融合核心公式：按z加权两个模态 ---
            hh[j] = z[j] * ha[j] + (1.0f - z[j]) * he[j];
        }
        // --- 输出层：融合结果 → logit ---
        logit = bo;
        for (int j = 0; j < H; ++j) logit += wo[j] * hh[j];
        return logit;
    }

    // =================================================================
    // 反向传播：grad = dL/dlogit = p - y（交叉熵+sigmoid合并求导的结果）
    // 链式法则从输出层一路回传到编码器和门控：
    //   输出层  → dwo, dbo
    //   融合    → dho = grad·wo
    //   门控    → dz = dho·(h_a - h_e)，再乘sigmoid导数 z(1-z)
    //   编码器  → dha = dho·z, dhe = dho·(1-z)，再乘tanh导数 1-tanh²
    // 所有梯度用 += 累加（一个batch内多个样本梯度求和）
    // =================================================================
    // grad = dL/dlogit
    void backward(float grad) {
        for (int j = 0; j < H; ++j) {
            // 输出层梯度：dL/dwo = grad·h
            dwo[j] += grad * hh[j];
            // dL/dh（融合结果的梯度），链式回传
            const float dho = grad * wo[j];
            // 融合公式 h = z·h_a + (1-z)·h_e 对 h_a/h_e 求导
            const float dha = dho * z[j];      // dL/dh_a（经z加权）
            const float dhe = dho * (1.0f - z[j]);  // dL/dh_e
            // 门控自身梯度（仅gated=true时需要更新门控权重）
            if (gated) {
                // 融合对z求导：dh/dz = h_a - h_e
                const float dz = dho * (ha[j] - he[j]);
                // sigmoid导数：dz/d(pre) = z(1-z)
                const float dpre = dz * z[j] * (1.0f - z[j]);
                // 门控权重梯度
                for (int i = 0; i < DG; ++i) dwg[j * DG + i] += dpre * gx[i];
                dbg[j] += dpre;
            }
            // 音频编码器梯度：tanh导数 = 1 - tanh²
            const float dpa = dha * (1.0f - ha[j] * ha[j]);
            for (int i = 0; i < DA; ++i) dwa[j * DA + i] += dpa * ax[i];
            dba[j] += dpa;
            // 环境编码器梯度：tanh导数 = 1 - tanh²
            const float dpe = dhe * (1.0f - he[j] * he[j]);
            for (int i = 0; i < DE; ++i) dwe[j * DE + i] += dpe * ex[i];
            dbe[j] += dpe;
        }
        // 输出层偏置梯度
        dbo += grad;
    }

    // =================================================================
    // 梯度清零：把所有梯度数组置0
    // 每个batch开始前调用，防止上一个batch的梯度残留叠加
    // =================================================================
    void zero_grad() {
        std::fill(dwa.begin(), dwa.end(), 0.0f);
        std::fill(dwe.begin(), dwe.end(), 0.0f);
        std::fill(dwg.begin(), dwg.end(), 0.0f);
        std::fill(dwo.begin(), dwo.end(), 0.0f);
        std::fill(dba, dba + H, 0.0f);
        std::fill(dbe, dbe + H, 0.0f);
        std::fill(dbg, dbg + H, 0.0f);
        dbo = 0.0f;
    }

    // =================================================================
    // SGD参数更新：w -= lr/batch_size · dw
    // 梯度在backward中是累加的（batch内所有样本梯度之和），
    // 这里除以batch_size求平均梯度，再乘学习率更新参数。
    // =================================================================
    void step(float lr, int batch_size) {
        const float scale = lr / static_cast<float>(batch_size);  // 学习率/样本数
        // 更新4组权重
        for (size_t i = 0; i < wa.size(); ++i) wa[i] -= scale * dwa[i];
        for (size_t i = 0; i < we.size(); ++i) we[i] -= scale * dwe[i];
        for (size_t i = 0; i < wg.size(); ++i) wg[i] -= scale * dwg[i];
        for (size_t i = 0; i < wo.size(); ++i) wo[i] -= scale * dwo[i];
        // 更新3组偏置
        for (int j = 0; j < H; ++j) {
            ba[j] -= scale * dba[j];
            be[j] -= scale * dbe[j];
            bg[j] -= scale * dbg[j];
        }
        // 更新输出层偏置
        bo -= scale * dbo;
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
    float mean_gate = 0.0f;  // average weight given to the audio branch
};

static Metrics evaluate(FusionNet& model, const vector<Sample>& data) {
    int tp = 0, fp = 0, tn = 0, fn = 0;
    float brier = 0.0f;
    float gate_sum = 0.0f;

    for (const Sample& s : data) {
        const float p = sigmoid(model.forward(s));
        const int pred = (p >= 0.5f) ? 1 : 0;
        tp += (pred == 1 && s.y == 1);
        fp += (pred == 1 && s.y == 0);
        tn += (pred == 0 && s.y == 0);
        fn += (pred == 0 && s.y == 1);
        brier += (p - static_cast<float>(s.y)) * (p - static_cast<float>(s.y));
        for (int j = 0; j < H; ++j) gate_sum += model.z[j];
    }

    const float n = static_cast<float>(data.size());
    Metrics m;
    m.accuracy = static_cast<float>(tp + tn) / n;
    const float precision = (tp + fp > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    const float recall = (tp + fn > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    m.f1 = (precision + recall > 0.0f) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
    m.brier = brier / n;
    m.mean_gate = gate_sum / (n * static_cast<float>(H));
    return m;
}

static void train(FusionNet& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
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

static Sample make_regime_sample(std::mt19937& rng, int regime) {
    switch (regime) {
        case 0: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f);   // complete
        case 1: return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f);  // audio missing
        case 2: return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f);  // env missing
        case 3: return make_sample(rng, 0.0f, 0.0f, 0.2f, 0.9f);   // audio noisy
        default: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.2f);  // env noisy
    }
}

// =====================================================================
// 单次实验入口：跑一种融合规则（gated 或 fixed），输出5种场景的评估指标
// 参数：
//   title      本次实验的标题（打印用）
//   gated      true=可学习门控；false=固定等权 z=0.5（唯一变量）
//   test_data  5种测试场景的数据（main里生成好传进来，两组实验共用同一份）
// 设计要点：除了 gated 之外，训练数据、初始化种子、超参数全部固定，
//           保证两组实验的差异只来自"融合规则"本身。
// =====================================================================
static void run(const string& title, bool gated, const vector<vector<Sample>>& test_data) {
    // ---- 训练超参数（两组实验完全相同）----
    constexpr int train_size = 4000;     // 训练集样本数
    constexpr int epochs = 200;          // 训练轮数
    constexpr float lr = 0.04f;          // 学习率
    constexpr int batch_size = 64;       // mini-batch 大小

    // ---- 生成训练集：固定种子，两组实验数据完全一致 ----
    std::mt19937 data_rng(20261009);
    vector<Sample> train_data;
    train_data.reserve(train_size);      // 预分配内存，优化 push_back 性能
    for (int i = 0; i < train_size; ++i) {
        // 训练集：随机20%丢音频、20%丢环境，质量分随机
        train_data.push_back(make_sample(data_rng, 0.20f, 0.20f));
    }

    // ---- 构建并训练模型：固定初始化种子，两组模型初始权重完全一致 ----
    std::mt19937 init_rng(777);
    FusionNet model(gated, init_rng);    // gated 决定用门控还是固定等权
    train(model, train_data, epochs, lr, batch_size);  // 完整训练

    // ---- 5种测试场景的名称（与 test_data 下标一一对应）----
    const vector<string> names = {
        "complete", "missing-audio", "missing-env", "noisy-audio", "noisy-env",
    };

    // ---- 打印表头 ----
    cout << title << '\n';
    cout << "Regime          Acc     F1      Brier   gate(audio)\n";
    cout << "-----------------------------------------------------\n";
    cout << std::fixed << std::setprecision(4);   // 所有浮点数保留4位小数

    // ---- 逐个场景评估并打印指标 ----
    for (int regime = 0; regime < 5; ++regime) {
        const Metrics m = evaluate(model, test_data[regime]);  // 在该场景数据上评估
        cout << std::left << std::setw(16) << names[regime]    // 场景名，左对齐占16格
             << std::setw(8) << m.accuracy                     // 准确率
             << std::setw(8) << m.f1                           // F1分数
             << std::setw(8) << m.brier                        // Brier分数（概率校准误差）
             << std::setw(8) << m.mean_gate << '\n';           // 门控平均给音频的权重
    }
    cout << '\n';
} // end run


int main() {
    constexpr int test_size = 1200;

    std::mt19937 rng(20261009);
    vector<vector<Sample>> test_data(5);
    for (int regime = 0; regime < 5; ++regime) {
        test_data[regime].reserve(test_size);
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_regime_sample(rng, regime));
        }
    }

    cout << "Gated fusion vs fixed equal-weight fusion (same encoders)\n\n";
    run("Fixed equal-weight fusion: h = 0.5*h_a + 0.5*h_e", false, test_data);
    run("Gated fusion: z = sigmoid(W_g [x_a; x_e]), h = z*h_a + (1-z)*h_e", true, test_data);
    return 0;
}
