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

// 每个模态编码器输出特征维度
static constexpr int H = 8;   // width of each modality encoder
// 双线性池化输出通道数
static constexpr int K = 8;   // bilinear output dimension
// 双线性输出固定缩放系数，用于维持数值范围稳定
static constexpr float POOL_SCALE = 5.0f;

// 双线性池化模式枚举
enum class PoolMode {
    FullBilinear,      // 完整双线性池化
    FactorizedBilinear,// 分解式（低秩）双线性池化
};

/**
 * @brief sigmoid激活函数，防止数值溢出做分支处理
 * @param x 输入logit值
 * @return 输出概率，值域(0,1)
 */
static float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

/**
 * @brief 向量点积计算
 * @param w 权重向量
 * @param x 输入特征向量
 * @return 点积结果 sum(w[i]*x[i])
 */
static float dot(const vector<float>& w, const vector<float>& x) {
    float sum = 0.0f;
    for (size_t i = 0; i < w.size(); ++i) {
        sum += w[i] * x[i];
    }
    return sum;
}

/**
 * @brief Box‑Muller算法生成标准正态分布随机数 N(0,1)
 * @param rng mt19937随机引擎引用
 * @return 正态分布浮点数
 */
static float randn(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(1e-7f, 1.0f);
    const float a = u(rng);
    const float b = u(rng);
    return std::sqrt(-2.0f * std::log(a)) * std::cos(2.0f * 3.14159265358979323846f * b);
}

/**
 * @brief 生成[lo,hi]区间均匀分布随机浮点数
 * @param rng mt19937随机引擎引用
 * @param lo 下限
 * @param hi 上限
 * @return 均匀分布随机值
 */
static float runif(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    return u(rng);
}

/**
 * @brief 单条样本结构体
 */
struct Sample {
    float a0 = 0.0f;    // 音频原始特征0
    float a1 = 0.0f;    // 音频原始特征1
    float e0 = 0.0f;    // 环境原始特征0
    float e1 = 0.0f;    // 环境原始特征1
    float qa = 0.0f;    // 音频质量分，越大质量越好
    float qe = 0.0f;    // 环境质量分，越大质量越好
    int ma = 1;         // 音频是否存在标记：1存在，0丢失
    int me = 1;         // 环境是否存在标记：1存在，0丢失
    int y = 0;          // 样本真实标签 0/1
};

/**
 * @brief 生成合成样本
 * @param rng 随机数引擎
 * @param miss_a 音频丢失概率
 * @param miss_e 环境丢失概率
 * @param qa_override 强制覆盖音频质量分，-1代表不覆盖，随机生成
 * @param qe_override 强制覆盖环境质量分，-1代表不覆盖，随机生成
 * @return 生成好的Sample样本
 * @note 交互项被设置为主导项，正是双线性池化要捕捉的跨模态交互模式
 */
static Sample make_sample(
    std::mt19937& rng,
    float miss_a = 0.20f,
    float miss_e = 0.20f,
    float qa_override = -1.0f,
    float qe_override = -1.0f
) {
    Sample s;
    // 隐式真实特征，裁剪到[-3,3]
    const float la0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float la1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le1 = std::clamp(randn(rng), -3.0f, 3.0f);

    // The interaction is deliberately the dominant term. It is the pattern
    // that bilinear pooling is designed to capture.
    // 风险对数几率，跨模态交互项la0*le0系数最大，作为核心信号
    const float risk_logit =
        0.5f * la0 + 0.5f * le0 + 2.8f * la0 * le0 +
        0.10f * la1 - 0.08f * le1 - 0.25f;
    // 根据logit采样得到0/1标签
    s.y = (runif(rng, 0.0f, 1.0f) < sigmoid(risk_logit)) ? 1 : 0;

    // 随机决定该样本音频、环境模态是否丢失
    s.ma = (runif(rng, 0.0f, 1.0f) > miss_a) ? 1 : 0;
    s.me = (runif(rng, 0.0f, 1.0f) > miss_e) ? 1 : 0;

    // 设置质量分，支持外部强制覆盖
    s.qa = (qa_override >= 0.0f) ? qa_override : runif(rng, 0.40f, 1.0f);
    s.qe = (qe_override >= 0.0f) ? qe_override : runif(rng, 0.40f, 1.0f);

    // 如果模态丢失，则质量分置0
    if (!s.ma) {
        s.qa = 0.0f;
    }
    if (!s.me) {
        s.qe = 0.0f;
    }

    // 噪声强度：质量分越低，噪声越大
    const float na = 0.10f + 0.70f * (1.0f - s.qa);
    const float ne = 0.10f + 0.70f * (1.0f - s.qe);

    // 模态存在：隐特征加噪声得到观测特征；丢失则特征置0
    s.a0 = s.ma ? (la0 + na * randn(rng)) : 0.0f;
    s.a1 = s.ma ? (la1 + na * randn(rng)) : 0.0f;
    s.e0 = s.me ? (le0 + ne * randn(rng)) : 0.0f;
    s.e1 = s.me ? (le1 + ne * randn(rng)) : 0.0f;
    return s;
}

/**
 * @brief 组装音频分支输入特征向量
 * @param s 样本
 * @return 向量 {a0,a1,ma,qa}
 */
static vector<float> audio_input(const Sample& s) {
    return {s.a0, s.a1, static_cast<float>(s.ma), s.qa};
}

/**
 * @brief 组装环境分支输入特征向量
 * @param s 样本
 * @return 向量 {e0,e1,me,qe}
 */
static vector<float> env_input(const Sample& s) {
    return {s.e0, s.e1, static_cast<float>(s.me), s.qe};
}

/**
 * @brief 线性层 + tanh激活 模块
 * 实现：y = tanh(Wx + b)，自带前向、反向、梯度清零、SGD更新
 */
struct LinearTanh {
    int in = 0;                 // 输入维度
    int out = 0;                // 输出维度
    vector<float> w;             // 权重参数 w[out * in]
    vector<float> dw;            // 权重梯度，反向传播累加
    vector<float> b;             // 偏置参数
    vector<float> db;            // 偏置梯度，反向传播累加
    vector<float> x;             // 缓存本次前向输入，反向要用
    vector<float> h;             // 缓存tanh输出激活，反向要用

    /**
     * @brief 构造函数，初始化权重、偏置、梯度
     * @param input_dim 输入维度
     * @param output_dim 输出维度
     * @param seed 随机初始化种子
     */
    LinearTanh(int input_dim, int output_dim, int seed)
        : in(input_dim), out(output_dim) {
        std::mt19937 rng(seed);
        // Xavier初始化界限，防止激活饱和
        const float limit = 1.0f / std::sqrt(static_cast<float>(in));
        std::uniform_real_distribution<float> init(-limit, limit);
        w.resize(in * out);
        dw.assign(w.size(), 0.0f);
        b.assign(out, 0.0f);
        db.assign(out, 0.0f);
        for (float& value : w) {
            value = init(rng);
        }
    }

    /**
     * @brief 前向传播：线性变换 + tanh激活
     * @param input 输入向量
     * @return tanh激活后的输出向量
     */
    vector<float> forward(const vector<float>& input) {
        x = input;
        h.assign(out, 0.0f);
        for (int o = 0; o < out; ++o) {
            float z = b[o];
            for (int i = 0; i < in; ++i) {
                z += w[o * in + i] * x[i];
            }
            h[o] = std::tanh(z);
        }
        return h;
    }

    /**
     * @brief 反向传播，链式求导，累加梯度
     * @param dh dLoss/dh，损失对本层输出的上游梯度
     * @return dx dLoss/dx，损失对本层输入的梯度，传给上一层
     */
    vector<float> backward(const vector<float>& dh) {
        vector<float> dx(in, 0.0f);
        for (int o = 0; o < out; ++o) {
            // tanh导数 dz = dh * (1‑h²)
            const float dz = dh[o] * (1.0f - h[o] * h[o]);
            for (int i = 0; i < in; ++i) {
                dw[o * in + i] += dz * x[i];
                dx[i] += w[o * in + i] * dz;
            }
            db[o] += dz;
        }
        return dx;
    }

    /**
     * @brief 梯度清零，每个batch开始调用
     */
    void zero_grad() {
        std::fill(dw.begin(), dw.end(), 0.0f);
        std::fill(db.begin(), db.end(), 0.0f);
    }

    /**
     * @brief SGD参数更新：w -= lr/batch * dw
     * @param lr 学习率
     * @param batch_size 当前批次样本数，求梯度均值
     */
    void step(float lr, int batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] -= scale * dw[i];
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] -= scale * db[i];
        }
    }
};

/**
 * @brief 手写逻辑回归，线性分类头 y = w·x + b
 */
struct Logistic {
    vector<float> w;    // 权重
    vector<float> dw;   // 权重梯度
    float b = 0.0f;     // 偏置
    float db = 0.0f;    // 偏置梯度

    explicit Logistic(int n) : w(n, 0.0f), dw(n, 0.0f) {}

    /**
     * @brief 前向，计算logit
     * @param x 输入特征向量
     * @return w·x + b
     */
    float forward(const vector<float>& x) const {
        return dot(w, x) + b;
    }

    /**
     * @brief 反向传播，累加梯度
     * @param grad dLoss/dLogit = p‑y
     * @param x 当前样本输入特征
     */
    void backward(float grad, const vector<float>& x) {
        for (size_t i = 0; i < w.size(); ++i) {
            dw[i] += grad * x[i];
        }
        db += grad;
    }

    /**
     * @brief 梯度清零，每个batch开始调用
     */
    void zero_grad() {
        std::fill(dw.begin(), dw.end(), 0.0f);
        db = 0.0f;
    }

    /**
     * @brief SGD参数更新
     * @param lr 学习率
     * @param batch_size 当前batch样本数
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
 * @brief 双线性融合模型：支持完整双线性 / 分解低秩双线性两种模式
 * 核心：显式建模两个模态特征的两两交互；并且根据模态质量分做抑制
 */
struct BilinearFusionModel {
    PoolMode mode = PoolMode::FullBilinear; // 选择双线性模式
    LinearTanh audio_encoder{4, H, 101};     // 音频编码器：输入4维输出H维，种子101
    LinearTanh env_encoder{4, H, 202};      // 环境编码器：输入4维输出H维，种子202
    Logistic head{K};                        // 输出分类头，输入K维双线性特征

    // Full bilinear weights: K * H * H.
    // 完整双线性权重：每个输出通道k，H×H的权重矩阵 W_k
    vector<float> wb;
    vector<float> dwb; // wb对应的梯度

    // Factorized bilinear weights: K * H each.
    // 分解双线性：U、V，低秩近似，参数量远小于完整双线性
    vector<float> u;
    vector<float> v;
    vector<float> du;
    vector<float> dv;

    // Linear unimodal terms: z_k = bilinear + A_k^T a + B_k^T e.
    // 单模态线性项，双线性交互之外，额外保留音频、环境单独线性贡献
    vector<float> va;
    vector<float> ve;
    vector<float> dva;
    vector<float> dve;

    // 缓存中间变量，供反向传播复用
    vector<float> ax;
    vector<float> ex;
    vector<float> ha;   // 音频编码器输出
    vector<float> he;   // 环境编码器输出
    vector<float> a;    // ha * qa，质量加权之后音频特征
    vector<float> e;    // he * qe，质量加权之后环境特征
    vector<float> z;    // 经过双线性+线性项之后K维特征
    float qa = 0.0f;    // 当前样本音频质量分缓存
    float qe = 0.0f;    // 当前样本环境质量分缓存

    /**
     * @brief 构造函数，按模式分配参数并随机初始化
     * @param pool_mode 双线性模式：完整 / 分解低秩
     * @param seed 参数初始化随机种子
     */
    BilinearFusionModel(PoolMode pool_mode, int seed) : mode(pool_mode) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> init(-0.25f, 0.25f);
        std::uniform_real_distribution<float> init_factor(-0.80f, 0.80f);

        wb.assign(K * H * H, 0.0f);
        dwb.assign(wb.size(), 0.0f);
        u.assign(K * H, 0.0f);
        v.assign(K * H, 0.0f);
        du.assign(u.size(), 0.0f);
        dv.assign(v.size(), 0.0f);
        va.assign(K * H, 0.0f);
        ve.assign(K * H, 0.0f);
        dva.assign(va.size(), 0.0f);
        dve.assign(ve.size(), 0.0f);

        for (float& value : wb) {
            value = init(rng);
        }
        for (float& value : u) value = init_factor(rng);
        for (float& value : v) value = init_factor(rng);
        for (float& value : va) value = init(rng);
        for (float& value : ve) value = init(rng);
    }

    /**
     * @brief 统计模型总参数量，区分两种双线性模式
     * @return 全部可训练参数总个数
     */
    int parameter_count() const {
        const int encoder = H * 4 + 4 * H + H + H;
        const int pooling = (mode == PoolMode::FullBilinear) ? (K * H * H) : (2 * K * H);
        const int linear_terms = 2 * K * H;
        const int classifier = K + 1;
        return encoder + pooling + linear_terms + classifier;
    }

    /**
     * @brief 模型前向传播
     * @param s 输入样本
     * @return 分类头输出logit（未经过sigmoid）
     * @note Quality‑aware scaling: a missing or unreliable modality is suppressed
     * before the bilinear interaction is computed.
     * 质量感知缩放：在计算双线性交互之前，丢失/不可靠模态会被质量分抑制
     */
    float forward(const Sample& s) {
        ax = audio_input(s);
        ex = env_input(s);
        ha = audio_encoder.forward(ax);
        he = env_encoder.forward(ex);

        // 获取有效质量分；模态丢失则qa/qe强制为0
        qa = s.ma ? s.qa : 0.0f;
        qe = s.me ? s.qe : 0.0f;

        // 质量加权：质量差的模态特征整体被缩小
        a.resize(H);
        e.resize(H);
        for (int i = 0; i < H; ++i) {
            a[i] = ha[i] * qa;
            e[i] = he[i] * qe;
        }

        z.assign(K, 0.0f);
        if (mode == PoolMode::FullBilinear) {
            // z_k = sum_ij W_kij * a_i * e_j
            // 完整双线性：每个输出通道k，计算a、e所有维度两两乘积加权求和
            for (int k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < H; ++j) {
                        sum += wb[(k * H + i) * H + j] * a[i] * e[j];
                    }
                }
                z[k] = sum;
            }
        } else {
            // z_k = sum_i (U_ki * a_i) * (V_ki * e_i)
            // 分解（低秩）双线性，只做逐元素相乘求和，参数量大幅下降
            for (int k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (int i = 0; i < H; ++i) {
                    sum += (u[k * H + i] * a[i]) * (v[k * H + i] * e[i]);
                }
                z[k] = sum;
            }
        }

        // 叠加单模态线性项，补充非交互部分信号
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < H; ++i) {
                z[k] += va[k * H + i] * a[i] + ve[k * H + i] * e[i];
            }
        }

        // A fixed scale keeps the low‑rank branch in a useful numeric range.
        // 固定缩放系数，保证低秩分支输出数值稳定
        for (int k = 0; k < K; ++k) {
            z[k] *= POOL_SCALE;
        }

        // 送入逻辑回归分类头，输出logit
        return head.forward(z);
    }

    /**
     * @brief 反向传播，从分类头上游梯度回传，累加全部参数梯度
     * @param grad dLoss/dLogit = p‑y，损失对最终logit的导数
     */
    void backward(float grad) {
        vector<float> dz(K, 0.0f);
        // 1.分类头head反向
        for (int k = 0; k < K; ++k) {
            head.dw[k] += grad * z[k];
            dz[k] = grad * head.w[k] * POOL_SCALE;
        }
        head.db += grad;

        vector<float> da(H, 0.0f);
        vector<float> de(H, 0.0f);

        // 2.双线性部分反向，区分完整双线性 / 分解双线性
        if (mode == PoolMode::FullBilinear) {
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < H; ++j) {
                        const int idx = (k * H + i) * H + j;
                        dwb[idx] += dz[k] * a[i] * e[j];
                        da[i] += dz[k] * wb[idx] * e[j];
                        de[j] += dz[k] * wb[idx] * a[i];
                    }
                }
            }
        } else {
            for (int k = 0; k < K; ++k) {
                for (int i = 0; i < H; ++i) {
                    const int idx = k * H + i;
                    const float p = u[idx] * a[i];
                    const float q = v[idx] * e[i];
                    const float dp = dz[k] * q;
                    const float dq = dz[k] * p;
                    du[idx] += dp * a[i];
                    dv[idx] += dq * e[i];
                    da[i] += dp * u[idx];
                    de[i] += dq * v[idx];
                }
            }
        }

        // 3.单模态线性项 va ve 的反向传播
        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < H; ++i) {
                dva[k * H + i] += dz[k] * a[i];
                dve[k * H + i] += dz[k] * e[i];
                da[i] += dz[k] * va[k * H + i];
                de[i] += dz[k] * ve[k * H + i];
            }
        }

        // 4.回推质量加权这一步的梯度，再送入编码器反向
        vector<float> dha(H, 0.0f);
        vector<float> dhe(H, 0.0f);
        for (int i = 0; i < H; ++i) {
            dha[i] = da[i] * qa;
            dhe[i] = de[i] * qe;
        }
        audio_encoder.backward(dha);
        env_encoder.backward(dhe);
    }

    /**
     * @brief 全部模块梯度清零，每个batch开始调用
     */
    void zero_grad() {
        audio_encoder.zero_grad();
        env_encoder.zero_grad();
        head.zero_grad();
        std::fill(dwb.begin(), dwb.end(), 0.0f);
        std::fill(du.begin(), du.end(), 0.0f);
        std::fill(dv.begin(), dv.end(), 0.0f);
        std::fill(dva.begin(), dva.end(), 0.0f);
        std::fill(dve.begin(), dve.end(), 0.0f);
    }

    /**
     * @brief SGD更新全部参数
     * @param lr 学习率
     * @param batch_size 当前批次样本数量，求梯度均值
     */
    void step(float lr, int batch_size) {
        audio_encoder.step(lr, batch_size);
        env_encoder.step(lr, batch_size);
        head.step(lr, batch_size);
        const float scale = lr / static_cast<float>(batch_size);
        for (size_t i = 0; i < wb.size(); ++i) {
            wb[i] -= scale * dwb[i];
        }
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] -= scale * du[i];
            v[i] -= scale * dv[i];
            va[i] -= scale * dva[i];
            ve[i] -= scale * dve[i];
        }
    }
};

/**
 * @brief 评估指标结构体：准确率、F1分数、Brier分数（概率校准损失）
 */
struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};

/**
 * @brief 在数据集上做评估，计算分类指标
 * @param model 待评估双线性融合模型
 * @param data 测试数据集
 * @return Metrics 准确率、F1、Brier
 */
static Metrics evaluate(BilinearFusionModel& model, const vector<Sample>& data) {
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    float brier = 0.0f;
    for (const Sample& s : data) {
        const float p = sigmoid(model.forward(s));
        const int pred = (p >= 0.5f) ? 1 : 0;
        tp += (pred == 1 && s.y == 1);
        fp += (pred == 1 && s.y == 0);
        tn += (pred == 0 && s.y == 0);
        fn += (pred == 0 && s.y == 1);
        brier += (p - static_cast<float>(s.y)) * (p - static_cast<float>(s.y));
    }
    const float n = static_cast<float>(data.size());
    Metrics m;
    m.accuracy = static_cast<float>(tp + tn) / n;
    const float precision = (tp + fp > 0)
        ? static_cast<float>(tp) / static_cast<float>(tp + fp)
        : 0.0f;
    const float recall = (tp + fn > 0)
        ? static_cast<float>(tp) / static_cast<float>(tp + fn)
        : 0.0f;
    m.f1 = (precision + recall > 0.0f)
        ? 2.0f * precision * recall / (precision + recall)
        : 0.0f;
    m.brier = brier / n;
    return m;
}

/**
 * @brief Mini‑batch SGD训练循环
 * @param model 双线性融合模型
 * @param data 训练数据集
 * @param epochs 训练轮数
 * @param lr 学习率
 * @param batch_size mini‑batch大小
 */
static void train(BilinearFusionModel& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
    std::mt19937 rng(12345);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(data.begin(), data.end(), rng); // epoch开始打乱样本
        for (size_t start = 0; start < data.size(); start += batch_size) {
            const size_t end = std::min(data.size(), start + batch_size);
            model.zero_grad(); // batch开始清空梯度
            for (size_t i = start; i < end; ++i) {
                const float p = sigmoid(model.forward(data[i]));
                const float grad = p - static_cast<float>(data[i].y);
                model.backward(grad); // 累加本batch所有样本梯度
            }
            model.step(lr, static_cast<int>(end - start)); // 更新参数
        }
    }
}

/**
 * @brief 生成测试集样本，5种固定测试场景
 * @param rng 随机引擎
 * @param regime 场景编号
 * @return 对应测试条件的样本
 * 0:complete双模态完好；1:missing‑audio音频丢失；2:missing‑env环境丢失；
 * 3:noisy‑audio音频低质量；4:noisy‑env环境低质量
 */
static Sample make_test_sample(std::mt19937& rng, int regime) {
    switch (regime) {
        case 0: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f);
        case 1: return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f);
        case 2: return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f);
        case 3: return make_sample(rng, 0.0f, 0.0f, 0.2f, 0.9f);
        default: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.2f);
    }
}

/**
 * @brief main函数：对比完整双线性 vs 分解低秩双线性模型
 * 相同训练数据、相同初始化种子，只改变双线性模式，对比参数量与5种场景指标
 */
int main() {
    constexpr int train_size = 3000;
    constexpr int test_size = 1000;
    constexpr int epochs = 160;
    constexpr float lr = 0.04f;
    constexpr int batch_size = 64;
    std::mt19937 rng(20261011);

    // 生成训练数据集
    vector<Sample> base_train;
    base_train.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        base_train.push_back(make_sample(rng, 0.20f, 0.20f));
    }

    // 5种测试场景数据集
    vector<vector<Sample>> test_data(5);
    for (int regime = 0; regime < 5; ++regime) {
        test_data[regime].reserve(test_size);
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    // 构造两个模型：完整双线性、分解低秩双线性，相同初始化种子777
    BilinearFusionModel full(PoolMode::FullBilinear, 777);
    BilinearFusionModel factorized(PoolMode::FactorizedBilinear, 777);

    // 两份完全一样的训练集，避免原始base_train被shuffle互相干扰
    vector<Sample> train_full = base_train;
    vector<Sample> train_factorized = base_train;

    // 分别训练
    train(full, train_full, epochs, lr, batch_size);
    train(factorized, train_factorized, epochs, lr, batch_size);

    // 5个测试场景名称
    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };

    // 打印两个模型参数量
    cout << "Full bilinear params:       " << full.parameter_count() << '\n';
    cout << "Factorized bilinear params: " << factorized.parameter_count() << "\n\n";

    // 循环跑两组模型，打印5场景评估表格
    for (const auto& item : vector<std::pair<string, BilinearFusionModel*>>{
             {"Full bilinear: z_k = a^T W_k e", &full},
             {"Factorized bilinear: z_k = sum_i (U_ki a_i)(V_ki e_i)", &factorized},
         }) {
        cout << item.first << '\n';
        cout << "Regime          Acc     F1      Brier\n";
        cout << "----------------------------------------\n";
        cout << std::fixed << std::setprecision(4);
        for (int regime = 0; regime < 5; ++regime) {
            const Metrics m = evaluate(*item.second, test_data[regime]);
            cout << std::left << std::setw(16) << names[regime]
                 << std::setw(8) << m.accuracy
                 << std::setw(8) << m.f1
                 << std::setw(8) << m.brier << '\n';
        }
        cout << '\n';
    }
    return 0;
}

