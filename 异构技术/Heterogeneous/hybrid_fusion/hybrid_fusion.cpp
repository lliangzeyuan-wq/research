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
static constexpr int H = 8;

//cpp语法 ： enum class 解决了传统的enum的 命名的冲突（调用的
//-- 时候必须用Act::）和 隐形的数值转换。用法上和enum大体上一样
//【特异性】激活函数枚举：可配置每层用什么激活
enum class Act {
    //公式: y=x
    Linear,   // 无激活，直接输出线性结果 
    //公式: y=(exp(x)-exp(-x)) / (exp(x)+exp(-x))
    Tanh,     // Tanh激活，输出(-1,1)
    //公式: y=max(0,x)
    ReLU,     // ReLU激活，负数截断为0
};

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

//【特异性】样本数据结构：比之前多了时间对齐和数据年龄字段
struct Sample {
    // ===== 音频模态观测特征 =====
    float a0 = 0.0f;        // 观测到的音频特征0（受噪声影响）
    float a1 = 0.0f;        // 观测到的音频特征1（受噪声影响）

    // ===== 环境模态观测特征 =====
    float e0 = 0.0f;        // 观测到的环境特征0（受噪声影响）
    float e1 = 0.0f;        // 观测到的环境特征1（受噪声影响）

    // ===== 模态质量评分 =====
    float qa = 0.0f;        // 音频模态质量：0最差（噪声大/缺失），1最好（清晰）
    float qe = 0.0f;        // 环境模态质量：0最差，1最好

    // ===== 【新增】时间对齐与数据年龄 =====
    float dt = 0.0f;        // 两个模态采集的时间偏差（音频和环境不是同一时刻采的）
    float age_a = 0.0f;     // 音频数据距今的时间（数据有多"旧"，越旧可信度越低）
    float age_e = 0.0f;     // 环境数据距今的时间（环境数据的"年龄"）

    // ===== 模态缺失标记 =====
    int ma = 1;             // ma=1：音频数据存在；ma=0：音频数据缺失（直接填0）
    int me = 1;             // me=1：环境数据存在；me=0：环境数据缺失

    // ===== 真实标签 =====
    int y = 0;              // 分类标签：1=有风险（异响/摔倒/异常），0=无风险（正常安静）
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
    //【特异性】生成时间对齐和数据年龄噪声
    s.dt = std::clamp(randn(rng) * 0.5f, -2.0f, 2.0f);
    s.age_a = std::abs(randn(rng)) * 0.2f;
    s.age_e = std::abs(randn(rng)) * 1.5f;
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

//【特异性】早融合路径输入：11维，包含所有原始特征+时间对齐+数据年龄+缺失标记+质量分
static vector<float> early_input(const Sample& s) {
    return {
        s.a0,                      // 音频特征0
        s.a1,                      // 音频特征1
        s.e0,                      // 环境特征0
        s.e1,                      // 环境特征1
        s.dt,                      // 时间对齐偏差
        s.age_a,                   // 音频数据年龄
        s.age_e,                   // 环境数据年龄
        static_cast<float>(s.ma),  // 音频缺失标记
        s.qa,                      // 音频质量分
        static_cast<float>(s.me),  // 环境缺失标记
        s.qe,                      // 环境质量分
    };
}

//【特异性】层配置描述符：用来描述一个层的"规格说明书"，不存数据，只存配置
struct LayerSpec {
    int in = 0;             // 输入维度：这一层接收几个特征数字
    int out = 0;           // 输出维度：这一层输出几个特征数字
    Act act = Act::Linear;  // 激活函数类型：Linear(无激活)/Tanh/ReLU
};

//【特异性】通用神经网络层：可配置激活函数，支持前向+反向传播
struct Layer {

    // ===== 层规格 ===== 对应LayerSpec
    int in = 0;                 // 输入维度：这一层接收几个特征数字
    int out = 0;                // 输出维度：这一层输出几个特征数字
    Act act = Act::Linear;      // 这一层用什么激活函数（Linear/Tanh/ReLU）

    // ===== 可训练参数（要学习的权重） =====
    vector<float> w;            // 权重矩阵：一维平铺存储，大小 = out × in
                                // w[o * in + i] 表示第o个输出对应第i个输入的权重
    vector<float> dw;           // 权重w对应的梯度，反向传播时累加，用来更新w
    vector<float> b;            // 偏置向量，维度out，每个输出神经元一个偏置
    vector<float> db;           // 偏置b对应的梯度

    // ===== 前向传播缓存（反向传播要用） =====
    vector<float> x;            // 缓存本次前向的输入向量（backward时算dw要用）
    vector<float> z;            // 缓存线性预激活值 z = W·x + b（过激活函数之前的原始结果）
    vector<float> h;            // 缓存激活输出 h = act(z)（过激活函数之后的最终结果）


    //【特异性】Layer构造函数：根据层配置spec初始化权重和偏置
    Layer(const LayerSpec& spec, std::mt19937& rng)
        : in(spec.in), out(spec.out), act(spec.act) {  // 初始化层规格：输入维度、输出维度、激活函数
        
        // Xavier初始化：限制权重范围 = 1/sqrt(输入维度)
        // 防止权重太大导致激活值爆炸，或太小导致梯度消失
        const float limit = 1.0f / std::sqrt(static_cast<float>(in));
        
        // 在[-limit, limit]均匀分布内随机采样初始化权重
        std::uniform_real_distribution<float> init(-limit, limit);
        
        // 权重数组大小 = 输出数 × 输入数（二维矩阵平铺成一维）
        w.resize(in * out);
        
        // 梯度数组初始化为0（反向传播时累加）
        dw.assign(w.size(), 0.0f);
        
        // 偏置向量初始化为0（每个输出神经元一个偏置）
        b.assign(out, 0.0f);
        
        // 偏置梯度初始化为0
        db.assign(out, 0.0f);
        
        // 遍历所有权重，用随机分布采样赋值
        for (float& value : w) {
            value = init(rng);
        }
    }


    vector<float> forward(const vector<float>& input) {
        x = input;
        z.assign(out, 0.0f);
        h.assign(out, 0.0f);
        for (int o = 0; o < out; ++o) {
            float sum = b[o];
            for (int i = 0; i < in; ++i) {
                sum += w[o * in + i] * x[i];
            }
            z[o] = sum;
            //【特异性】根据配置选择不同激活函数
            if (act == Act::Tanh) {
                h[o] = std::tanh(sum);
            } else if (act == Act::ReLU) {
                h[o] = std::max(0.0f, sum);
            } else {
                h[o] = sum;   // Linear：无激活
            }
        }
        return h;
    }

    vector<float> backward(const vector<float>& dh) {
        vector<float> dx(in, 0.0f);
        vector<float> delta(out, 0.0f);
        for (int o = 0; o < out; ++o) {
            float d = dh[o];
            //【特异性】根据激活函数选择对应导数
            if (act == Act::Tanh) {
                d *= (1.0f - h[o] * h[o]);        // tanh导数：1-h²
            } else if (act == Act::ReLU) {
                d *= (z[o] > 0.0f ? 1.0f : 0.0f); // ReLU导数：正=1，负=0
            }
            delta[o] = d;
            for (int i = 0; i < in; ++i) {
                dw[o * in + i] += d * x[i];
                dx[i] += w[o * in + i] * d;
            }
            db[o] += d;
        }
        return dx;
    }

    void zero_grad() {
        std::fill(dw.begin(), dw.end(), 0.0f);
        std::fill(db.begin(), db.end(), 0.0f);
    }
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

//【特异性】通用MLP：多层Layer堆叠，可任意配置深度和激活函数组合
// MLP : 多层网络容器
struct MLP {
    vector<Layer> layers;

    MLP(const vector<LayerSpec>& specs, int seed) {
        std::mt19937 rng(seed);
        layers.reserve(specs.size());
        for (const LayerSpec& spec : specs) {
            // 每个配置创建一个Layer，放进数组（就是一个添加一个层的操作）
            layers.emplace_back(spec, rng); 
        }
    }

    // 前向：从第一层到最后一层顺序计算
    vector<float> forward(const vector<float>& input) {
        vector<float> y = input;
        for (Layer& layer : layers) {
            y = layer.forward(y);
        }
        return y;
    }

    // 反向：从最后一层往回逆序传播梯度
    vector<float> backward(const vector<float>& grad_output) {
        vector<float> grad = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = it->backward(grad);
        }
        return grad;
    }

    void zero_grad() {
        for (Layer& layer : layers) {
            layer.zero_grad();
        }
    }
    void step(float lr, int batch_size) {
        for (Layer& layer : layers) {
            layer.step(lr, batch_size);
        }
    }
};

//【特异性】混合融合模型：同时跑早融合、特征融合、晚融合三条路径，最后用combiner做最终决策
struct HybridFusionModel {
    // ===== 路径1：早融合 =====
    // 11维全特征 → Tanh编码 → 8维共享特征h_early
    MLP early_encoder{
        {{11, H, Act::Tanh}},
        101
    };

    // ===== 路径2：特征融合 =====
    // 音频编码器：4维→8维
    MLP audio_encoder{
        {{4, H, Act::Tanh}},
        202
    };
    // 环境编码器：4维→8维
    MLP env_encoder{
        {{4, H, Act::Tanh}},
        303
    };
    // 融合编码器：h1(8)+h2(8)拼接 → 16维 → Tanh → 8维h_feature
    MLP feature_encoder{
        {{2 * H, H, Act::Tanh}},
        404
    };

    // ===== 路径3：晚融合 =====
    // 音频独立预测：4维→1维logit
    MLP audio_head{
        {{4, 1, Act::Linear}},
        505
    };
    // 环境独立预测：4维→1维logit
    MLP env_head{
        {{4, 1, Act::Linear}},
        606
    };

    // ===== 最终组合器combiner =====
    // 输入 = [p1, p2, h_early(8), h_feature(8), ma, qa, me, qe] = 2+16+4 = 22维
    // 两层MLP：22维 → 16维(ReLU) → 1维logit
    MLP combiner{
        {
            {2 + 2 * H + 4, 16, Act::ReLU},
            {16, 1, Act::Linear},
        },
        707
    };

    // 缓存中间结果，反向传播复用
    vector<float> ax;
    vector<float> ex;
    vector<float> early_x;
    vector<float> h_early;      // 早融合路径输出(8维)
    vector<float> h1;           // 音频编码器输出(8维)
    vector<float> h2;           // 环境编码器输出(8维)
    vector<float> h_feature;    // 特征融合路径输出(8维)
    vector<float> combined;      // 拼给combiner的组合向量
    float p1 = 0.5f;            // 晚融合：音频独立预测概率
    float p2 = 0.5f;            // 晚融合：环境独立预测概率

    float forward(const Sample& s) {
        ax = audio_input(s);
        ex = env_input(s);
        early_x = early_input(s);

        // 路径1：早融合 —— 所有特征拼一起，编码成h_early
        h_early = early_encoder.forward(early_x);

        // 路径2：特征融合 —— 双编码器提取隐特征，再融合成h_feature
        h1 = audio_encoder.forward(ax);
        h2 = env_encoder.forward(ex);
        vector<float> h12 = h1;
        h12.insert(h12.end(), h2.begin(), h2.end());  // 拼接h1+h2
        h_feature = feature_encoder.forward(h12);     // 融合编码

        // 路径3：晚融合 —— 两个模态独立预测各自概率
        const float z1 = audio_head.forward(ax)[0];
        const float z2 = env_head.forward(ex)[0];
        p1 = sigmoid(z1);
        p2 = sigmoid(z2);

        //【特异性】三条路径输出全部拼在一起，送入combiner做最终决策
        combined = {p1, p2};
        combined.insert(combined.end(), h_early.begin(), h_early.end());
        combined.insert(combined.end(), h_feature.begin(), h_feature.end());
        combined.push_back(static_cast<float>(s.ma));
        combined.push_back(s.qa);
        combined.push_back(static_cast<float>(s.me));
        combined.push_back(s.qe);

        return combiner.forward(combined)[0];
    }

    void backward(float grad) {
        // 先从combiner反向，切出各部分的梯度
        const vector<float> dcombined = combiner.backward({grad});
        const float dp1 = dcombined[0];
        const float dp2 = dcombined[1];
        const vector<float> dh_early(          // 第2~9位：h_early的梯度
            dcombined.begin() + 2,
            dcombined.begin() + 2 + H
        );
        const vector<float> dh_feature(        // 第10~17位：h_feature的梯度
            dcombined.begin() + 2 + H,
            dcombined.begin() + 2 + 2 * H
        );

        // 晚融合路径：过sigmoid导数，回传给两个独立分支
        const float dz1 = dp1 * p1 * (1.0f - p1);
        const float dz2 = dp2 * p2 * (1.0f - p2);
        audio_head.backward({dz1});
        env_head.backward({dz2});

        // 早融合路径：梯度回传early_encoder
        early_encoder.backward(dh_early);

        // 特征融合路径：先回传feature_encoder，再拆成两路回传给audio/env编码器
        const vector<float> dh12 = feature_encoder.backward(dh_feature);
        const vector<float> dh1(dh12.begin(), dh12.begin() + H);
        const vector<float> dh2(dh12.begin() + H, dh12.end());
        audio_encoder.backward(dh1);
        env_encoder.backward(dh2);
    }

    void zero_grad() {
        early_encoder.zero_grad();
        audio_encoder.zero_grad();
        env_encoder.zero_grad();
        feature_encoder.zero_grad();
        audio_head.zero_grad();
        env_head.zero_grad();
        combiner.zero_grad();
    }
    void step(float lr, int batch_size) {
        early_encoder.step(lr, batch_size);
        audio_encoder.step(lr, batch_size);
        env_encoder.step(lr, batch_size);
        feature_encoder.step(lr, batch_size);
        audio_head.step(lr, batch_size);
        env_head.step(lr, batch_size);
        combiner.step(lr, batch_size);
    }
};
struct Metrics {
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};
static Metrics evaluate(HybridFusionModel& model, const vector<Sample>& data) {
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
    m.precision = (tp + fp > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    m.recall = (tp + fn > 0) ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    m.f1 = (m.precision + m.recall > 0.0f)
        ? 2.0f * m.precision * m.recall / (m.precision + m.recall)
        : 0.0f;
    m.brier = brier / n;
    return m;
}
static void train(HybridFusionModel& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
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
    HybridFusionModel model;
    train(model, train_data, epochs, lr, batch_size);
    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };
    cout << "Hybrid fusion: g(p1, p2, h_early, h_feature, mask, quality)\n";
    cout << "Regime          Acc     Prec    Recall  F1      Brier\n";
    cout << "-------------------------------------------------------\n";
    cout << std::fixed << std::setprecision(4);
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

