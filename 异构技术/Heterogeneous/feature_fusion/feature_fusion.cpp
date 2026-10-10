
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


static float sigmoid(float x)
{
    if (x >= 0.0f)
    {
        const float z = std::exp(-x);

        return 1.0f / (1.0f + z);
    }

    const float z = std::exp(x);

    return z / (1.0f + z);
}


/**
 * @brief Box-Muller算法，生成标准正态分布随机数（均值0，方差1）
 * @param rng 随机数引擎
 * @return 正态随机数
 */
static float randn(std::mt19937 & rng)
{
    std::uniform_real_distribution <float> u(1e - 7f, 1.0f);
    const float a = u(rng);
    const float b = u(rng);

    return std::sqrt(-2.0f * std::log(a)) *std::cos(2.0f * 3.14159265358979323846f * b);
}


static float runif(std::mt19937 & rng, float lo, float hi)
{
    std::uniform_real_distribution <float> u(lo, hi);
    return u(rng);
}


/**
 * @brief 一条样本数据结构
 * 两个模态：audio音频，env环境
 */
struct Sample
{
    float a0 = 0.0f;                                // 观测到的音频特征0
    float a1 = 0.0f;                                // 观测到的音频特征1
    float e0 = 0.0f;                                // 观测到的环境特征0
    float e1 = 0.0f;                                // 观测到的环境特征1

    float qa = 0.0f;                                // 音频模态质量：0最差，1最好
    float qe = 0.0f;                                // 环境模态质量：0最差，1最好

    int ma = 1;                                     // ma=1：音频存在；ma=0：音频缺失
    int me = 1;                                     // me=1：环境存在；me=0：环境缺失

    //有风险（y=1）：听到异响、摔倒、异常噪音。
    //没有风险（y=0）：正常安静环境、正常说话，一切平安。
    int y = 0;                                      // 真实标签：1=有风险，0=无风险
};


static Sample make_sample(std::mt19937 & rng, 
    float miss_a = 0.20f, 
    float miss_e = 0.20f, 
    float qa_override = -1.0f, 
    float qe_override = -1.0f)
{
    Sample s;
    const float la0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float la1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le1 = std::clamp(randn(rng), -3.0f, 3.0f);

    const float risk_logit = 0.8f * la0 + 0.8f * le0 + 2.2f * la0 * le0 + 0.15f * la1 - 0.10f * le1 - 0.25f;

    s.y = (runif(rng, 0.0f, 1.0f) < sigmoid(risk_logit)) ? 1: 0;

    s.ma = (runif(rng, 0.0f, 1.0f) > miss_a) ? 1: 0;
    s.me = (runif(rng, 0.0f, 1.0f) > miss_e) ? 1: 0;
    s.qa = (qa_override >= 0.0f) ? qa_override: runif(rng, 0.40f, 1.0f);
    s.qe = (qe_override >= 0.0f) ? qe_override: runif(rng, 0.40f, 1.0f);

    if (!s.ma)
    {
        s.qa = 0.0f;
    }

    if (!s.me)
    {
        s.qe = 0.0f;
    }

    const float na = 0.10f + 0.70f * (1.0f - s.qa);
    const float ne = 0.10f + 0.70f * (1.0f - s.qe);

    s.a0 = s.ma ? (la0 + na * randn(rng)): 0.0f;
    s.a1 = s.ma ? (la1 + na * randn(rng)): 0.0f;
    s.e0 = s.me ? (le0 + ne * randn(rng)): 0.0f;
    s.e1 = s.me ? (le1 + ne * randn(rng)): 0.0f;
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


static vector <float> env_input(const Sample & s)
{
    return
    {
        s.e0, s.e1, static_cast <float> (s.me), s.qe
    };
}


/**
 * @brief 单层网络模块：线性变换 + Tanh激活函数
 * 用作模态编码器，对输入特征做非线性映射，输出隐表征
 * 包含前向传播、反向求导、梯度清零、SGD参数更新
 */
struct LinearTanh
{
    int in = 0;                 // 输入特征维度 (输入向量里面，有多少个特征数字)
    int out = 0;                // 输出特征维度 (输出向量里面，有多少个特征数字)
    vector<float> w;            // 权重矩阵，一维平铺存储：w[out * in]
    //梯度 ： 参数往哪个方向改动，损失会变大
    vector<float> dw;           // 权重w对应的梯度，用于反向传播累加
    // z = Wx + b : b就是偏置 ， 长度等于输出纬度out
    vector<float> b;            // 偏置向量，维度out
    //梯度 ，从而知道b往哪个方向改，能让预测误差变小
    vector<float> db;           // 偏置b对应的梯度


    vector<float> x;            // 缓存本次前向的原始输入向量，反向传播要用
    vector<float> z;            // 缓存预激活值 z = Wx + b
    vector<float> h;            // 缓存激活输出 h = tanh(z)

    /**
     * @brief 构造函数：初始化权重与偏置
     * @param input_dim 输入维度
     * @param output_dim 输出维度
     * @param seed 随机种子，保证每次初始化权重固定可复现
     */
    LinearTanh(int input_dim, int output_dim, int seed) : in(input_dim), out(output_dim)
    {
        std::mt19937 rng(seed);
        // Xavier初始化：limit = 1/sqrt(in)，防止激活输入过大/过小，缓解梯度消失爆炸
        const float limit = 1.0f / std::sqrt(static_cast<float>(input_dim));
        // 在[-limit, limit]均匀分布采样初始化权重
        std::uniform_real_distribution<float> init(-limit, limit);

        // 权重数组大小 = 输出数 × 输入数，二维矩阵平铺成一维存储
        w.resize(in * out);
        //assign(n，val) ： 清空vector原来的内容
        //   , 重新创建n个元素，全部赋值为val
        dw.assign(w.size(), 0.0f); // 梯度数组初始化为0
        b.assign(out, 0.0f);       // 偏置初始化为0
        db.assign(out, 0.0f);

        // 遍历所有权重，随机初始化
        for (float & value: w)
        {
            value = init(rng);
        }
    }

    /**
     * @brief 前向传播：计算线性变换+Tanh激活
     * @param input 输入特征向量，维度in
     * @return h：经过tanh激活后的输出向量，维度out
     */
    vector<float> forward(const vector<float> &input)
    {
        x = input;                  // 保存输入，反向传播需要
        z.assign(out, 0.0f);        // 预激活数组清零
        h.assign(out, 0.0f);        // 激活输出数组清零

        // 遍历每一个输出神经元
        for (int o = 0; o < out; ++o)
        {
            float sum = b[o];       // 先加偏置项
            // 输入各特征加权求和
            for (int i = 0; i < in; ++i)
            {
                // w[o * in + i]：第o个输出神经元，对应第i个输入的权重
                //二维表 ， 行是输出 ，列是输入 。
                //一行有in个元素，先跳过前面o整行，再偏移i
                sum += w[o * in + i] * x[i];
            }
            z[o] = sum;             // 保存线性预激活结果 z = Wx + b
            // std::tanh : (e(x)-exp(-x) ) / (exp(x)+exp(-x) )
            //输入 ： R         输出 ：(-1 ,   1)
            h[o] = std::tanh(sum);  // tanh激活，映射到(-1,1)
        }
        return h;
    }

    /**
     * @brief 反向传播：根据输出梯度dh，计算参数梯度并回传输入梯度
     * @param dh 损失对本层输出h的梯度，维度out
     * @return dx 损失对本层输入x的梯度，回传给上一层
     */
    vector<float> backward(const vector<float> &dh)
    {
        vector<float> dx(in, 0.0f); // 保存对输入x的梯度
        vector<float> dz(out, 0.0f);

        for (int o = 0; o < out; ++o)
        {
            // tanh导数：d(tanh(z))/dz = 1 - h²
            // dz = dh * d(tanh)/dz
            dz[o] = dh[o] * (1.0f - h[o] * h[o]);

            // 计算权重梯度 + 回传输入梯度
            for (int i = 0; i < in; ++i)
            {
                // dw = dz * x ：损失对权重w的梯度，累加
                dw[o * in + i] += dz[o] * x[i];
                // dx = dz * w ：损失对输入x的梯度，回传给前一层
                dx[i] += w[o * in + i] * dz[o];
            }
            // 偏置梯度：db = dz
            db[o] += dz[o];
        }
        return dx;
    }

    /**
     * @brief 梯度清零
     * 每个batch开始前调用，清空上一轮累加的梯度
     */
    void zero_grad()
    {
        std::fill(dw.begin(), dw.end(), 0.0f);
        std::fill(db.begin(), db.end(), 0.0f);
    }

    /**
     * @brief SGD随机梯度下降更新参数
     * @param lr 学习率
     * @param batch_size 当前批次样本数量，梯度需要除以batch求平均
     */
    void step(float lr, int batch_size)
    {
        // scale = lr / batch_size，对batch内累加的梯度做平均
        const float scale = lr / static_cast<float>(batch_size);
        // 更新权重 w = w - lr * dw / batch_size
        for (size_t i = 0; i < w.size(); ++i)
        {
            w[i] -= scale * dw[i];
        }
        // 更新偏置 b = b - lr * db / batch_size
        for (size_t i = 0; i < b.size(); ++i)
        {
            b[i] -= scale * db[i];
        }
    }
};






/**
 * @brief 两层MLP分类头
 * 结构：输入层 -> 线性w1+b1 -> ReLU激活 -> 线性w2+b2 -> 输出单个logit
 * 包含：前向传播forward、反向求导backward、梯度清零zero_grad、SGD参数更新step
 */
struct MLPHead
{
    int in = 0;          // 输入特征维度
    int hidden = 0;      // 隐藏层神经元数量

    vector <float> w1;   // 第一层权重矩阵，一维平铺存储：w1[hidden * in]
    vector <float> dw1;  // w1对应的梯度
    vector <float> b1;   // 第一层偏置，维度hidden
    vector <float> db1;  // b1对应的梯度

    vector <float> w2;   // 第二层权重，维度hidden（隐藏层→单个输出）
    vector <float> dw2;  // w2对应的梯度
    float b2 = 0.0f;     // 第二层偏置，输出只有1个值，所以是float标量
    float db2 = 0.0f;    // b2对应的梯度

    vector <float> x;    // 缓存本次forward的输入向量，反向传播要用
    vector <float> u;    // 缓存第一层线性结果 u = w1*x + b1
    vector <float> a;    // 缓存ReLU激活输出 a = ReLU(u)

    /**
     * @brief MLP构造函数，初始化权重与偏置
     * @param input_dim 输入特征维度
     * @param hidden_dim 隐藏层神经元个数
     * @param seed 随机种子，保证权重初始化可复现
     */
    MLPHead(int input_dim, int hidden_dim, int seed) : in(input_dim), hidden(hidden_dim)
    {
        std::mt19937 rng(seed);
        // Xavier初始化，控制权重范围，缓解梯度消失/爆炸
        const float limit = 1.0f / std::sqrt(static_cast <float> (input_dim));
        std::uniform_real_distribution <float> init(-limit, limit);

        // 分配第一层权重内存
        w1.resize(hidden * in);
        dw1.assign(w1.size(), 0.0f); // 梯度初始化为0
        b1.assign(hidden, 0.0f);
        db1.assign(hidden, 0.0f);

        // 第二层权重、梯度初始化
        w2.assign(hidden, 0.0f);
        dw2.assign(hidden, 0.0f);

        // 遍历w1，随机初始化权重
        for (float & value: w1)
        {
            value = init(rng);
        }
        // 遍历w2，随机初始化权重
        for (float & value: w2)
        {
            value = init(rng);
        }
    }

    /**
     * @brief 前向传播，计算MLP输出logit
     * @param input 输入特征向量
     * @return logit 模型原始输出（未经过sigmoid）
     */
    float forward(const vector <float> &input)
    {
        x = input;               // 保存输入，反向传播需要
        u.assign(hidden, 0.0f);
        a.assign(hidden, 0.0f);

        // 第一层：线性变换 + ReLU激活
        for (int j = 0; j < hidden; ++j)
        {
            float sum = b1[j];    // 先加偏置b1[j]
            for (int i = 0; i < in; ++i)
            {
                sum += w1[j * in + i] * x[i];
            }
            u[j] = sum;
            a[j] = std::max(0.0f, sum);             // ReLU：小于0截断为0
        }

        // 第二层：隐藏层到输出单个logit
        float logit = b2;
        for (int j = 0; j < hidden; ++j)
        {
            logit += w2[j] * a[j];
        }
        return logit;
    }

    /**
     * @brief 反向传播，计算各参数梯度并累加
     * @param grad 输出logit传来的上游梯度 dLoss/dlogit
     * @return dx 对输入x的梯度 dLoss/dx，传回前面网络
     */
    vector <float> backward(float grad)
    {
        vector <float> dx(in, 0.0f);
        vector <float> du(hidden, 0.0f);

        // 求第二层w2、b2梯度，同时求du（损失对u的梯度）
        for (int j = 0; j < hidden; ++j)
        {
            dw2[j] += grad * a[j];
            // ReLU导数：u>0导数为1，否则0
            du[j] = grad * w2[j] * (u[j] > 0.0f ? 1.0f: 0.0f);
        }
        db2 += grad;

        // 求第一层w1、b1梯度，累加dx
        for (int j = 0; j < hidden; ++j)
        {
            for (int i = 0; i < in; ++i)
            {
                dw1[j * in + i] += du[j] * x[i];
                dx[i] += w1[j * in + i] * du[j];
            }
            db1[j] += du[j];
        }
        return dx;
    }

    /**
     * @brief 梯度清零，每一轮batch训练前调用，防止梯度累加
     */
    void zero_grad()
    {
        std::fill(dw1.begin(), dw1.end(), 0.0f);
        std::fill(db1.begin(), db1.end(), 0.0f);
        std::fill(dw2.begin(), dw2.end(), 0.0f);
        db2 = 0.0f;
    }

    /**
     * @brief SGD参数更新，用累积梯度更新权重和偏置
     * @param lr 学习率
     * @param batch_size 批次大小，用来平均梯度
     */
    void step(float lr, int batch_size)
    {
        const float scale = lr / static_cast <float> (batch_size);
        // 更新w1
        for (size_t i = 0; i < w1.size(); ++i)
        {
            w1[i] -= scale * dw1[i];
        }
        // 更新b1
        for (size_t i = 0; i < b1.size(); ++i)
        {
            b1[i] -= scale * db1[i];
        }
        // 更新w2
        for (size_t i = 0; i < w2.size(); ++i)
        {
            w2[i] -= scale * dw2[i];
        }
        // 更新标量偏置b2
        b2 -= scale * db2;
    }
};




struct FeatureFusionModel
{
    //音频编码器
    LinearTanh audio_encoder
    {
        4, H, 101
    };
    //环境编码器
    LinearTanh env_encoder
    {
        4, H, 202
    };
    //两层全连接 MLP
    MLPHead head
    {
        2 * H, 16, 303
    };
    vector <float> h1;
    vector <float> h2;
    //`fused` 用来存放**音频特征 h1 和环境特征 h2 拼接在一起的融合特征**，
    //   然后送入后面的 MLP 头网络做预测。
    vector <float> fused;


    float forward(const Sample & s)
    {
        h1 = audio_encoder.forward(audio_input(s)); // h1 = E1(x1)
        h2 = env_encoder.forward(env_input(s));     // h2 = E2(x2)
        //h1长度为8 ， h2长度为8 , 拼接之后长度为16
        fused = h1;
        fused.insert(fused.end(), h2.begin(), h2.end()); // F(h1, h2) = concat
        return head.forward(fused);
    }


    void backward(float grad)
    {
        const vector <float> dfused = head.backward(grad);
        const vector <float> dh1(dfused.begin(), dfused.begin() +H);
        const vector <float> dh2(dfused.begin() +H, dfused.end());

        audio_encoder.backward(dh1);
        env_encoder.backward(dh2);
    }


    void zero_grad()
    {
        audio_encoder.zero_grad();
        env_encoder.zero_grad();
        head.zero_grad();
    }


    void step(float lr, int batch_size)
    {
        audio_encoder.step(lr, batch_size);
        env_encoder.step(lr, batch_size);
        head.step(lr, batch_size);
    }
};


struct Metrics
{
    float accuracy = 0.0f;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};


static Metrics evaluate(FeatureFusionModel & model, const vector <Sample> &data)
{
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    float brier = 0.0f;

    for (const Sample & s: data)
    {
        const float p = sigmoid(model.forward(s));
        const int pred = (p >= 0.5f) ? 1: 0;

        tp += (pred == 1 && s.y == 1);
        fp += (pred == 1 && s.y == 0);
        tn += (pred == 0 && s.y == 0);
        fn += (pred == 0 && s.y == 1);
        brier += (p - static_cast <float> (s.y)) * (p - static_cast <float> (s.y));
    }

    const float n = static_cast <float> (data.size());
    Metrics m;

    m.accuracy = static_cast <float> (tp + tn) / n;
    m.precision = (tp + fp > 0) ? static_cast <float> (tp) / static_cast <float> (tp + fp): 0.0f;
    m.recall = (tp + fn > 0) ? static_cast <float> (tp) / static_cast <float> (tp + fn): 0.0f;
    m.f1 = (m.precision + m.recall > 0.0f) ? 2.0f * m.precision * m.recall / (m.precision + m.recall): 0.0f;
    m.brier = brier / n;
    return m;
}


static void train(FeatureFusionModel & model, vector <Sample> &data, int epochs, float lr, int batch_size)
{
    std::mt19937 rng(12345);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(data.begin(), data.end(), rng);

        for (size_t start = 0; start < data.size(); start += batch_size)
        {
            const size_t end = std::min(data.size(), start + batch_size);
            //每个批次先清空梯度
            model.zero_grad();

            for (size_t i = start; i < end; ++i)
            {
                const float p = sigmoid(model.forward(data[i]));

                model.backward(p - static_cast <float> (data[i].y));
            }

            model.step(lr, static_cast <int> (end - start));
        }
    }
}


static Sample make_test_sample(std::mt19937 & rng, int regime)
{
    switch (regime)
    {
        case 0:
            return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f);

        case 1:
            return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f);

        case 2:
            return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f);

        case 3:
            return make_sample(rng, 0.0f, 0.0f, 0.2f, 0.9f);

        default:
            return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.2f);
    }
}


int main()
{
    constexpr int train_size = 4000;
    constexpr int test_size = 1200;
    constexpr int epochs = 200;
    constexpr float lr = 0.04f;
    constexpr int batch_size = 64;

    std::mt19937 rng(20261009);
    vector <Sample> train_data;
    train_data.reserve(train_size);

    for (int i = 0; i < train_size; ++i)
    {
        train_data.push_back(make_sample(rng, 0.20f, 0.20f));
    }

    vector < vector < Sample >> test_data(5);

    for (int regime = 0; regime < 5; ++regime)
    {
        test_data[regime].reserve(test_size);

        for (int i = 0; i < test_size; ++i)
        {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    FeatureFusionModel model;

    train(model, train_data, epochs, lr, batch_size);

    const vector <string> names =
    {
        "complete", 
        "missing-audio", 
        "missing-env", 
        "noisy-audio", 
        "noisy-env", 
    };

    cout << "Feature-level fusion: h1=E1(audio), h2=E2(env), [h1;h2] -> shared head\n";
    cout << "Regime          Acc     Prec    Recall  F1      Brier\n";
    cout << "-------------------------------------------------------\n";
    cout << std::fixed << std::setprecision(4);

    for (int regime = 0; regime < 5; ++regime)
    {
        const Metrics m = evaluate(model, test_data[regime]);

        cout << std::left << std::setw(16) << names[regime] << std::setw(8) << m.accuracy << std::setw(8) << m.precision << std::setw(8) << m.recall << std::setw(8) << m.f1 << std::setw(8) << m.brier << '\n';
    }

    return 0;
}


