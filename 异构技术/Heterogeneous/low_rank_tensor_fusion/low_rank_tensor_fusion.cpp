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

static constexpr int H = 12;   // modality encoder width
static constexpr int R = 4;    // low-rank factors
static constexpr int DH = 8;   // LMF hidden output dimension

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
    float qe_override = -1.0f,
    float noise_a = 0.08f,
    float noise_e = 0.08f
) {
    Sample s;
    const float la0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float la1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le1 = std::clamp(randn(rng), -3.0f, 3.0f);

    const float risk_logit =
        0.6f * la0 + 0.6f * le0 + 2.8f * la0 * le0 +
        0.10f * la1 - 0.08f * le1 - 0.35f;
    s.y = (runif(rng, 0.0f, 1.0f) < sigmoid(risk_logit)) ? 1 : 0;

    s.ma = (runif(rng, 0.0f, 1.0f) > miss_a) ? 1 : 0;
    s.me = (runif(rng, 0.0f, 1.0f) > miss_e) ? 1 : 0;
    s.qa = (qa_override >= 0.0f) ? qa_override : runif(rng, 0.45f, 1.0f);
    s.qe = (qe_override >= 0.0f) ? qe_override : runif(rng, 0.45f, 1.0f);
    if (!s.ma) {
        s.qa = 0.0f;
    }
    if (!s.me) {
        s.qe = 0.0f;
    }

    const float na = noise_a + 0.35f * (1.0f - s.qa);
    const float ne = noise_e + 0.35f * (1.0f - s.qe);
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

struct LinearTanh {
    int in = 0;
    int out = 0;
    vector<float> w;
    vector<float> dw;
    vector<float> b;
    vector<float> db;
    vector<float> x;
    vector<float> h;

    LinearTanh(int input_dim, int output_dim, int seed)
        : in(input_dim), out(output_dim) {
        std::mt19937 rng(seed);
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

    vector<float> backward(const vector<float>& dh) {
        vector<float> dx(in, 0.0f);
        for (int o = 0; o < out; ++o) {
            const float dz = dh[o] * (1.0f - h[o] * h[o]);
            for (int i = 0; i < in; ++i) {
                dw[o * in + i] += dz * x[i];
                dx[i] += w[o * in + i] * dz;
            }
            db[o] += dz;
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

struct Logistic {
    vector<float> w;
    vector<float> dw;
    float b = 0.0f;
    float db = 0.0f;

    explicit Logistic(int n) : w(n, 0.0f), dw(n, 0.0f) {}

    float forward(const vector<float>& x) const {
        return dot(w, x) + b;
    }

    vector<float> backward(float grad, const vector<float>& x) {
        vector<float> dx(w.size(), 0.0f);
        for (size_t i = 0; i < w.size(); ++i) {
            dw[i] += grad * x[i];
            dx[i] = grad * w[i];
        }
        db += grad;
        return dx;
    }

    void zero_grad() {
        std::fill(dw.begin(), dw.end(), 0.0f);
        db = 0.0f;
    }

    void step(float lr, int batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] -= scale * dw[i];
        }
        b -= scale * db;
    }
};

struct FullTensorModel {
    LinearTanh audio_encoder{4, H, 101};
    LinearTanh env_encoder{4, H, 202};
    Logistic head{(H + 1) * (H + 1)};

    vector<float> a_aug;
    vector<float> e_aug;
    vector<float> tensor;
    float qa = 0.0f;
    float qe = 0.0f;

    int parameter_count() const {
        const int encoder = 2 * (4 * H + H);
        const int fusion_classifier = (H + 1) * (H + 1) + 1;
        return encoder + fusion_classifier;
    }

    float forward(const Sample& s) {
        qa = s.ma ? s.qa : 0.0f;
        qe = s.me ? s.qe : 0.0f;
        const vector<float> h1 = audio_encoder.forward(audio_input(s));
        const vector<float> h2 = env_encoder.forward(env_input(s));

        a_aug.assign(H + 1, 0.0f);
        e_aug.assign(H + 1, 0.0f);
        for (int i = 0; i < H; ++i) {
            a_aug[i] = h1[i] * qa;
            e_aug[i] = h2[i] * qe;
        }
        a_aug[H] = 1.0f;
        e_aug[H] = 1.0f;

        tensor.assign((H + 1) * (H + 1), 0.0f);
        for (int i = 0; i <= H; ++i) {
            for (int j = 0; j <= H; ++j) {
                tensor[i * (H + 1) + j] = a_aug[i] * e_aug[j];
            }
        }
        return head.forward(tensor);
    }

    void backward(float grad) {
        const vector<float> dtensor = head.backward(grad, tensor);
        vector<float> da_aug(H + 1, 0.0f);
        vector<float> de_aug(H + 1, 0.0f);

        for (int i = 0; i <= H; ++i) {
            for (int j = 0; j <= H; ++j) {
                const float g = dtensor[i * (H + 1) + j];
                da_aug[i] += g * e_aug[j];
                de_aug[j] += g * a_aug[i];
            }
        }

        vector<float> dh1(H, 0.0f);
        vector<float> dh2(H, 0.0f);
        for (int i = 0; i < H; ++i) {
            dh1[i] = da_aug[i] * qa;
            dh2[i] = de_aug[i] * qe;
        }
        audio_encoder.backward(dh1);
        env_encoder.backward(dh2);
    }

    void zero_grad() {
        audio_encoder.zero_grad();
        env_encoder.zero_grad();
        head.zero_grad();
    }

    void step(float lr, int batch_size) {
        audio_encoder.step(lr, batch_size);
        env_encoder.step(lr, batch_size);
        head.step(lr, batch_size);
    }
};

struct LowRankTensorModel {
    LinearTanh audio_encoder{4, H, 101};
    LinearTanh env_encoder{4, H, 202};

    // Modality-specific factors and output factors.
    vector<float> fa;  // R * (H + 1)
    vector<float> fe;  // R * (H + 1)
    vector<float> fo;  // R * DH
    vector<float> dfa;
    vector<float> dfe;
    vector<float> dfo;

    Logistic head{DH};

    vector<float> a_aug;
    vector<float> e_aug;
    vector<float> sa;
    vector<float> se;
    vector<float> z;
    float qa = 0.0f;
    float qe = 0.0f;

    explicit LowRankTensorModel(int seed) {
        fa.assign(R * (H + 1), 0.0f);
        fe.assign(R * (H + 1), 0.0f);
        fo.assign(R * DH, 0.0f);
        dfa.assign(fa.size(), 0.0f);
        dfe.assign(fe.size(), 0.0f);
        dfo.assign(fo.size(), 0.0f);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> init(-0.35f, 0.35f);
        for (float& value : fa) value = init(rng);
        for (float& value : fe) value = init(rng);
        for (float& value : fo) value = init(rng);
    }

    int parameter_count() const {
        const int encoder = 2 * (4 * H + H);
        const int factors = R * (2 * (H + 1) + DH);
        const int classifier = DH + 1;
        return encoder + factors + classifier;
    }

    float forward(const Sample& s) {
        qa = s.ma ? s.qa : 0.0f;
        qe = s.me ? s.qe : 0.0f;
        const vector<float> h1 = audio_encoder.forward(audio_input(s));
        const vector<float> h2 = env_encoder.forward(env_input(s));

        a_aug.assign(H + 1, 0.0f);
        e_aug.assign(H + 1, 0.0f);
        for (int i = 0; i < H; ++i) {
            a_aug[i] = h1[i] * qa;
            e_aug[i] = h2[i] * qe;
        }
        a_aug[H] = 1.0f;
        e_aug[H] = 1.0f;

        sa.assign(R, 0.0f);
        se.assign(R, 0.0f);
        z.assign(DH, 0.0f);

        for (int r = 0; r < R; ++r) {
            for (int i = 0; i <= H; ++i) {
                sa[r] += fa[r * (H + 1) + i] * a_aug[i];
                se[r] += fe[r * (H + 1) + i] * e_aug[i];
            }
            const float scale = sa[r] * se[r];
            for (int k = 0; k < DH; ++k) {
                z[k] += scale * fo[r * DH + k];
            }
        }
        return head.forward(z);
    }

    void backward(float grad) {
        const vector<float> dz = head.backward(grad, z);
        vector<float> da_aug(H + 1, 0.0f);
        vector<float> de_aug(H + 1, 0.0f);

        for (int r = 0; r < R; ++r) {
            float ds = 0.0f;
            for (int k = 0; k < DH; ++k) {
                dfo[r * DH + k] += sa[r] * se[r] * dz[k];
                ds += dz[k] * fo[r * DH + k];
            }

            const float dsa = ds * se[r];
            const float dse = ds * sa[r];
            for (int i = 0; i <= H; ++i) {
                dfa[r * (H + 1) + i] += dsa * a_aug[i];
                dfe[r * (H + 1) + i] += dse * e_aug[i];
                da_aug[i] += dsa * fa[r * (H + 1) + i];
                de_aug[i] += dse * fe[r * (H + 1) + i];
            }
        }

        vector<float> dh1(H, 0.0f);
        vector<float> dh2(H, 0.0f);
        for (int i = 0; i < H; ++i) {
            dh1[i] = da_aug[i] * qa;
            dh2[i] = de_aug[i] * qe;
        }
        audio_encoder.backward(dh1);
        env_encoder.backward(dh2);
    }

    void zero_grad() {
        audio_encoder.zero_grad();
        env_encoder.zero_grad();
        head.zero_grad();
        std::fill(dfa.begin(), dfa.end(), 0.0f);
        std::fill(dfe.begin(), dfe.end(), 0.0f);
        std::fill(dfo.begin(), dfo.end(), 0.0f);
    }

    void step(float lr, int batch_size) {
        audio_encoder.step(lr, batch_size);
        env_encoder.step(lr, batch_size);
        head.step(lr, batch_size);

        const float scale = lr / static_cast<float>(batch_size);
        for (size_t i = 0; i < fa.size(); ++i) {
            fa[i] -= scale * dfa[i];
            fe[i] -= scale * dfe[i];
        }
        for (size_t i = 0; i < fo.size(); ++i) {
            fo[i] -= scale * dfo[i];
        }
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};

template <typename Model>
static Metrics evaluate(Model& model, const vector<Sample>& data) {
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

template <typename Model>
static void train(Model& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
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
        case 0: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f, 0.03f, 0.03f);
        case 1: return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f, 0.03f, 0.03f);
        case 2: return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f, 0.03f, 0.03f);
        case 3: return make_sample(rng, 0.0f, 0.0f, 0.20f, 0.9f, 0.70f, 0.03f);
        default: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.20f, 0.03f, 0.70f);
    }
}

int main() {
    constexpr int train_size = 3500;
    constexpr int test_size = 900;
    constexpr int epochs = 220;
    constexpr float lr = 0.035f;
    constexpr int batch_size = 64;

    std::mt19937 rng(20261012);
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

    FullTensorModel full;
    LowRankTensorModel low_rank(777);

    vector<Sample> full_train = train_data;
    vector<Sample> low_rank_train = train_data;
    train(full, full_train, epochs, lr, batch_size);
    train(low_rank, low_rank_train, epochs, lr, batch_size);

    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };

    cout << "Full tensor parameters:     " << full.parameter_count() << '\n';
    cout << "Low-rank tensor parameters: " << low_rank.parameter_count() << "\n\n";

    cout << "Full tensor fusion\n";
    cout << "Regime          Acc     F1      Brier\n";
    cout << "----------------------------------------\n";
    cout << std::fixed << std::setprecision(4);
    for (int regime = 0; regime < 5; ++regime) {
        const Metrics m = evaluate(full, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier << '\n';
    }

    cout << "\nLow-rank tensor fusion\n";
    cout << "Regime          Acc     F1      Brier\n";
    cout << "----------------------------------------\n";
    for (int regime = 0; regime < 5; ++regime) {
        const Metrics m = evaluate(low_rank, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier << '\n';
    }
    return 0;
}
