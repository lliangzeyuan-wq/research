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

static constexpr int K = 4;  // number of experts

enum class Act {
    Linear,
    ReLU,
    Tanh,
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
        0.7f * la0 + 0.7f * le0 + 2.4f * la0 * le0 +
        0.12f * la1 - 0.10f * le1 - 0.10f;
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

static vector<float> joint_input(const Sample& s) {
    return {
        s.a0, s.a1, static_cast<float>(s.ma), s.qa,
        s.e0, s.e1, static_cast<float>(s.me), s.qe,
    };
}

static vector<float> gate_input(const Sample& s) {
    return {
        s.a0, s.a1, static_cast<float>(s.ma), s.qa,
        s.e0, s.e1, static_cast<float>(s.me), s.qe,
    };
}

static vector<float> fallback_input(const Sample& s) {
    return {
        static_cast<float>(s.ma),
        s.qa,
        static_cast<float>(s.me),
        s.qe,
    };
}

struct LayerSpec {
    int in = 0;
    int out = 0;
    Act act = Act::Linear;
};

struct Layer {
    int in = 0;
    int out = 0;
    Act act = Act::Linear;
    vector<float> w;
    vector<float> dw;
    vector<float> b;
    vector<float> db;
    vector<float> x;
    vector<float> z;
    vector<float> h;

    Layer(const LayerSpec& spec, std::mt19937& rng)
        : in(spec.in), out(spec.out), act(spec.act) {
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
        z.assign(out, 0.0f);
        h.assign(out, 0.0f);
        for (int o = 0; o < out; ++o) {
            float sum = b[o];
            for (int i = 0; i < in; ++i) {
                sum += w[o * in + i] * x[i];
            }
            z[o] = sum;
            if (act == Act::ReLU) {
                h[o] = std::max(0.0f, sum);
            } else if (act == Act::Tanh) {
                h[o] = std::tanh(sum);
            } else {
                h[o] = sum;
            }
        }
        return h;
    }

    vector<float> backward(const vector<float>& dh) {
        vector<float> dx(in, 0.0f);
        for (int o = 0; o < out; ++o) {
            float d = dh[o];
            if (act == Act::ReLU) {
                d *= (z[o] > 0.0f ? 1.0f : 0.0f);
            } else if (act == Act::Tanh) {
                d *= (1.0f - h[o] * h[o]);
            }
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

struct MLP {
    vector<Layer> layers;

    MLP(const vector<LayerSpec>& specs, int seed) {
        std::mt19937 rng(seed);
        for (const LayerSpec& spec : specs) {
            layers.emplace_back(spec, rng);
        }
    }

    vector<float> forward(const vector<float>& input) {
        vector<float> y = input;
        for (Layer& layer : layers) {
            y = layer.forward(y);
        }
        return y;
    }

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

struct MoEModel {
    bool learn_gate = true;

    // Expert 0: audio, 1: environment, 2: joint fusion, 3: missing fallback.
    MLP audio_expert{
        {{4, 8, Act::ReLU}, {8, 1, Act::Linear}},
        101
    };
    MLP env_expert{
        {{4, 8, Act::ReLU}, {8, 1, Act::Linear}},
        202
    };
    MLP joint_expert{
        {{8, 16, Act::ReLU}, {16, 1, Act::Linear}},
        303
    };
    MLP fallback_expert{
        {{4, 8, Act::ReLU}, {8, 1, Act::Linear}},
        404
    };
    MLP gate{
        {{8, K, Act::Linear}},
        505
    };

    vector<float> gx;
    vector<float> gate_logits;
    vector<float> gates;
    vector<float> expert_logits;
    bool force_fallback = false;

    MoEModel(bool use_gate, int seed) : learn_gate(use_gate) {
        (void)seed;
    }

    int parameter_count() const {
        const int audio = 4 * 8 + 8 + 8 * 1 + 1;
        const int env = 4 * 8 + 8 + 8 * 1 + 1;
        const int joint = 8 * 16 + 16 + 16 * 1 + 1;
        const int fallback = 4 * 8 + 8 + 8 * 1 + 1;
        const int gating = 8 * K + K;
        return audio + env + joint + fallback + gating;
    }

    float forward(const Sample& s) {
        gx = gate_input(s);
        gate_logits = gate.forward(gx);

        gates.assign(K, 0.0f);
        force_fallback = (!s.ma && !s.me);
        if (force_fallback) {
            gates[0] = 0.0f;
            gates[1] = 0.0f;
            gates[2] = 0.0f;
            gates[3] = 1.0f;
        } else if (learn_gate) {
            float max_logit = *std::max_element(gate_logits.begin(), gate_logits.end());
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                gates[k] = std::exp(gate_logits[k] - max_logit);
                sum += gates[k];
            }
            for (int k = 0; k < K; ++k) {
                gates[k] /= sum;
            }
        } else {
            for (int k = 0; k < K; ++k) {
                gates[k] = 1.0f / static_cast<float>(K);
            }
        }

        expert_logits.assign(K, 0.0f);
        expert_logits[0] = audio_expert.forward(audio_input(s))[0];
        expert_logits[1] = env_expert.forward(env_input(s))[0];
        expert_logits[2] = joint_expert.forward(joint_input(s))[0];
        expert_logits[3] = fallback_expert.forward(fallback_input(s))[0];

        float logit = 0.0f;
        for (int k = 0; k < K; ++k) {
            logit += gates[k] * expert_logits[k];
        }
        return logit;
    }

    void backward(float grad) {
        vector<float> d_gate_logits(K, 0.0f);
        float weighted_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            d_gate_logits[k] = grad * expert_logits[k];
            weighted_sum += gates[k] * d_gate_logits[k];
        }

        // Softmax derivative.
        for (int k = 0; k < K; ++k) {
            d_gate_logits[k] = gates[k] * (d_gate_logits[k] - weighted_sum);
        }

        audio_expert.backward({grad * gates[0]});
        env_expert.backward({grad * gates[1]});
        joint_expert.backward({grad * gates[2]});
        fallback_expert.backward({grad * gates[3]});

        if (learn_gate && !force_fallback) {
            gate.backward(d_gate_logits);
        }
    }

    void zero_grad() {
        audio_expert.zero_grad();
        env_expert.zero_grad();
        joint_expert.zero_grad();
        fallback_expert.zero_grad();
        gate.zero_grad();
    }

    void step(float lr, int batch_size) {
        audio_expert.step(lr, batch_size);
        env_expert.step(lr, batch_size);
        joint_expert.step(lr, batch_size);
        fallback_expert.step(lr, batch_size);
        if (learn_gate) {
            gate.step(lr, batch_size);
        }
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};

static Metrics evaluate(MoEModel& model, const vector<Sample>& data) {
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

static void train(MoEModel& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
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
        case 4: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.20f, 0.03f, 0.70f);
        default: return make_sample(rng, 1.0f, 1.0f, -1.0f, -1.0f, 0.03f, 0.03f);
    }
}

int main() {
    constexpr int train_size = 4000;
    constexpr int test_size = 900;
    constexpr int epochs = 220;
    constexpr float lr = 0.04f;
    constexpr int batch_size = 64;

    std::mt19937 rng(20261012);
    vector<Sample> train_data;
    train_data.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        train_data.push_back(make_sample(rng, 0.20f, 0.20f));
    }

    vector<vector<Sample>> test_data(6);
    for (int regime = 0; regime < 6; ++regime) {
        test_data[regime].reserve(test_size);
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    MoEModel fixed(false, 777);
    MoEModel learned(true, 777);
    vector<Sample> fixed_train = train_data;
    vector<Sample> learned_train = train_data;
    train(fixed, fixed_train, epochs, lr, batch_size);
    train(learned, learned_train, epochs, lr, batch_size);

    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
        "both-missing",
    };

    cout << "MoE parameters: " << learned.parameter_count() << "\n\n";

    cout << "Fixed equal-weight expert mixture\n";
    cout << "Regime          Acc     F1      Brier\n";
    cout << "----------------------------------------\n";
    cout << std::fixed << std::setprecision(4);
    for (int regime = 0; regime < 6; ++regime) {
        const Metrics m = evaluate(fixed, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier << '\n';
    }

    cout << "\nLearned gating MoE\n";
    cout << "Regime          Acc     F1      Brier\n";
    cout << "----------------------------------------\n";
    for (int regime = 0; regime < 6; ++regime) {
        const Metrics m = evaluate(learned, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier << '\n';
    }
    return 0;
}
