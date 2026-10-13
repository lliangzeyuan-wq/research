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

static constexpr int DA = 4;
static constexpr int DE = 4;
static constexpr int DS = 4;  // shared factor dimension
static constexpr int DU = 4;  // unique factor dimension
static constexpr float TAU = 0.10f;

static float dot(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief Box-Muller算法，生成标准正态分布随机数（均值0，方差1）
 * @param rng 随机数引擎
 * @return 正态随机数
 */
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
    vector<float> audio;
    vector<float> env;
    int paired = 1;
    float noise_a = 0.0f;
    float noise_e = 0.0f;
};

struct MixMatrix {
    vector<float> values;
    int rows = 0;
    int cols = 0;

    float at(int r, int c) const {
        return values[r * cols + c];
    }
};

static MixMatrix make_mix(int rows, int cols, int seed) {
    MixMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.values.assign(rows * cols, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> init(-0.8f, 0.8f);
    for (float& value : m.values) {
        value = init(rng);
    }
    return m;
}

static vector<float> apply_mix(const MixMatrix& m, const vector<float>& z) {
    vector<float> out(m.rows, 0.0f);
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            out[r] += m.at(r, c) * z[c];
        }
    }
    return out;
}

static Sample make_sample(
    std::mt19937& rng,
    const MixMatrix& a_shared,
    const MixMatrix& a_unique,
    const MixMatrix& e_shared,
    const MixMatrix& e_unique,
    float miss_prob = 0.0f,
    float noise_a = 0.10f,
    float noise_e = 0.10f
) {
    Sample s;
    vector<float> z(DS);
    vector<float> ua(DU);
    vector<float> ue(DU);
    for (float& value : z) value = randn(rng);
    for (float& value : ua) value = randn(rng);
    for (float& value : ue) value = randn(rng);

    const vector<float> shared_a = apply_mix(a_shared, z);
    const vector<float> unique_a = apply_mix(a_unique, ua);
    const vector<float> shared_e = apply_mix(e_shared, z);
    const vector<float> unique_e = apply_mix(e_unique, ue);

    s.paired = (runif(rng, 0.0f, 1.0f) > miss_prob) ? 1 : 0;
    s.audio.resize(DA);
    s.env.resize(DE);
    for (int i = 0; i < DA; ++i) {
        s.audio[i] = shared_a[i]
                   + 0.3f * unique_a[i]
                   + noise_a * randn(rng);
    }
    for (int i = 0; i < DE; ++i) {
        s.env[i] = shared_e[i]
                 + 0.3f * unique_e[i]
                 + noise_e * randn(rng);
    }
    s.noise_a = noise_a;
    s.noise_e = noise_e;
    if (!s.paired) {
        // A broken pair is an independent environment sample.
        for (float& value : s.env) value = noise_e * randn(rng);
    }
    return s;
}

struct Linear {
    int in = 0;
    int out = 0;
    vector<float> w;
    vector<float> dw;

    Linear(int input_dim, int output_dim, int seed)
        : in(input_dim), out(output_dim) {
        std::mt19937 rng(seed);
        const float limit = 1.0f / std::sqrt(static_cast<float>(in));
        std::uniform_real_distribution<float> init(-limit, limit);
        w.resize(in * out);
        dw.assign(w.size(), 0.0f);
        for (float& value : w) {
            value = init(rng);
        }
    }

    vector<float> forward(const vector<float>& x) const {
        vector<float> y(out, 0.0f);
        for (int o = 0; o < out; ++o) {
            for (int i = 0; i < in; ++i) {
                y[o] += w[o * in + i] * x[i];
            }
        }
        return y;
    }

    vector<float> backward(const vector<float>& dy, const vector<float>& x) {
        vector<float> dx(in, 0.0f);
        for (int o = 0; o < out; ++o) {
            for (int i = 0; i < in; ++i) {
                dw[o * in + i] += dy[o] * x[i];
                dx[i] += w[o * in + i] * dy[o];
            }
        }
        return dx;
    }

    void zero_grad() {
        std::fill(dw.begin(), dw.end(), 0.0f);
    }

    void step(float lr, int batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] -= scale * dw[i];
        }
    }
};

struct Normalized {
    vector<float> unit;
    float norm = 1.0f;
};

static Normalized normalize(const vector<float>& x) {
    Normalized result;
    float sum = 0.0f;
    for (float value : x) {
        sum += value * value;
    }
    result.norm = std::sqrt(std::max(sum, 1e-8f));
    result.unit.resize(x.size(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        result.unit[i] = x[i] / result.norm;
    }
    return result;
}

static vector<float> normalize_backward(
    const vector<float>& grad_unit,
    const vector<float>& unit,
    float norm
) {
    vector<float> grad_raw(grad_unit.size(), 0.0f);
    const float projection = dot(grad_unit, unit);
    for (size_t i = 0; i < grad_unit.size(); ++i) {
        grad_raw[i] = (grad_unit[i] - unit[i] * projection) / norm;
    }
    return grad_raw;
}

struct Model {
    Linear audio_shared{DA, DS, 101};
    Linear audio_unique{DA, DU, 102};
    Linear env_shared{DE, DS, 201};
    Linear env_unique{DE, DU, 202};
    float orth_weight = 0.0f;

    vector<vector<float>> sa;
    vector<vector<float>> se;
    vector<vector<float>> ua;
    vector<vector<float>> ue;
    vector<vector<float>> sa_unit;
    vector<vector<float>> se_unit;
    vector<vector<float>> ua_unit;
    vector<vector<float>> ue_unit;
    vector<Normalized> nsa;
    vector<Normalized> nse;
    vector<Normalized> nua;
    vector<Normalized> nue;
    vector<vector<float>> dsa_unit;
    vector<vector<float>> dse_unit;
    vector<vector<float>> dua_unit;
    vector<vector<float>> due_unit;

    Model(float orth, int seed) : orth_weight(orth) {
        (void)seed;
    }

    float compute_loss_and_grads(const vector<Sample>& data, const vector<int>& index) {
        const int n = static_cast<int>(index.size());
        sa.assign(n, {});
        se.assign(n, {});
        ua.assign(n, {});
        ue.assign(n, {});
        sa_unit.assign(n, {});
        se_unit.assign(n, {});
        ua_unit.assign(n, {});
        ue_unit.assign(n, {});
        nsa.assign(n, {});
        nse.assign(n, {});
        nua.assign(n, {});
        nue.assign(n, {});

        for (int i = 0; i < n; ++i) {
            const Sample& s = data[index[i]];
            sa[i] = audio_shared.forward(s.audio);
            ua[i] = audio_unique.forward(s.audio);
            se[i] = env_shared.forward(s.env);
            ue[i] = env_unique.forward(s.env);
            nsa[i] = normalize(sa[i]);
            nse[i] = normalize(se[i]);
            nua[i] = normalize(ua[i]);
            nue[i] = normalize(ue[i]);
            sa_unit[i] = nsa[i].unit;
            se_unit[i] = nse[i].unit;
            ua_unit[i] = nua[i].unit;
            ue_unit[i] = nue[i].unit;
        }

        vector<vector<float>> logits(n, vector<float>(n, 0.0f));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                logits[i][j] = dot(sa_unit[i], se_unit[j]) / TAU;
            }
        }

        vector<vector<float>> p_row(n, vector<float>(n, 0.0f));
        vector<vector<float>> p_col(n, vector<float>(n, 0.0f));
        float loss_row = 0.0f;
        float loss_col = 0.0f;

        for (int i = 0; i < n; ++i) {
            float max_logit = -1e30f;
            for (int j = 0; j < n; ++j) max_logit = std::max(max_logit, logits[i][j]);
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                p_row[i][j] = std::exp(logits[i][j] - max_logit);
                sum += p_row[i][j];
            }
            for (int j = 0; j < n; ++j) p_row[i][j] /= sum;
            loss_row += -std::log(std::max(p_row[i][i], 1e-8f));
        }

        for (int j = 0; j < n; ++j) {
            float max_logit = -1e30f;
            for (int i = 0; i < n; ++i) max_logit = std::max(max_logit, logits[i][j]);
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) {
                p_col[i][j] = std::exp(logits[i][j] - max_logit);
                sum += p_col[i][j];
            }
            for (int i = 0; i < n; ++i) p_col[i][j] /= sum;
            loss_col += -std::log(std::max(p_col[j][j], 1e-8f));
        }

        loss_row /= static_cast<float>(n);
        loss_col /= static_cast<float>(n);
        const float loss = 0.5f * (loss_row + loss_col);

        dsa_unit.assign(n, vector<float>(DS, 0.0f));
        dse_unit.assign(n, vector<float>(DS, 0.0f));
        dua_unit.assign(n, vector<float>(DU, 0.0f));
        due_unit.assign(n, vector<float>(DU, 0.0f));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const float indicator = (i == j) ? 1.0f : 0.0f;
                const float grad_logit =
                    0.5f * ((p_row[i][j] - indicator) / static_cast<float>(n))
                  + 0.5f * ((p_col[i][j] - indicator) / static_cast<float>(n));

                for (int d = 0; d < DS; ++d) {
                    dsa_unit[i][d] += grad_logit * se_unit[j][d] / TAU;
                    dse_unit[j][d] += grad_logit * sa_unit[i][d] / TAU;
                }
            }
        }

        float orth_loss = 0.0f;
        if (orth_weight > 0.0f) {
            for (int i = 0; i < n; ++i) {
                const float oa = dot(sa_unit[i], ua_unit[i]);
                const float oe = dot(se_unit[i], ue_unit[i]);
                orth_loss += oa * oa + oe * oe;

                const float goa = 2.0f * oa / static_cast<float>(n);
                const float goe = 2.0f * oe / static_cast<float>(n);
                for (int d = 0; d < DS; ++d) {
                    dsa_unit[i][d] += goa * ua_unit[i][d];
                    dse_unit[i][d] += goe * ue_unit[i][d];
                }
                for (int d = 0; d < DU; ++d) {
                    dua_unit[i][d] += goa * sa_unit[i][d];
                    due_unit[i][d] += goe * se_unit[i][d];
                }
            }
            orth_loss /= static_cast<float>(n);
        }

        return loss + orth_weight * orth_loss;
    }

    void backward(const vector<Sample>& data, const vector<int>& index) {
        const int n = static_cast<int>(index.size());
        for (int i = 0; i < n; ++i) {
            const Sample& s = data[index[i]];

            const vector<float> dsa = normalize_backward(dsa_unit[i], sa_unit[i], nsa[i].norm);
            const vector<float> dse = normalize_backward(dse_unit[i], se_unit[i], nse[i].norm);
            const vector<float> dua = normalize_backward(dua_unit[i], ua_unit[i], nua[i].norm);
            const vector<float> due = normalize_backward(due_unit[i], ue_unit[i], nue[i].norm);

            audio_shared.backward(dsa, s.audio);
            audio_unique.backward(dua, s.audio);
            env_shared.backward(dse, s.env);
            env_unique.backward(due, s.env);
        }
    }

    void zero_grad() {
        audio_shared.zero_grad();
        audio_unique.zero_grad();
        env_shared.zero_grad();
        env_unique.zero_grad();
    }

    void step(float lr, int batch_size) {
        audio_shared.step(lr, batch_size);
        audio_unique.step(lr, batch_size);
        env_shared.step(lr, batch_size);
        env_unique.step(lr, batch_size);
    }
};

static float retrieval_accuracy(Model& model, const vector<Sample>& data) {
    const int n = static_cast<int>(data.size());
    vector<vector<float>> a(n);
    vector<vector<float>> e(n);
    for (int i = 0; i < n; ++i) {
        a[i] = normalize(model.audio_shared.forward(data[i].audio)).unit;
        e[i] = normalize(model.env_shared.forward(data[i].env)).unit;
    }

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        int best = 0;
        float best_sim = -1e30f;
        for (int j = 0; j < n; ++j) {
            const float sim = dot(a[i], e[j]);
            if (sim > best_sim) {
                best_sim = sim;
                best = j;
            }
        }
        if (best == i) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(n);
}

static void train(Model& model, const vector<Sample>& data, int epochs, float lr, int batch_size) {
    std::mt19937 rng(12345);
    vector<int> order(data.size());
    for (int i = 0; i < static_cast<int>(data.size()); ++i) order[i] = i;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t start = 0; start < order.size(); start += batch_size) {
            const size_t end = std::min(order.size(), start + batch_size);
            vector<int> batch(order.begin() + start, order.begin() + end);
            model.zero_grad();
            model.compute_loss_and_grads(data, batch);
            model.backward(data, batch);
            model.step(lr, static_cast<int>(batch.size()));
        }
    }
}

int main() {
    constexpr int train_size = 2500;
    constexpr int test_size = 200;
    constexpr int epochs = 400;
    constexpr float lr = 0.10f;
    constexpr int batch_size = 128;

    const MixMatrix a_shared = make_mix(DA, DS, 11);
    const MixMatrix a_unique = make_mix(DA, DU, 12);
    const MixMatrix e_shared = make_mix(DE, DS, 21);
    const MixMatrix e_unique = make_mix(DE, DU, 22);

    std::mt19937 rng(20261012);
    vector<Sample> train_data;
    train_data.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        train_data.push_back(make_sample(
            rng, a_shared, a_unique, e_shared, e_unique,
            0.12f, 0.10f, 0.10f
        ));
    }
    // Contrastive pairs need both modalities.
    train_data.erase(
        std::remove_if(
            train_data.begin(),
            train_data.end(),
            [](const Sample& s) { return s.paired == 0; }
        ),
        train_data.end()
    );

    vector<Sample> test_clean;
    vector<Sample> test_noisy;
    for (int i = 0; i < test_size; ++i) {
        test_clean.push_back(make_sample(
            rng, a_shared, a_unique, e_shared, e_unique,
            0.0f, 0.03f, 0.03f
        ));
        test_noisy.push_back(make_sample(
            rng, a_shared, a_unique, e_shared, e_unique,
            0.0f, 0.35f, 0.35f
        ));
    }

    Model info_nce_only(0.0f, 777);
    Model factorized(0.10f, 777);

    train(info_nce_only, train_data, epochs, lr, batch_size);
    train(factorized, train_data, epochs, lr, batch_size);

    cout << "Cross-modal retrieval (audio shared -> environment shared)\n";
    cout << std::fixed << std::setprecision(4);
    cout << "Method             clean    noisy\n";
    cout << "----------------------------------\n";
    cout << std::left << std::setw(19) << "InfoNCE only"
         << std::setw(9) << retrieval_accuracy(info_nce_only, test_clean)
         << std::setw(9) << retrieval_accuracy(info_nce_only, test_noisy) << '\n';
    cout << std::left << std::setw(19) << "Factorized (FCL)"
         << std::setw(9) << retrieval_accuracy(factorized, test_clean)
         << std::setw(9) << retrieval_accuracy(factorized, test_noisy) << '\n';
    return 0;
}
