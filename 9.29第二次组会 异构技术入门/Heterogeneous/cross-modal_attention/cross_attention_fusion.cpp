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

static constexpr int D = 6;   // feature dimension of each time step
static constexpr int DM = 8;  // attention projection dimension
static constexpr int TA = 6;  // audio sequence length
static constexpr int TE = 3;  // environment sequence length

using Seq = vector<vector<float>>;

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
    Seq audio;
    Seq env;
    float qa = 0.0f;
    float qe = 0.0f;
    int ma = 1;
    int me = 1;
    int y = 0;
};

static Seq zeros_seq(int length, int dim) {
    return Seq(length, vector<float>(dim, 0.0f));
}

static vector<float> mean_seq(const Seq& seq) {
    const int length = static_cast<int>(seq.size());
    const int dim = static_cast<int>(seq[0].size());
    vector<float> mean(dim, 0.0f);
    for (const vector<float>& row : seq) {
        for (int j = 0; j < dim; ++j) {
            mean[j] += row[j];
        }
    }
    for (float& value : mean) {
        value /= static_cast<float>(length);
    }
    return mean;
}

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
    s.audio.assign(TA, vector<float>(D, 0.0f));
    s.env.assign(TE, vector<float>(D, 0.0f));

    // Latent events happen at different clocks. Audio is sampled every second,
    // while the environment is sampled every two seconds.
    const float ta = runif(rng, 0.0f, static_cast<float>(TA - 1));
    const float te = runif(rng, 0.0f, static_cast<float>((TE - 1) * 2));
    const float aa = runif(rng, 0.4f, 1.4f);
    const float ae = runif(rng, 0.4f, 1.4f);
    const float alignment = std::exp(-std::abs(ta - te));
    const float risk_logit =
        4.0f * aa * ae * alignment - 1.2f + 0.15f * randn(rng);
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

    const float sigma_a = 0.65f + 0.25f * (1.0f - s.qa);
    const float sigma_e = 0.90f + 0.25f * (1.0f - s.qe);

    for (int t = 0; t < TA; ++t) {
        const float dt = static_cast<float>(t) - ta;
        const float time = static_cast<float>(t);
        s.audio[t][0] = aa * std::exp(-0.5f * dt * dt / (sigma_a * sigma_a));
        s.audio[t][1] = 1.0f;
        s.audio[t][2] = std::sin(time);
        s.audio[t][3] = std::cos(time);
        s.audio[t][4] = static_cast<float>(t) / static_cast<float>(TA - 1);
        s.audio[t][5] = 0.05f * randn(rng);
        if (s.ma) {
            s.audio[t][0] += noise_a * randn(rng);
            s.audio[t][2] += noise_a * randn(rng);
        } else {
            std::fill(s.audio[t].begin(), s.audio[t].end(), 0.0f);
        }
    }

    for (int t = 0; t < TE; ++t) {
        const float dt = static_cast<float>(t * 2) - te;
        const float time = static_cast<float>(t * 2);
        s.env[t][0] = ae * std::exp(-0.5f * dt * dt / (sigma_e * sigma_e));
        s.env[t][1] = 1.0f;
        s.env[t][2] = std::sin(time);
        s.env[t][3] = std::cos(time);
        s.env[t][4] = static_cast<float>(t) / static_cast<float>(TE - 1);
        s.env[t][5] = 0.05f * randn(rng);
        if (s.me) {
            s.env[t][0] += noise_e * randn(rng);
            s.env[t][2] += noise_e * randn(rng);
        } else {
            std::fill(s.env[t].begin(), s.env[t].end(), 0.0f);
        }
    }
    return s;
}

static Seq project_forward(
    const Seq& x,
    const vector<float>& weight,
    const vector<float>& bias,
    int in_dim,
    int out_dim
) {
    const int length = static_cast<int>(x.size());
    Seq y(length, vector<float>(out_dim, 0.0f));
    for (int t = 0; t < length; ++t) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = bias[o];
            for (int i = 0; i < in_dim; ++i) {
                sum += weight[o * in_dim + i] * x[t][i];
            }
            y[t][o] = sum;
        }
    }
    return y;
}

static Seq project_backward(
    const Seq& x,
    const vector<float>& weight,
    const Seq& dy,
    vector<float>& dweight,
    vector<float>& dbias,
    int in_dim,
    int out_dim
) {
    const int length = static_cast<int>(x.size());
    Seq dx = zeros_seq(length, in_dim);
    for (int t = 0; t < length; ++t) {
        for (int o = 0; o < out_dim; ++o) {
            const float g = dy[t][o];
            for (int i = 0; i < in_dim; ++i) {
                dweight[o * in_dim + i] += g * x[t][i];
                dx[t][i] += weight[o * in_dim + i] * g;
            }
            dbias[o] += g;
        }
    }
    return dx;
}

static Seq scale_seq(const Seq& x, float scale) {
    Seq y = x;
    for (auto& row : y) {
        for (float& value : row) {
            value *= scale;
        }
    }
    return y;
}

static void attention_forward(
    const Seq& q,
    const Seq& k,
    const Seq& v,
    Seq& attention,
    Seq& output
) {
    const int tq = static_cast<int>(q.size());
    const int tk = static_cast<int>(k.size());
    const int dim = static_cast<int>(q[0].size());
    const float scale = 3.0f / std::sqrt(static_cast<float>(dim));

    attention.assign(tq, vector<float>(tk, 0.0f));
    output.assign(tq, vector<float>(dim, 0.0f));

    for (int i = 0; i < tq; ++i) {
        float max_score = -1e30f;
        for (int j = 0; j < tk; ++j) {
            float score = 0.0f;
            for (int d = 0; d < dim; ++d) {
                score += q[i][d] * k[j][d];
            }
            attention[i][j] = score * scale;
            max_score = std::max(max_score, attention[i][j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < tk; ++j) {
            attention[i][j] = std::exp(attention[i][j] - max_score);
            sum += attention[i][j];
        }
        for (int j = 0; j < tk; ++j) {
            attention[i][j] /= sum;
        }

        for (int j = 0; j < tk; ++j) {
            for (int d = 0; d < dim; ++d) {
                output[i][d] += attention[i][j] * v[j][d];
            }
        }
    }
}

static void attention_backward(
    const Seq& q,
    const Seq& k,
    const Seq& v,
    const Seq& attention,
    const Seq& doutput,
    Seq& dq,
    Seq& dk,
    Seq& dv
) {
    const int tq = static_cast<int>(q.size());
    const int tk = static_cast<int>(k.size());
    const int dim = static_cast<int>(q[0].size());
    const float scale = 3.0f / std::sqrt(static_cast<float>(dim));

    for (int i = 0; i < tq; ++i) {
        vector<float> da(tk, 0.0f);
        float inner = 0.0f;
        for (int j = 0; j < tk; ++j) {
            for (int d = 0; d < dim; ++d) {
                da[j] += doutput[i][d] * v[j][d];
                dv[j][d] += attention[i][j] * doutput[i][d];
            }
            inner += da[j] * attention[i][j];
        }

        for (int j = 0; j < tk; ++j) {
            const float ds = attention[i][j] * (da[j] - inner);
            for (int d = 0; d < dim; ++d) {
                dq[i][d] += ds * k[j][d] * scale;
                dk[j][d] += ds * q[i][d] * scale;
            }
        }
    }
}

struct CrossAttentionModel {
    vector<float> wq_a, wk_e, wv_e, bq_a, bk_e, bv_e;
    vector<float> wq_e, wk_a, wv_a, bq_e, bk_a, bv_a;
    vector<float> dwq_a, dwk_e, dwv_e, dbq_a, dbk_e, dbv_e;
    vector<float> dwq_e, dwk_a, dwv_a, dbq_e, dbk_a, dbv_a;
    vector<float> wo, dwo;
    float bo = 0.0f;
    float dbo = 0.0f;

    Seq audio_obs, env_obs;
    Seq q_a_raw, k_e_raw, v_e_raw;
    Seq q_e_raw, k_a_raw, v_a_raw;
    Seq q_a, k_e, v_e;
    Seq q_e, k_a, v_a;
    Seq attention_a, attention_e;
    Seq output_a, output_e;
    vector<float> pooled_a, pooled_e;
    vector<float> unimodal_a, unimodal_e;
    vector<float> fused;
    float qa = 0.0f;
    float qe = 0.0f;

    explicit CrossAttentionModel(int seed) {
        const size_t proj = static_cast<size_t>(DM * D);
        wq_a.assign(proj, 0.0f);
        wk_e.assign(proj, 0.0f);
        wv_e.assign(proj, 0.0f);
        wq_e.assign(proj, 0.0f);
        wk_a.assign(proj, 0.0f);
        wv_a.assign(proj, 0.0f);
        bq_a.assign(DM, 0.0f);
        bk_e.assign(DM, 0.0f);
        bv_e.assign(DM, 0.0f);
        bq_e.assign(DM, 0.0f);
        bk_a.assign(DM, 0.0f);
        bv_a.assign(DM, 0.0f);

        dwq_a.assign(proj, 0.0f);
        dwk_e.assign(proj, 0.0f);
        dwv_e.assign(proj, 0.0f);
        dwq_e.assign(proj, 0.0f);
        dwk_a.assign(proj, 0.0f);
        dwv_a.assign(proj, 0.0f);
        dbq_a.assign(DM, 0.0f);
        dbk_e.assign(DM, 0.0f);
        dbv_e.assign(DM, 0.0f);
        dbq_e.assign(DM, 0.0f);
        dbk_a.assign(DM, 0.0f);
        dbv_a.assign(DM, 0.0f);

        wo.assign(2 * DM + 2 * D + 2, 0.0f);
        dwo.assign(wo.size(), 0.0f);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> init(-0.50f, 0.50f);
        for (float& value : wq_a) value = init(rng);
        for (float& value : wk_e) value = init(rng);
        for (float& value : wv_e) value = init(rng);
        for (float& value : wq_e) value = init(rng);
        for (float& value : wk_a) value = init(rng);
        for (float& value : wv_a) value = init(rng);
        for (float& value : wo) value = init(rng);
    }

    float forward(const Sample& s) {
        audio_obs = s.audio;
        env_obs = s.env;
        qa = s.ma ? s.qa : 0.0f;
        qe = s.me ? s.qe : 0.0f;

        q_a_raw = project_forward(audio_obs, wq_a, bq_a, D, DM);
        k_e_raw = project_forward(env_obs, wk_e, bk_e, D, DM);
        v_e_raw = project_forward(env_obs, wv_e, bv_e, D, DM);
        q_e_raw = project_forward(env_obs, wq_e, bq_e, D, DM);
        k_a_raw = project_forward(audio_obs, wk_a, bk_a, D, DM);
        v_a_raw = project_forward(audio_obs, wv_a, bv_a, D, DM);

        q_a = scale_seq(q_a_raw, qa);
        k_e = scale_seq(k_e_raw, qe);
        v_e = scale_seq(v_e_raw, qe);
        q_e = scale_seq(q_e_raw, qe);
        k_a = scale_seq(k_a_raw, qa);
        v_a = scale_seq(v_a_raw, qa);

        attention_forward(q_a, k_e, v_e, attention_a, output_a);
        attention_forward(q_e, k_a, v_a, attention_e, output_e);

        pooled_a = mean_seq(output_a);
        pooled_e = mean_seq(output_e);
        unimodal_a = mean_seq(audio_obs);
        unimodal_e = mean_seq(env_obs);
        for (float& value : unimodal_a) value *= qa;
        for (float& value : unimodal_e) value *= qe;

        fused.clear();
        fused.insert(fused.end(), pooled_a.begin(), pooled_a.end());
        fused.insert(fused.end(), pooled_e.begin(), pooled_e.end());
        fused.insert(fused.end(), unimodal_a.begin(), unimodal_a.end());
        fused.insert(fused.end(), unimodal_e.begin(), unimodal_e.end());
        fused.push_back(qa);
        fused.push_back(qe);

        return dot(wo, fused) + bo;
    }

    void backward(float grad) {
        vector<float> dz(fused.size(), 0.0f);
        for (size_t i = 0; i < wo.size(); ++i) {
            dwo[i] += grad * fused[i];
            dz[i] = grad * wo[i];
        }
        dbo += grad;

        vector<float> d_pooled_a(dz.begin(), dz.begin() + DM);
        vector<float> d_pooled_e(dz.begin() + DM, dz.begin() + 2 * DM);
        vector<float> d_unimodal_a(
            dz.begin() + 2 * DM,
            dz.begin() + 2 * DM + D
        );
        vector<float> d_unimodal_e(
            dz.begin() + 2 * DM + D,
            dz.begin() + 2 * DM + 2 * D
        );

        Seq d_output_a = zeros_seq(TA, DM);
        Seq d_output_e = zeros_seq(TE, DM);
        for (int t = 0; t < TA; ++t) {
            for (int j = 0; j < DM; ++j) {
                d_output_a[t][j] = d_pooled_a[j] / static_cast<float>(TA);
            }
        }
        for (int t = 0; t < TE; ++t) {
            for (int j = 0; j < DM; ++j) {
                d_output_e[t][j] = d_pooled_e[j] / static_cast<float>(TE);
            }
        }

        Seq dq_a = zeros_seq(TA, DM);
        Seq dk_e = zeros_seq(TE, DM);
        Seq dv_e = zeros_seq(TE, DM);
        Seq dq_e = zeros_seq(TE, DM);
        Seq dk_a = zeros_seq(TA, DM);
        Seq dv_a = zeros_seq(TA, DM);

        attention_backward(q_a, k_e, v_e, attention_a, d_output_a, dq_a, dk_e, dv_e);
        attention_backward(q_e, k_a, v_a, attention_e, d_output_e, dq_e, dk_a, dv_a);

        Seq dq_a_raw = scale_seq(dq_a, qa);
        Seq dk_e_raw = scale_seq(dk_e, qe);
        Seq dv_e_raw = scale_seq(dv_e, qe);
        Seq dq_e_raw = scale_seq(dq_e, qe);
        Seq dk_a_raw = scale_seq(dk_a, qa);
        Seq dv_a_raw = scale_seq(dv_a, qa);

        Seq d_audio = zeros_seq(TA, D);
        Seq d_env = zeros_seq(TE, D);

        project_backward(audio_obs, wq_a, dq_a_raw, dwq_a, dbq_a, D, DM);
        project_backward(env_obs, wk_e, dk_e_raw, dwk_e, dbk_e, D, DM);
        project_backward(env_obs, wv_e, dv_e_raw, dwv_e, dbv_e, D, DM);
        project_backward(env_obs, wq_e, dq_e_raw, dwq_e, dbq_e, D, DM);
        project_backward(audio_obs, wk_a, dk_a_raw, dwk_a, dbk_a, D, DM);
        project_backward(audio_obs, wv_a, dv_a_raw, dwv_a, dbv_a, D, DM);

        for (int t = 0; t < TA; ++t) {
            for (int d = 0; d < D; ++d) {
                d_audio[t][d] += d_unimodal_a[d] * qa / static_cast<float>(TA);
            }
        }
        for (int t = 0; t < TE; ++t) {
            for (int d = 0; d < D; ++d) {
                d_env[t][d] += d_unimodal_e[d] * qe / static_cast<float>(TE);
            }
        }

        // The projection functions accumulate parameter gradients. The
        // returned input gradients are not needed by the outer model here.
        // They would be used if an upstream encoder existed.
    }

    void zero_grad() {
        std::fill(dwq_a.begin(), dwq_a.end(), 0.0f);
        std::fill(dwk_e.begin(), dwk_e.end(), 0.0f);
        std::fill(dwv_e.begin(), dwv_e.end(), 0.0f);
        std::fill(dwq_e.begin(), dwq_e.end(), 0.0f);
        std::fill(dwk_a.begin(), dwk_a.end(), 0.0f);
        std::fill(dwv_a.begin(), dwv_a.end(), 0.0f);
        std::fill(dbq_a.begin(), dbq_a.end(), 0.0f);
        std::fill(dbk_e.begin(), dbk_e.end(), 0.0f);
        std::fill(dbv_e.begin(), dbv_e.end(), 0.0f);
        std::fill(dbq_e.begin(), dbq_e.end(), 0.0f);
        std::fill(dbk_a.begin(), dbk_a.end(), 0.0f);
        std::fill(dbv_a.begin(), dbv_a.end(), 0.0f);
        std::fill(dwo.begin(), dwo.end(), 0.0f);
        dbo = 0.0f;
    }

    void step(float lr, int batch_size) {
        const float scale = lr / static_cast<float>(batch_size);
        auto update = [scale](vector<float>& param, const vector<float>& grad) {
            for (size_t i = 0; i < param.size(); ++i) {
                param[i] -= scale * grad[i];
            }
        };
        update(wq_a, dwq_a);
        update(wk_e, dwk_e);
        update(wv_e, dwv_e);
        update(wq_e, dwq_e);
        update(wk_a, dwk_a);
        update(wv_a, dwv_a);
        update(bq_a, dbq_a);
        update(bk_e, dbk_e);
        update(bv_e, dbv_e);
        update(bq_e, dbq_e);
        update(bk_a, dbk_a);
        update(bv_a, dbv_a);
        update(wo, dwo);
        bo -= scale * dbo;
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};

static Metrics evaluate(CrossAttentionModel& model, const vector<Sample>& data) {
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

static void train(CrossAttentionModel& model, vector<Sample>& data, int epochs, float lr, int batch_size) {
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
        case 0: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.9f, 0.05f, 0.05f);
        case 1: return make_sample(rng, 1.0f, 0.0f, -1.0f, 0.9f, 0.05f, 0.05f);
        case 2: return make_sample(rng, 0.0f, 1.0f, 0.9f, -1.0f, 0.05f, 0.05f);
        case 3: return make_sample(rng, 0.0f, 0.0f, 0.20f, 0.9f, 0.80f, 0.05f);
        default: return make_sample(rng, 0.0f, 0.0f, 0.9f, 0.20f, 0.05f, 0.80f);
    }
}

int main() {
    constexpr int train_size = 4000;
    constexpr int test_size = 800;
    constexpr int epochs = 600;
    constexpr float lr = 0.05f;
    constexpr int batch_size = 32;

    std::mt19937 rng(20261011);
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

    CrossAttentionModel model(777);
    train(model, train_data, epochs, lr, batch_size);

    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };

    cout << "Cross-modal attention: audio<->environment, softmax QK^T/sqrt(d)\n";
    cout << "Regime          Acc     F1      Brier\n";
    cout << "----------------------------------------\n";
    cout << std::fixed << std::setprecision(4);

    for (int regime = 0; regime < 5; ++regime) {
        const Metrics m = evaluate(model, test_data[regime]);
        cout << std::left << std::setw(16) << names[regime]
             << std::setw(8) << m.accuracy
             << std::setw(8) << m.f1
             << std::setw(8) << m.brier << '\n';
    }
    return 0;
}
