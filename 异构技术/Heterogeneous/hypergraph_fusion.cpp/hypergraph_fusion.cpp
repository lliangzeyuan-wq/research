#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::string;
using std::vector;

static constexpr int D = 6;
static constexpr int L = 2;
static constexpr int B = 2;

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
    int barn = 0;
    int y = 0;
};

static Sample make_sample(
    std::mt19937& rng,
    int barn,
    float miss_a = 0.20f,
    float miss_e = 0.20f,
    float qa_override = -1.0f,
    float qe_override = -1.0f,
    float noise_a = 0.08f,
    float noise_e = 0.08f
) {
    Sample s;
    s.barn = barn;
    const float la0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float la1 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le0 = std::clamp(randn(rng), -3.0f, 3.0f);
    const float le1 = std::clamp(randn(rng), -3.0f, 3.0f);

    const float barn_effect = (barn == 0) ? 0.65f : -0.65f;
    const float risk_logit =
        barn_effect +
        0.9f * la0 + 0.9f * le0 + 2.0f * la0 * le0 +
        0.10f * la1 - 0.08f * le1 - 0.30f;
    s.y = (runif(rng, 0.0f, 1.0f) < sigmoid(risk_logit)) ? 1 : 0;

    s.ma = (runif(rng, 0.0f, 1.0f) > miss_a) ? 1 : 0;
    s.me = (runif(rng, 0.0f, 1.0f) > miss_e) ? 1 : 0;
    s.qa = (qa_override >= 0.0f) ? qa_override : runif(rng, 0.45f, 1.0f);
    s.qe = (qe_override >= 0.0f) ? qe_override : runif(rng, 0.45f, 1.0f);
    if (!s.ma) s.qa = 0.0f;
    if (!s.me) s.qe = 0.0f;

    const float na = noise_a + 0.35f * (1.0f - s.qa);
    const float ne = noise_e + 0.35f * (1.0f - s.qe);
    s.a0 = s.ma ? (la0 + na * randn(rng)) : 0.0f;
    s.a1 = s.ma ? (la1 + na * randn(rng)) : 0.0f;
    s.e0 = s.me ? (le0 + ne * randn(rng)) : 0.0f;
    s.e1 = s.me ? (le1 + ne * randn(rng)) : 0.0f;
    return s;
}

struct Hypergraph {
    int n_samples = 0;
    int n_nodes = 0;
    int n_edges = 0;
    vector<vector<float>> x;
    vector<int> sample_nodes;
    vector<vector<std::pair<int, float>>> edge_nodes;
    vector<vector<std::pair<int, float>>> node_edges;
    vector<float> node_degree;
    vector<float> edge_degree;
};

static void add_incidence(Hypergraph& g, int node, int edge, float weight) {
    if (weight <= 0.0f) return;
    g.edge_nodes[edge].push_back({node, weight});
    g.node_edges[node].push_back({edge, weight});
    g.node_degree[node] += weight;
    g.edge_degree[edge] += weight;
}

static Hypergraph build_hypergraph(const vector<Sample>& samples) {
    Hypergraph g;
    g.n_samples = static_cast<int>(samples.size());
    g.n_nodes = 3 * g.n_samples;
    g.n_edges = g.n_samples + B + 2;  // sample, barn, audio-global, env-global
    g.x.assign(g.n_nodes, vector<float>(D, 0.0f));
    g.sample_nodes.resize(g.n_samples);
    g.edge_nodes.assign(g.n_edges, {});
    g.node_edges.assign(g.n_nodes, {});
    g.node_degree.assign(g.n_nodes, 0.0f);
    g.edge_degree.assign(g.n_edges, 0.0f);

    for (int i = 0; i < g.n_samples; ++i) {
        const Sample& s = samples[i];
        const int sn = i;
        const int an = g.n_samples + i;
        const int en = 2 * g.n_samples + i;
        g.sample_nodes[i] = sn;

        g.x[sn] = {
            s.a0 * static_cast<float>(s.ma),
            s.a1 * static_cast<float>(s.ma),
            s.e0 * static_cast<float>(s.me),
            s.e1 * static_cast<float>(s.me),
            static_cast<float>(s.ma),
            static_cast<float>(s.me),
        };
        g.x[an] = {s.a0, s.a1, static_cast<float>(s.ma), s.qa, 0.0f, 0.0f};
        g.x[en] = {s.e0, s.e1, static_cast<float>(s.me), s.qe, 0.0f, 0.0f};
    }

    const int audio_global_edge = g.n_samples + B;
    const int env_global_edge = g.n_samples + B + 1;

    for (int i = 0; i < g.n_samples; ++i) {
        const Sample& s = samples[i];
        const int sn = i;
        const int an = g.n_samples + i;
        const int en = 2 * g.n_samples + i;
        const int sample_edge = i;
        const int barn_edge = g.n_samples + s.barn;

        add_incidence(g, sn, sample_edge, 1.0f);
        if (s.ma) {
            add_incidence(g, an, sample_edge, s.qa);
            add_incidence(g, an, audio_global_edge, s.qa);
        }
        if (s.me) {
            add_incidence(g, en, sample_edge, s.qe);
            add_incidence(g, en, env_global_edge, s.qe);
        }
        add_incidence(g, sn, barn_edge, 1.0f);
    }

    for (float& value : g.node_degree) value = std::max(value, 1e-6f);
    for (float& value : g.edge_degree) value = std::max(value, 1e-6f);
    return g;
}

struct HypergraphModel {
    vector<float> wn;
    vector<float> dwn;
    vector<float> bn;
    vector<float> dbn;
    vector<float> we;
    vector<float> dwe;
    vector<float> be;
    vector<float> dbe;
    vector<float> ws;
    vector<float> dws;
    vector<float> bs;
    vector<float> dbs;
    vector<float> wo;
    vector<float> dwo;
    float bo = 0.0f;
    float dbo = 0.0f;

    vector<vector<vector<float>>> history_in;
    vector<vector<vector<float>>> history_edge;
    vector<vector<vector<float>>> history_out;
    vector<vector<float>> final_h;
    vector<float> sample_logits;

    explicit HypergraphModel(int seed) {
        const int mat = L * D * D;
        const int vec = L * D;
        wn.assign(mat, 0.0f);
        dwn.assign(mat, 0.0f);
        we.assign(mat, 0.0f);
        dwe.assign(mat, 0.0f);
        ws.assign(mat, 0.0f);
        dws.assign(mat, 0.0f);
        bn.assign(vec, 0.0f);
        dbn.assign(vec, 0.0f);
        be.assign(vec, 0.0f);
        dbe.assign(vec, 0.0f);
        bs.assign(vec, 0.0f);
        dbs.assign(vec, 0.0f);
        wo.assign(D, 0.0f);
        dwo.assign(D, 0.0f);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> init(-0.35f, 0.35f);
        for (float& value : wn) value = init(rng);
        for (float& value : we) value = init(rng);
        for (float& value : ws) value = init(rng);
        for (float& value : wo) value = init(rng);
    }

    int idx(int layer, int row, int col) const {
        return (layer * D + row) * D + col;
    }

    int vidx(int layer, int row) const {
        return layer * D + row;
    }

    vector<float> forward(const Hypergraph& g) {
        vector<vector<float>> h = g.x;
        history_in.assign(L, {});
        history_edge.assign(L, {});
        history_out.assign(L, {});

        for (int layer = 0; layer < L; ++layer) {
            history_in[layer] = h;
            vector<vector<float>> edge_pre(g.n_edges, vector<float>(D, 0.0f));
            vector<vector<float>> edge(g.n_edges, vector<float>(D, 0.0f));

            for (int e = 0; e < g.n_edges; ++e) {
                const float inv_degree = 1.0f / g.edge_degree[e];
                for (int r = 0; r < D; ++r) {
                    edge_pre[e][r] = bn[vidx(layer, r)];
                }
                for (const auto& item : g.edge_nodes[e]) {
                    const int i = item.first;
                    const float w = item.second * inv_degree;
                    for (int r = 0; r < D; ++r) {
                        float sum = 0.0f;
                        for (int c = 0; c < D; ++c) {
                            sum += wn[idx(layer, r, c)] * h[i][c];
                        }
                        edge_pre[e][r] += w * sum;
                    }
                }
                for (int r = 0; r < D; ++r) {
                    edge[e][r] = std::tanh(edge_pre[e][r]);
                }
            }

            vector<vector<float>> next(g.n_nodes, vector<float>(D, 0.0f));
            for (int i = 0; i < g.n_nodes; ++i) {
                const float inv_degree = 1.0f / g.node_degree[i];
                vector<float> pre(D, 0.0f);
                for (int r = 0; r < D; ++r) {
                    pre[r] = bs[vidx(layer, r)] + be[vidx(layer, r)];
                    for (int c = 0; c < D; ++c) {
                        pre[r] += ws[idx(layer, r, c)] * h[i][c];
                    }
                }
                for (const auto& item : g.node_edges[i]) {
                    const int e = item.first;
                    const float w = item.second * inv_degree;
                    for (int r = 0; r < D; ++r) {
                        float sum = 0.0f;
                        for (int c = 0; c < D; ++c) {
                            sum += we[idx(layer, r, c)] * edge[e][c];
                        }
                        pre[r] += w * sum;
                    }
                }
                for (int r = 0; r < D; ++r) {
                    next[i][r] = std::tanh(pre[r]);
                }
            }

            history_edge[layer] = edge;
            history_out[layer] = next;
            h = next;
        }

        final_h = h;
        sample_logits.assign(g.n_samples, 0.0f);
        for (int i = 0; i < g.n_samples; ++i) {
            const int node = g.sample_nodes[i];
            for (int r = 0; r < D; ++r) {
                sample_logits[i] += wo[r] * final_h[node][r];
            }
            sample_logits[i] += bo;
        }
        return sample_logits;
    }

    void backward(const Hypergraph& g, const vector<float>& dlogits) {
        vector<vector<float>> dh(g.n_nodes, vector<float>(D, 0.0f));
        for (int i = 0; i < g.n_samples; ++i) {
            const int node = g.sample_nodes[i];
            for (int r = 0; r < D; ++r) {
                dwo[r] += dlogits[i] * final_h[node][r];
                dh[node][r] += dlogits[i] * wo[r];
            }
            dbo += dlogits[i];
        }

        for (int layer = L - 1; layer >= 0; --layer) {
            vector<vector<float>> dnode_pre(g.n_nodes, vector<float>(D, 0.0f));
            vector<vector<float>> dh_in(g.n_nodes, vector<float>(D, 0.0f));
            vector<vector<float>> d_edge(g.n_edges, vector<float>(D, 0.0f));

            for (int i = 0; i < g.n_nodes; ++i) {
                for (int r = 0; r < D; ++r) {
                    const float out_r = history_out[layer][i][r];
                    dnode_pre[i][r] = dh[i][r] * (1.0f - out_r * out_r);
                    dbs[vidx(layer, r)] += dnode_pre[i][r];
                    dbe[vidx(layer, r)] += dnode_pre[i][r];
                    for (int c = 0; c < D; ++c) {
                        dws[idx(layer, r, c)] +=
                            dnode_pre[i][r] * history_in[layer][i][c];
                        dh_in[i][c] +=
                            ws[idx(layer, r, c)] * dnode_pre[i][r];
                    }
                }

                const float inv_degree = 1.0f / g.node_degree[i];
                for (const auto& item : g.node_edges[i]) {
                    const int e = item.first;
                    const float w = item.second * inv_degree;
                    for (int r = 0; r < D; ++r) {
                        const float gv = dnode_pre[i][r] * w;
                        for (int c = 0; c < D; ++c) {
                            dwe[idx(layer, r, c)] +=
                                gv * history_edge[layer][e][c];
                            d_edge[e][c] +=
                                we[idx(layer, r, c)] * gv;
                        }
                    }
                }
            }

            vector<vector<float>> d_edge_pre(g.n_edges, vector<float>(D, 0.0f));
            for (int e = 0; e < g.n_edges; ++e) {
                for (int r = 0; r < D; ++r) {
                    const float edge_out = history_edge[layer][e][r];
                    d_edge_pre[e][r] = d_edge[e][r] * (1.0f - edge_out * edge_out);
                    dbn[vidx(layer, r)] += d_edge_pre[e][r];
                }

                const float inv_degree = 1.0f / g.edge_degree[e];
                for (const auto& item : g.edge_nodes[e]) {
                    const int i = item.first;
                    const float w = item.second * inv_degree;
                    for (int r = 0; r < D; ++r) {
                        const float gv = d_edge_pre[e][r] * w;
                        for (int c = 0; c < D; ++c) {
                            dwn[idx(layer, r, c)] +=
                                gv * history_in[layer][i][c];
                            dh_in[i][c] +=
                                wn[idx(layer, r, c)] * gv;
                        }
                    }
                }
            }

            dh = dh_in;
        }
    }

    void zero_grad() {
        std::fill(dwn.begin(), dwn.end(), 0.0f);
        std::fill(dwe.begin(), dwe.end(), 0.0f);
        std::fill(dws.begin(), dws.end(), 0.0f);
        std::fill(dbn.begin(), dbn.end(), 0.0f);
        std::fill(dbe.begin(), dbe.end(), 0.0f);
        std::fill(dbs.begin(), dbs.end(), 0.0f);
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
        update(wn, dwn);
        update(we, dwe);
        update(ws, dws);
        update(bn, dbn);
        update(be, dbe);
        update(bs, dbs);
        update(wo, dwo);
        bo -= scale * dbo;
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};

static Metrics evaluate(HypergraphModel& model, const vector<Sample>& samples) {
    Hypergraph g = build_hypergraph(samples);
    const vector<float> logits = model.forward(g);
    int tp = 0, fp = 0, tn = 0, fn = 0;
    float brier = 0.0f;

    for (int i = 0; i < g.n_samples; ++i) {
        const float p = sigmoid(logits[i]);
        const int pred = (p >= 0.5f) ? 1 : 0;
        tp += (pred == 1 && samples[i].y == 1);
        fp += (pred == 1 && samples[i].y == 0);
        tn += (pred == 0 && samples[i].y == 0);
        fn += (pred == 0 && samples[i].y == 1);
        brier += (p - static_cast<float>(samples[i].y))
               * (p - static_cast<float>(samples[i].y));
    }

    const float n = static_cast<float>(samples.size());
    Metrics m;
    m.accuracy = static_cast<float>(tp + tn) / n;
    const float precision = (tp + fp > 0)
        ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    const float recall = (tp + fn > 0)
        ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    m.f1 = (precision + recall > 0.0f)
        ? 2.0f * precision * recall / (precision + recall) : 0.0f;
    m.brier = brier / n;
    return m;
}

static void train(
    HypergraphModel& model,
    const Hypergraph& g,
    const vector<Sample>& samples,
    int epochs,
    float lr
) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const vector<float> logits = model.forward(g);
        vector<float> grad(static_cast<size_t>(g.n_samples), 0.0f);
        for (int i = 0; i < g.n_samples; ++i) {
            grad[i] = sigmoid(logits[i]) - static_cast<float>(samples[i].y);
        }
        model.zero_grad();
        model.backward(g, grad);
        model.step(lr, g.n_samples);
    }
}

static Sample make_test_sample(std::mt19937& rng, int regime) {
    const int barn = (runif(rng, 0.0f, 1.0f) < 0.5f) ? 0 : 1;
    switch (regime) {
        case 0: return make_sample(rng, barn, 0.0f, 0.0f, 0.9f, 0.9f, 0.03f, 0.03f);
        case 1: return make_sample(rng, barn, 1.0f, 0.0f, -1.0f, 0.9f, 0.03f, 0.03f);
        case 2: return make_sample(rng, barn, 0.0f, 1.0f, 0.9f, -1.0f, 0.03f, 0.03f);
        case 3: return make_sample(rng, barn, 0.0f, 0.0f, 0.20f, 0.9f, 0.70f, 0.03f);
        default: return make_sample(rng, barn, 0.0f, 0.0f, 0.9f, 0.20f, 0.03f, 0.70f);
    }
}

int main() {
    constexpr int train_size = 240;
    constexpr int test_size = 300;
    constexpr int epochs = 500;
    constexpr float lr = 0.08f;

    std::mt19937 rng(20261012);
    vector<Sample> train_samples;
    train_samples.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        train_samples.push_back(make_sample(rng, i % 2, 0.20f, 0.20f));
    }
    Hypergraph train_graph = build_hypergraph(train_samples);

    vector<vector<Sample>> test_data(5);
    for (int regime = 0; regime < 5; ++regime) {
        test_data[regime].reserve(test_size);
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    HypergraphModel model(777);
    train(model, train_graph, train_samples, epochs, lr);

    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };

    cout << "Hypergraph fusion: sample, modality, and barn hyperedges\n";
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
