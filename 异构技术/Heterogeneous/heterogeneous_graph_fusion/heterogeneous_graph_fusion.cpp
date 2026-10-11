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

static constexpr int D = 6;   // node feature dimension
static constexpr int L = 3;   // graph layers
static constexpr int R = 6;   // heterogeneous relation types
static constexpr int B = 2;   // number of barns

enum Relation {
    SAMPLE_TO_AUDIO = 0,
    AUDIO_TO_SAMPLE = 1,
    SAMPLE_TO_ENV = 2,
    ENV_TO_SAMPLE = 3,
    SAMPLE_TO_BARN = 4,
    BARN_TO_SAMPLE = 5,
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
    //barn在代码中代表所属区的意思
    //   每一个区的都有群体倾向（比如说风险高/低）
    //  本代码中共有B（2）个区（一个区风险高，一个区风险低）
    //   本代码中的风险高=》大概率y=1
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

    // The barn has its own risk level. A sample node cannot see the barn
    // identity directly; it can only receive it through graph messages.
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

struct Edge {
    int src = 0;
    int dst = 0;
    int rel = 0;
};

struct Graph {
    int n_samples = 0;                 // ① 样本数
    int n_nodes = 0;                   // ② 节点总数
    vector<vector<float>> x;           // ③ 每个节点的特征
    vector<int> sample_nodes;          // ④ 样本节点的编号清单
    vector<int> barn_nodes;            // ⑤ barn节点的编号清单
    vector<Edge> edges;                // ⑥ 所有边（关系）
    vector<int> degree;                // ⑦ 每个节点的"入度"
    vector<float> edge_scale;          // ⑧ 传消息时的缩放系数
};


static Graph build_graph(const vector<Sample>& samples) {
    Graph graph;
    graph.n_samples = static_cast<int>(samples.size());
    graph.n_nodes = 3 * graph.n_samples + B;
    graph.x.assign(graph.n_nodes, vector<float>(D, 0.0f));
    graph.sample_nodes.resize(graph.n_samples);
    graph.barn_nodes.resize(B);

    for (int i = 0; i < graph.n_samples; ++i) {
        const Sample& s = samples[i];
        const int sample_node = i;
        const int audio_node = graph.n_samples + i;
        const int env_node = 2 * graph.n_samples + i;

        graph.sample_nodes[i] = sample_node;
        graph.x[sample_node] = {
            s.a0 * static_cast<float>(s.ma),
            s.a1 * static_cast<float>(s.ma),
            s.e0 * static_cast<float>(s.me),
            s.e1 * static_cast<float>(s.me),
            static_cast<float>(s.ma),
            static_cast<float>(s.me),
        };

        graph.x[audio_node] = {
            s.a0,
            s.a1,
            static_cast<float>(s.ma),
            s.qa,
            0.0f,
            0.0f,
        };

        graph.x[env_node] = {
            s.e0,
            s.e1,
            static_cast<float>(s.me),
            s.qe,
            0.0f,
            0.0f,
        };
    }

    for (int b = 0; b < B; ++b) {
        graph.barn_nodes[b] = 3 * graph.n_samples + b;
        // The barn embedding itself is a trainable model parameter.
        graph.x[graph.barn_nodes[b]].assign(D, 0.0f);
    }

    for (int i = 0; i < graph.n_samples; ++i) {
        const Sample& s = samples[i];
        const int sample_node = graph.sample_nodes[i];
        const int audio_node = graph.n_samples + i;
        const int env_node = 2 * graph.n_samples + i;
        const int barn_node = graph.barn_nodes[s.barn];

        if (s.ma) {
            graph.edges.push_back({sample_node, audio_node, SAMPLE_TO_AUDIO});
            graph.edges.push_back({audio_node, sample_node, AUDIO_TO_SAMPLE});
        }
        if (s.me) {
            graph.edges.push_back({sample_node, env_node, SAMPLE_TO_ENV});
            graph.edges.push_back({env_node, sample_node, ENV_TO_SAMPLE});
        }
        graph.edges.push_back({sample_node, barn_node, SAMPLE_TO_BARN});
        graph.edges.push_back({barn_node, sample_node, BARN_TO_SAMPLE});
    }

    graph.degree.assign(graph.n_nodes, 0);
    for (const Edge& edge : graph.edges) {
        graph.degree[edge.dst] += 1;
    }
    graph.edge_scale.assign(graph.n_nodes, 0.0f);
    for (int i = 0; i < graph.n_nodes; ++i) {
        graph.edge_scale[i] = 1.0f / std::sqrt(static_cast<float>(graph.degree[i] + 1));
    }
    return graph;
}

struct GraphModel {
    // Relation-specific weight matrices: L * R * D * D.
    vector<float> rel_w;
    vector<float> d_rel_w;

    // Self transform per layer: L * D * D.
    vector<float> self_w;
    vector<float> d_self_w;
    vector<float> bias;
    vector<float> d_bias;

    vector<float> barn_embedding;
    vector<float> d_barn_embedding;

    vector<float> wo;
    vector<float> dwo;
    float bo = 0.0f;
    float dbo = 0.0f;

    vector<vector<vector<float>>> history_in;
    vector<vector<vector<float>>> history_out;
    vector<vector<float>> final_h;
    vector<float> sample_logits;

    explicit GraphModel(int seed) {
        rel_w.assign(L * R * D * D, 0.0f);
        d_rel_w.assign(rel_w.size(), 0.0f);
        self_w.assign(L * D * D, 0.0f);
        d_self_w.assign(self_w.size(), 0.0f);
        bias.assign(L * D, 0.0f);
        d_bias.assign(bias.size(), 0.0f);
        barn_embedding.assign(B * D, 0.0f);
        d_barn_embedding.assign(barn_embedding.size(), 0.0f);
        wo.assign(D, 0.0f);
        dwo.assign(D, 0.0f);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> init(-0.35f, 0.35f);
        for (float& value : rel_w) value = init(rng);
        for (float& value : self_w) value = init(rng);
        for (float& value : barn_embedding) value = init(rng);
        for (float& value : wo) value = init(rng);
    }

    int rel_index(int layer, int rel, int row, int col) const {
        return ((layer * R + rel) * D + row) * D + col;
    }

    int self_index(int layer, int row, int col) const {
        return (layer * D + row) * D + col;
    }

    int bias_index(int layer, int row) const {
        return layer * D + row;
    }

    vector<float> forward(const Graph& graph) {
        vector<vector<float>> h = graph.x;
        for (int b = 0; b < B; ++b) {
            h[graph.barn_nodes[b]] = vector<float>(
                barn_embedding.begin() + b * D,
                barn_embedding.begin() + (b + 1) * D
            );
        }

        history_in.assign(L, {});
        history_out.assign(L, {});

        for (int layer = 0; layer < L; ++layer) {
            history_in[layer] = h;
            vector<vector<float>> pre(graph.n_nodes, vector<float>(D, 0.0f));

            for (int i = 0; i < graph.n_nodes; ++i) {
                for (int row = 0; row < D; ++row) {
                    float sum = bias[bias_index(layer, row)];
                    for (int col = 0; col < D; ++col) {
                        sum += self_w[self_index(layer, row, col)] * h[i][col];
                    }
                    pre[i][row] = sum;
                }
            }

            for (const Edge& edge : graph.edges) {
                const float scale = graph.edge_scale[edge.dst];
                for (int row = 0; row < D; ++row) {
                    float message = 0.0f;
                    for (int col = 0; col < D; ++col) {
                        message += rel_w[rel_index(layer, edge.rel, row, col)]
                                 * h[edge.src][col];
                    }
                    pre[edge.dst][row] += scale * message;
                }
            }

            vector<vector<float>> next(graph.n_nodes, vector<float>(D, 0.0f));
            for (int i = 0; i < graph.n_nodes; ++i) {
                for (int row = 0; row < D; ++row) {
                    next[i][row] = std::tanh(pre[i][row]);
                }
            }
            history_out[layer] = next;
            h = next;
        }

        final_h = h;
        sample_logits.assign(graph.n_samples, 0.0f);
        for (int i = 0; i < graph.n_samples; ++i) {
            const int node = graph.sample_nodes[i];
            float logit = bo;
            for (int row = 0; row < D; ++row) {
                logit += wo[row] * final_h[node][row];
            }
            sample_logits[i] = logit;
        }
        return sample_logits;
    }

    void backward(const Graph& graph, const vector<float>& dlogits) {
        vector<vector<float>> dh(graph.n_nodes, vector<float>(D, 0.0f));

        for (int i = 0; i < graph.n_samples; ++i) {
            const int node = graph.sample_nodes[i];
            for (int row = 0; row < D; ++row) {
                dwo[row] += dlogits[i] * final_h[node][row];
                dh[node][row] += dlogits[i] * wo[row];
            }
            dbo += dlogits[i];
        }

        for (int layer = L - 1; layer >= 0; --layer) {
            vector<vector<float>> dpre(graph.n_nodes, vector<float>(D, 0.0f));
            vector<vector<float>> dh_in(graph.n_nodes, vector<float>(D, 0.0f));

            for (int i = 0; i < graph.n_nodes; ++i) {
                for (int row = 0; row < D; ++row) {
                    const float out = history_out[layer][i][row];
                    dpre[i][row] = dh[i][row] * (1.0f - out * out);
                    d_bias[bias_index(layer, row)] += dpre[i][row];

                    for (int col = 0; col < D; ++col) {
                        d_self_w[self_index(layer, row, col)] +=
                            dpre[i][row] * history_in[layer][i][col];
                        dh_in[i][col] +=
                            self_w[self_index(layer, row, col)] * dpre[i][row];
                    }
                }
            }

            for (const Edge& edge : graph.edges) {
                const float scale = graph.edge_scale[edge.dst];
                for (int row = 0; row < D; ++row) {
                    const float g = dpre[edge.dst][row] * scale;
                    for (int col = 0; col < D; ++col) {
                        d_rel_w[rel_index(layer, edge.rel, row, col)] +=
                            g * history_in[layer][edge.src][col];
                        dh_in[edge.src][col] +=
                            rel_w[rel_index(layer, edge.rel, row, col)] * g;
                    }
                }
            }

            dh = dh_in;
        }

        for (int b = 0; b < B; ++b) {
            const int node = graph.barn_nodes[b];
            for (int row = 0; row < D; ++row) {
                d_barn_embedding[b * D + row] += dh[node][row];
            }
        }
    }

    void zero_grad() {
        std::fill(d_rel_w.begin(), d_rel_w.end(), 0.0f);
        std::fill(d_self_w.begin(), d_self_w.end(), 0.0f);
        std::fill(d_bias.begin(), d_bias.end(), 0.0f);
        std::fill(d_barn_embedding.begin(), d_barn_embedding.end(), 0.0f);
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
        update(rel_w, d_rel_w);
        update(self_w, d_self_w);
        update(bias, d_bias);
        update(barn_embedding, d_barn_embedding);
        update(wo, dwo);
        bo -= scale * dbo;
    }
};

struct Metrics {
    float accuracy = 0.0f;
    float f1 = 0.0f;
    float brier = 0.0f;
};

static Metrics evaluate(GraphModel& model, const vector<Sample>& samples) {
    Graph graph = build_graph(samples);
    const vector<float> logits = model.forward(graph);
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    float brier = 0.0f;

    for (int i = 0; i < graph.n_samples; ++i) {
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

static void train(GraphModel& model, const Graph& graph, const vector<Sample>& samples, int epochs, float lr) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const vector<float> logits = model.forward(graph);
        vector<float> grad(static_cast<size_t>(graph.n_samples), 0.0f);
        for (int i = 0; i < graph.n_samples; ++i) {
            const float p = sigmoid(logits[i]);
            grad[i] = p - static_cast<float>(samples[i].y);
        }
        model.zero_grad();
        model.backward(graph, grad);
        model.step(lr, graph.n_samples);
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
    constexpr int epochs = 600;
    constexpr float lr = 0.08f;

    std::mt19937 rng(20261011);
    vector<Sample> train_samples;
    train_samples.reserve(train_size);
    for (int i = 0; i < train_size; ++i) {
        //奇偶交叉分配 ：
        // -目的：保证模型在训练时**两个片区都见过、数量均匀**，不会偏科。
        const int barn = (i % 2 == 0) ? 0 : 1;
        train_samples.push_back(make_sample(rng, barn, 0.20f, 0.20f));
    }
    Graph train_graph = build_graph(train_samples);

    vector<vector<Sample>> test_data(5);
    for (int regime = 0; regime < 5; ++regime) {
        test_data[regime].reserve(test_size);
        for (int i = 0; i < test_size; ++i) {
            test_data[regime].push_back(make_test_sample(rng, regime));
        }
    }

    GraphModel model(777);
    train(model, train_graph, train_samples, epochs, lr);

    const vector<string> names = {
        "complete",
        "missing-audio",
        "missing-env",
        "noisy-audio",
        "noisy-env",
    };

    cout << "Heterogeneous graph fusion: sample / audio / environment / barn nodes\n";
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
