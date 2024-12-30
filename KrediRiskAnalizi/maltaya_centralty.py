import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Veriyi Yükleme ve Hazırlama
data = pd.read_excel(r'C:\Users\admin\Desktop\Veri Madencilği\kredi_risk_analizi.xlsx')

# Modelde kullanılacak özellikler
features = [
    'Credit Score', 'Annual Income', 'AnnualExpenses', 'Loan Amount',
    'Debt-to-Income Ratio', 'Bankruptcy History', 'Previous Credit Defaults',
    'Payment History', 'Net Worth', 'InterestRate'
]
label = 'Credit Approval'  # Hedef sütun


data_accepted = data[data[label] == 1].sample(n=min(1000, len(data[data[label] == 1])), random_state=42)
data_rejected = data[data[label] == 0].sample(n=min(1000, len(data[data[label] == 0])), random_state=42)

# Kabul ve red verilerini ayırma
ornek_accepted = data_accepted.sample(n=10, random_state=42)
ornek_rejected = data_rejected.sample(n=10, random_state=42)

X_accepted = ornek_accepted[features]
X_rejected = ornek_rejected[features]

# 2. Kabul ve red için benzerlik grafiği oluşturma
def create_graph(data):
    graph = nx.Graph()
    for i in range(len(data)):
        graph.add_node(i, features=data.iloc[i].values, similarity_count=0)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            similarity = sum(data.iloc[i].values == data.iloc[j].values)
            if similarity > 0:
                graph.add_edge(i, j, weight=similarity)
    return graph

acceptance_graph = create_graph(X_accepted)
rejection_graph = create_graph(X_rejected)

# 3. Malatya Merkeziliği Hesaplama
def calculate_malatya_centrality(graph):
    centrality = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        total_weight = sum(graph[node][neighbor]["weight"] for neighbor in neighbors)
        centrality[node] = total_weight / len(neighbors) if neighbors else 0
    return centrality

acceptance_centrality = calculate_malatya_centrality(acceptance_graph)
rejection_centrality = calculate_malatya_centrality(rejection_graph)

# 4. Merkeziyet ve benzerlik kullanarak sınıflandırma
def classify_with_centrality(sample, acceptance_graph, rejection_graph,
                             acceptance_centrality, rejection_centrality):
    acceptance_scores = [
        sum(sample == data['features']) * acceptance_centrality[node]
        for node, data in acceptance_graph.nodes(data=True)
    ]
    rejection_scores = [
        sum(sample == data['features']) * rejection_centrality[node]
        for node, data in rejection_graph.nodes(data=True)
    ]
    acceptance_score = sum(acceptance_scores) / sum(acceptance_centrality.values())
    rejection_score = sum(rejection_scores) / sum(rejection_centrality.values())
    return 'Accepted' if acceptance_score > rejection_score else 'Rejected'


def add_sample_to_graph(sample, graph):
    new_node_index = len(graph.nodes)
    graph.add_node(new_node_index, features=sample, similarity_count=0)
    for node in graph.nodes:
        if node == new_node_index:
            continue
        # benzerlik hisaplama
        similarity = sum([1 for s, g in zip(sample, graph.nodes[node]['features']) if s == g])
        if similarity > 0:  # إذا كان هناك تشابه
            graph.add_edge(new_node_index, node, weight=similarity)
            graph.nodes[new_node_index]['similarity_count'] += similarity
            graph.nodes[node]['similarity_count'] += similarity
    return new_node_index


# 6. yeni düğüm için açıklayıcı graf
def draw_graphs_with_sample(new_sample, acceptance_graph, rejection_graph,
                            acceptance_centrality, rejection_centrality, new_sample_color):
    # Her çizime yeni örnek ekle
    acceptance_new_node = add_sample_to_graph(new_sample, acceptance_graph)
    rejection_new_node = add_sample_to_graph(new_sample, rejection_graph)

    # Yeni örneğin merkeziliğini hesaplayın
    acceptance_sample_centrality = calculate_malatya_centrality(acceptance_graph)[acceptance_new_node]
    rejection_sample_centrality = calculate_malatya_centrality(rejection_graph)[rejection_new_node]

    # kabul çizimi
    pos_acceptance = nx.spring_layout(acceptance_graph)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    nx.draw(acceptance_graph, pos_acceptance, node_color=['green' if node != acceptance_new_node else new_sample_color
                                                          for node in acceptance_graph.nodes],
            with_labels=True, node_size=500)
    plt.title(f"Acceptance Graph (Centrality: {acceptance_sample_centrality:.2f})")

    # Reddetme çizimi
    pos_rejection = nx.spring_layout(rejection_graph)
    plt.subplot(1, 2, 2)
    nx.draw(rejection_graph, pos_rejection, node_color=['red' if node != rejection_new_node else new_sample_color
                                                        for node in rejection_graph.nodes],
            with_labels=True, node_size=500)
    plt.title(f"Rejection Graph (Centrality: {rejection_sample_centrality:.2f})")

    plt.show()

#7. Tahmin Deneyimi
new_sample = [750, 50000, 20000, 10000, 0.4, 0, 1, 0.9, 100000, 5.5]

# Sonuçları hesapla ve konsola ekle.
def predict_and_print(new_sample, acceptance_graph, rejection_graph,
                      acceptance_centrality, rejection_centrality):
    # Her çizime yeni örnek ekle
    acceptance_new_node = add_sample_to_graph(new_sample, acceptance_graph)
    rejection_new_node = add_sample_to_graph(new_sample, rejection_graph)

    # Yeni örneğin merkeziliğini hesaplayın
    acceptance_sample_centrality = calculate_malatya_centrality(acceptance_graph)[acceptance_new_node]
    rejection_sample_centrality = calculate_malatya_centrality(rejection_graph)[rejection_new_node]

    # Kabul ve ret için yeni numunenin merkezi olarak basılması
    print("New Sample Results:")
    print(f" - Acceptance Centrality: {acceptance_sample_centrality:.2f}")
    print(f" - Rejection Centrality: {rejection_sample_centrality:.2f}")

    # Merkeziliğe dayalı tahmin
    if acceptance_sample_centrality > rejection_sample_centrality:
        prediction = "Accepted"
    else:
        prediction = "Rejected"

    # Beklenen sonucu yazdır
    print(f" - Prediction: {prediction}")

    # grafik çizimi
    draw_graphs_with_sample(new_sample, acceptance_graph, rejection_graph,
                            acceptance_centrality, rejection_centrality, 'blue')

# Tahmini çalıştır ve işlevi yazdır
predict_and_print(new_sample, acceptance_graph, rejection_graph,
                  acceptance_centrality, rejection_centrality)

X_accepted = data_accepted[features]  # Kabul verisi
X_rejected = data_rejected[features]  # Red verisi

# Test verisi oluşturma (150 örnek)
test_data = data.drop(data_accepted.index.union(data_rejected.index)).sample(n=150, random_state=42)
X_test = test_data[features]
y_test = test_data[label]

# 2. Kabul ve red için benzerlik grafiği oluşturma
def create_graph(data):
    graph = nx.Graph()
    for i in range(len(data)):
        graph.add_node(i, features=data.iloc[i].values, similarity_count=0)  # .iloc kullanımı
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            similarity = sum(data.iloc[i].values == data.iloc[j].values)  # .iloc kullanımı
            if similarity > 0:  # Benzerlik varsa
                graph.add_edge(i, j, weight=similarity)
                graph.nodes[i]['similarity_count'] += similarity
                graph.nodes[j]['similarity_count'] += similarity
    return graph

acceptance_graph = create_graph(X_accepted)
rejection_graph = create_graph(X_rejected)


acceptance_centrality = calculate_malatya_centrality(acceptance_graph)
rejection_centrality = calculate_malatya_centrality(rejection_graph)


# 6. Test verileri üzerinde modeli test etme
y_pred = []
for sample in X_test.values:
    result = classify_with_centrality(sample, acceptance_graph, rejection_graph, acceptance_centrality,
                                      rejection_centrality)
    y_pred.append(1 if result == 'Accepted' else 0)

# Karmaşıklık Matrisi
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percentage = (conf_matrix / conf_matrix.sum()) * 100

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='binary') * 100
recall = recall_score(y_test, y_pred, average='binary') * 100
f1 = f1_score(y_test, y_pred, average='binary') * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print("Confusion Matrix (Percentage):")
print(conf_matrix_percentage)


