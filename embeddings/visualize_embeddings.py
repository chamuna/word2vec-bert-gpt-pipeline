from sklearn.decomposition import PCA  # ไลบรารีสำหรับการลดมิติข้อมูล
import matplotlib.pyplot as plt  # ไลบรารีสำหรับการสร้างกราฟ

def visualize_embeddings(embedding_model):
    """
    ลดมิติและแสดงผล Embedding ในรูปแบบ 2 มิติ
    :param embedding_model: โมเดล Embedding เช่น Word2Vec, FastText, หรือ GloVe
                            ซึ่งสามารถดึงคำและเวกเตอร์ได้ผ่าน index_to_key และ vectors
    """
    # ดึงคำทั้งหมด (vocabulary) จากโมเดล
    words = list(embedding_model.index_to_key)

    # ดึงเวกเตอร์ของคำทั้งหมด
    vectors = embedding_model[words]

    # ใช้ PCA ลดมิติของเวกเตอร์จากหลายมิติให้เหลือ 2 มิติ
    pca = PCA(n_components=2)  # กำหนดให้ลดเหลือ 2 มิติ
    result = pca.fit_transform(vectors)  # ลดมิติของเวกเตอร์

    # แสดงผลในกราฟ 2 มิติ
    plt.scatter(result[:, 0], result[:, 1])  # จุดแสดงตำแหน่งของคำแต่ละคำ
    for i, word in enumerate(words):  # วนลูปเพื่อแสดงคำในกราฟ
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))  # ใส่ชื่อคำในตำแหน่งที่เหมาะสม
    plt.show()  # แสดงกราฟ
