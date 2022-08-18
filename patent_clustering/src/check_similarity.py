import os
import convert_config
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D


def check_similarity(opt, text_features_dict):
    for key, text_feature in text_features_dict.items():
        print(f"check sims of key : {key}")

        if opt.clustering.find_optimal_clusters:
            find_optimal_clusters(opt, text_feature, max_k=opt.clustering.kmeans_max_k)

        print("how many clusters you use ? : ")
        n_clusters = int(input())
        clusters = MiniBatchKMeans(
            n_clusters=n_clusters, init_size=1024, batch_size=2048, random_state=20
        ).fit_predict(text_feature)

        print(clusters)

        plot_tsne_pca(opt, text_feature, clusters)


def find_optimal_clusters(opt, data, max_k):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(
            MiniBatchKMeans(
                n_clusters=k, init_size=1024, batch_size=2048, random_state=20
            )
            .fit(data)
            .inertia_
        )
        print("Fit {} clusters".format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker="o")
    ax.set_xlabel("Cluster Centers")
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel("SSE")
    ax.set_title("SSE by Cluster Center Plot")

    buf, size = f.canvas.print_to_buffer()
    # print('size is : ',size)
    img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(size[1], size[0], -1)
    table = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
    img_path = os.path.join(opt.debug.home_dir, "find_optimal_clusters.jpeg")
    cv2.imwrite(img_path, table)


def make_table(self, fig, x_list, y_dic, frame_num):
    ax = fig.add_subplot(1, 1, 1)
    for clr in self.use_color:
        ax.plot(x_list, y_dic[clr], color=self.plot_color_dic[clr], label=clr)
        ax.scatter(frame_num, y_dic[clr][-1], c=self.plot_color_dic[clr])
    ax.legend(loc=0)
    buf, size = fig.canvas.print_to_buffer()
    # print('size is : ',size)
    img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(size[1], size[0], -1)
    table = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
    return table


from scipy.sparse import csr_matrix


def plot_tsne_pca(opt, data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    # max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)

    pca = PCA(n_components=opt.plot.pca_components).fit_transform(
        csr_matrix(data[max_items, :]).todense()
    )
    tsne = TSNE().fit_transform(
        PCA(n_components=opt.plot.tsne_components).fit_transform(
            csr_matrix(data[max_items, :]).todense()
        )
        # PCA(n_components=50).fit_transform(csr_matrix(data[max_items, :]).todense())
    )

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    # idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title("PCA Cluster Plot")

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title("TSNE Cluster Plot")

    buf, size = f.canvas.print_to_buffer()
    img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(size[1], size[0], -1)
    table = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)

    img_path = os.path.join(opt.debug.home_dir, "plot_tsne_pca.jpeg")
    cv2.imwrite(img_path, table)

    for key, data in zip(["pca", "tsne"], [pca, tsne]):

        fps = 10
        size = (432, 288)

        fourcc = cv2.VideoWriter_fourcc(["m", "p", "v", "4"])
        out_writer = cv2.VideoWriter(
            f"/home/kento/tomita/my_ML/clustering_cc/patent_clustering/debug/plot_3d_{key}.mp4",
            fourcc,
            fps,
            size,
        )
        for angle in range(0, 180):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(30, angle)
            ax.scatter(
                data[idx, 0], data[idx, 1], data[idx, 2], alpha=0.4, c=label_subset
            )

            buf, size = fig.canvas.print_to_buffer()
            img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(size[1], size[0], -1)
            frame = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)

            out_writer.write(frame)
            plt.clf()
            plt.close()

        out_writer.release()


if __name__ == "__main__":
    config = convert_config.convert_config(
        path="/home/kento/tomita/my_ML/clustering_cc/patent_clustering/config/base.yaml"
    )
    check_similarity(config)
