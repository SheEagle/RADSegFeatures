class RADSegSCGASearcher:
    def __init__(self, model_version="c-radio_v3-h", lang_model="siglip2-g", device="cuda"):
        self.device = device

        # 加载模型
        self.radseg = torch.hub.load(
            'RADSeg-OVSS/RADSeg', 'radseg_encoder',
            model_version=model_version, lang_model=lang_model, device=self.device, predict=False
        )
        if hasattr(self.radseg, 'model'):
            self.radseg.model.eval()
        else:
            self.radseg.eval()

    def spherical_kmeans(self, features, num_clusters=8, num_iters=100, tol=1e-4):
        x = F.normalize(features, p=2, dim=-1)
        N, D = x.shape
        indices = torch.randperm(N, device=x.device)[:num_clusters]
        centers = x[indices]

        for i in range(num_iters):
            sim = torch.matmul(x, centers.transpose(0, 1))
            labels = torch.argmax(sim, dim=-1)
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
            new_centers = F.normalize(new_centers, p=2, dim=-1)

            center_shift = (centers * new_centers).sum(dim=-1).mean()
            centers = new_centers
            if 1.0 - center_shift < tol:
                print(f"[Spherical K-Means] Converged at iteration {i + 1}")
                break

        return centers, labels

    @torch.no_grad()
    def run_search_comparison(self, image_path, query_text, negative_text="background", top_k=10, temperature=50.0,
                              num_clusters=8):
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # 1. 提取 Dense 特征
        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        B, C, H_f, W_f = visual_aligned.shape
        dense_flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        dense_flat = F.normalize(dense_flat, dim=-1)

        # 2. 提取 Cluster 特征 (聚类中心和每个像素的标签)
        centers, labels = self.spherical_kmeans(dense_flat, num_clusters=num_clusters)
        clustered_flat = centers[labels]  # 仅用于最后的 PCA 可视化还原

        # 3. 文本编码
        text_vecs = self.radseg.encode_prompts([query_text, negative_text], onehot=False)

        # ==================================================
        # 性能对决：分别测量 稠密搜索 vs 聚类搜索 的匹配耗时
        # ==================================================

        # --- A. 稠密搜索 (Dense Search) ---
        if self.device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()

        dense_logits = dense_flat @ text_vecs.T  # 矩阵乘法：(H*W, C) @ (C, 2)
        dense_probs = F.softmax(dense_logits * temperature, dim=-1)
        dense_heatmap = dense_probs[:, 0].reshape(H_f, W_f).cpu().numpy()

        if self.device == "cuda": torch.cuda.synchronize()
        dense_time = (time.perf_counter() - t0) * 1000  # 转换为毫秒 (ms)

        # --- B. 聚类搜索 (Clustered Search) 性能优化版 ---
        if self.device == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()

        # 核心优化：只用 K 个聚类中心去跟文本算相似度！(K, C) @ (C, 2)
        center_logits = centers @ text_vecs.T
        center_probs = F.softmax(center_logits * temperature, dim=-1)
        # 根据 label 将 K 个得分映射回整张图的像素
        clustered_probs = center_probs[labels]
        clustered_heatmap = clustered_probs[:, 0].reshape(H_f, W_f).cpu().numpy()

        if self.device == "cuda": torch.cuda.synchronize()
        cluster_time = (time.perf_counter() - t1) * 1000  # 转换为毫秒 (ms)

        print("-" * 50)
        print(f"[Dense]   Search Time: {dense_time:.3f} ms | Max Prob: {dense_heatmap.max():.4f}")
        print(f"[Cluster] Search Time: {cluster_time:.3f} ms | Max Prob: {clustered_heatmap.max():.4f}")
        print(f"-> Speedup: {dense_time / cluster_time:.2f}x faster!")
        print("-" * 50)

        # 4. 计算 PCA 用于可视化
        pca = PCA(n_components=3)
        dense_pca_flat = pca.fit_transform(dense_flat.cpu().numpy())
        dense_pca_flat = (dense_pca_flat - dense_pca_flat.min(0)) / (
                dense_pca_flat.max(0) - dense_pca_flat.min(0) + 1e-8)
        dense_pca = dense_pca_flat.reshape(H_f, W_f, 3)

        clustered_pca_flat = pca.transform(clustered_flat.cpu().numpy())
        clustered_pca_flat = (clustered_pca_flat - clustered_pca_flat.min(0)) / (
                clustered_pca_flat.max(0) - clustered_pca_flat.min(0) + 1e-8)
        clustered_pca = clustered_pca_flat.reshape(H_f, W_f, 3)

        self._visualize_comparison(img, dense_heatmap, clustered_heatmap, dense_pca, clustered_pca, query_text, top_k,
                                   dense_time, cluster_time)

    def _draw_bboxes_on_ax(self, ax, heatmap_res, top_k, h_orig, w_orig):
        """
        在指定的 matplotlib 轴 (ax) 上提取并绘制 Top-K Bounding Boxes
        """
        min_area = (h_orig * w_orig) * 0.005
        min_val, max_val = heatmap_res.min(), heatmap_res.max()

        boxes = []
        scores = []

        # 如果整张图的得分完全一致（聚类时可能发生全背景的情况），直接跳过
        if max_val - min_val < 1e-4:
            return

        for thresh_ratio in np.linspace(0.2, 0.9, 8):
            thresh = min_val + thresh_ratio * (max_val - min_val)
            binary_mask = (heatmap_res > thresh).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > min_area:
                    score = heatmap_res[y:y + h, x:x + w].max()
                    boxes.append([x, y, x + w, y + h])
                    scores.append(score)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.2)
            top_k_indices = keep_indices[:top_k]

            import matplotlib.cm as cm
            cmap = cm.get_cmap('autumn')

            for render_idx, idx in enumerate(reversed(top_k_indices)):
                rank = len(top_k_indices) - 1 - render_idx
                x1, y1, x2, y2 = boxes[idx]
                w, h = x2 - x1, y2 - y1
                score = scores[idx]

                color_ratio = rank / max(1, top_k - 1)
                color = cmap(color_ratio)
                line_w = max(1.5, 4.0 - rank * 0.3)

                rect = patches.Rectangle((x1, y1), w, h, linewidth=line_w, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                label_text = f"#{rank + 1}: {score:.2f}"
                y_offset = max(5, y1 - 8 - (rank % 3) * 12)

                ax.text(x1, y_offset, label_text, color='white', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor=color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

    def _visualize_comparison(self, img, dense_heatmap, cluster_heatmap, dense_pca, cluster_pca, query, top_k,
                              dense_time, cluster_time):
        h_orig, w_orig = img.size[1], img.size[0]

        dense_res = F.interpolate(torch.tensor(dense_heatmap).unsqueeze(0).unsqueeze(0), size=(h_orig, w_orig),
                                  mode='bilinear').squeeze().numpy()
        cluster_res = F.interpolate(torch.tensor(cluster_heatmap).unsqueeze(0).unsqueeze(0), size=(h_orig, w_orig),
                                    mode='nearest').squeeze().numpy()

        dense_pca_res = F.interpolate(torch.tensor(dense_pca).permute(2, 0, 1).unsqueeze(0), size=(h_orig, w_orig),
                                      mode='bilinear').squeeze().permute(1, 2, 0).numpy()
        cluster_pca_res = F.interpolate(torch.tensor(cluster_pca).permute(2, 0, 1).unsqueeze(0), size=(h_orig, w_orig),
                                        mode='nearest').squeeze().permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(2, 4, figsize=(32, 16), gridspec_kw={'width_ratios': [1, 1, 1.2, 1]})
        img_dark = np.array(img).astype(np.float32) * 0.7 / 255.0

        # ================= ROW 0: DENSE =================
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Dense: Original Image", fontsize=16, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(dense_pca_res)
        axes[0, 1].set_title("Dense: PCA Features", fontsize=16)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(img.convert('L'), cmap='gray')
        im1 = axes[0, 2].imshow(dense_res, cmap='Reds', alpha=0.6, vmin=0, vmax=1.0)
        # 加上耗时标签
        axes[0, 2].set_title(f"Dense Search: '{query}' ({dense_time:.2f} ms)", fontsize=16, fontweight='bold',
                             color='darkred')
        axes[0, 2].axis('off')
        fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04).set_label('Prob', fontsize=12)

        axes[0, 3].imshow(img_dark)
        self._draw_bboxes_on_ax(axes[0, 3], dense_res, top_k, h_orig, w_orig)
        axes[0, 3].set_title("Dense: Top BBoxes", fontsize=16, fontweight='bold')
        axes[0, 3].axis('off')

        # ================= ROW 1: CLUSTERED =================
        axes[1, 0].imshow(img)
        axes[1, 0].set_title(
            f"Clustered (K={cluster_heatmap.shape[0] if len(cluster_heatmap.shape) == 1 else 'Fixed'}): Original Image",
            fontsize=16, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cluster_pca_res)
        axes[1, 1].set_title("Clustered: PCA Features", fontsize=16)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(img.convert('L'), cmap='gray')
        im2 = axes[1, 2].imshow(cluster_res, cmap='Reds', alpha=0.6, vmin=0, vmax=1.0)
        # 加上耗时标签，并标红凸显速度
        axes[1, 2].set_title(f"Cluster Search: '{query}' ({cluster_time:.2f} ms)", fontsize=16, fontweight='bold',
                             color='darkgreen')
        axes[1, 2].axis('off')
        fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04).set_label('Prob', fontsize=12)

        axes[1, 3].imshow(img_dark)
        self._draw_bboxes_on_ax(axes[1, 3], cluster_res, top_k, h_orig, w_orig)
        axes[1, 3].set_title("Clustered: Top BBoxes", fontsize=16, fontweight='bold')
        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.show()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    searcher = RADSegSCGASearcher()

    print("\n--- Running Dense vs Clustered Comparison ---")

    searcher.run_search_comparison("football.png", query_text="kids", negative_text="background",
                                   top_k=10, temperature=80, num_clusters=100)
