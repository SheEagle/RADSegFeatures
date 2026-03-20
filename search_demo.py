import cv2
from matplotlib import patches
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from sklearn.decomposition import PCA
import torchvision.ops as ops

class RADSegSCGASearcher:
    def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda"):
        self.device = device

        # Load official model via torch.hub. predict=False returns feature maps directly.
        self.radseg = torch.hub.load(
            'RADSeg-OVSS/RADSeg', 'radseg_encoder',
            model_version=model_version, lang_model=lang_model, device=self.device, predict=False
        )
        self.radseg.model.eval()

    def compute_pca(self, feature_map):
        B, C, H, W = feature_map.shape
        feat_flat = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()
        pca = PCA(n_components=3)
        pca_feat = pca.fit_transform(feat_flat)
        pca_feat = (pca_feat - pca_feat.min(0)) / (pca_feat.max(0) - pca_feat.min(0) + 1e-8)
        return pca_feat.reshape(H, W, 3)

    def spherical_kmeans(self, features, num_clusters=8, num_iters=100, tol=1e-4):
        """
        Spherical K-Means clustering using cosine similarity.
        Args:
            features: Tensor of shape (N, D), N is pixel count, D is feature dim.
            num_clusters: Number of cluster centers (K).
            num_iters: Max iterations.
            tol: Convergence tolerance.
        Returns:
            centers: (K, D) cluster center features (L2 normalized).
            labels: (N,) labels assigning each pixel to a center.
        """
        # L2 normalize inputs (project to hypersphere)
        x = F.normalize(features, p=2, dim=-1)
        N, D = x.shape

        # 1. Randomly init K centers from existing features
        indices = torch.randperm(N, device=x.device)[:num_clusters]
        centers = x[indices]

        for i in range(num_iters):
            # 2. Compute cosine similarity between all pixels and centers
            # x: (N, D), centers.T: (D, K) -> sim: (N, K)
            sim = torch.matmul(x, centers.transpose(0, 1))

            # 3. Assign pixels to the most similar center
            labels = torch.argmax(sim, dim=-1)

            # 4. Update centers: sum features of pixels in the same cluster
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)

            # 5. Re-apply L2 normalization
            new_centers = F.normalize(new_centers, p=2, dim=-1)

            # Check convergence: cosine similarity between old and new centers
            center_shift = (centers * new_centers).sum(dim=-1).mean()
            centers = new_centers
            if 1.0 - center_shift < tol:
                print(f"[Spherical K-Means] Converged at iteration {i + 1}")
                break

        return centers, labels

    @torch.no_grad()
    def get_clustered_image_representation(self, image_path, num_clusters=8):
        """
        Helper: Extracts dense features and clusters them into core representations for image search.
        """
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # 1. Extract dense SCGA features
        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)

        # 2. Map to SigLIP2 semantic space via alignment adapter
        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        # 3. Flatten features
        B, C, H_f, W_f = visual_aligned.shape
        flat_features = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        flat_features = F.normalize(flat_features, dim=-1)  # (H*W, C)

        print(f"Original dense features shape: {flat_features.shape}")

        # 4. Run spherical K-Means clustering
        centers, labels = self.spherical_kmeans(flat_features, num_clusters=num_clusters)
        print(f"Clustered centers shape: {centers.shape}")

        return centers, labels, (H_f, W_f)

    @torch.no_grad()
    def run_search(self, image_path, query_text, negative_text="background", top_k=10, temperature=50.0):
        """
        Core update: Introduces negative_text and Softmax contrastive logic.
        """
        img = Image.open(image_path).convert('RGB')

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)
        pca_rgb = self.compute_pca(scga_feat)

        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        B, C, H_f, W_f = visual_aligned.shape
        flat = visual_aligned.permute(0, 2, 3, 1).reshape(-1, C)
        flat = F.normalize(flat, dim=-1)  # (H*W, C)

        # 1. Encode both query and negative text
        text_vecs = self.radseg.encode_prompts([query_text, negative_text], onehot=False)  # (2, C)

        # 2. Compute cosine similarity between image features and texts
        logits = flat @ text_vecs.T  # Shape: (H*W, 2)

        # 3. Scale by temperature and apply Softmax to get relative probabilities
        probs = F.softmax(logits * temperature, dim=-1)  # (H*W, 2)

        # 4. Extract probability of the target class (column 0) for the heatmap
        target_prob = probs[:, 0]
        heatmap = target_prob.reshape(H_f, W_f).cpu().numpy()

        print(f"Target Probability Range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        self._visualize(img, heatmap, pca_rgb, query_text, top_k)

    def _visualize(self, img, heatmap, pca_rgb, query, top_k):
        h_orig, w_orig = img.size[1], img.size[0]

        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heatmap_res = F.interpolate(heatmap_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().numpy()

        pca_tensor = torch.tensor(pca_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        pca_res = F.interpolate(pca_tensor, size=(h_orig, w_orig), mode='bilinear').squeeze().permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 4, figsize=(32, 8), gridspec_kw={'width_ratios': [1, 1, 1.2, 1]})

        axes[0].imshow(img)
        axes[0].set_title("1. Original Image", fontsize=15, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pca_res)
        axes[1].set_title("2. SCGA PCA Features", fontsize=15)
        axes[1].axis('off')

        axes[2].imshow(img.convert('L'), cmap='gray')
        im_heat = axes[2].imshow(heatmap_res, cmap='Reds', alpha=0.6, vmin=0, vmax=1.0)
        axes[2].set_title(f"3. Prob Heatmap: '{query}'", fontsize=15, fontweight='bold')
        axes[2].axis('off')

        cbar = fig.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label('Target Probability (Softmax)', fontsize=12)

        img_dark = np.array(img).astype(np.float32) * 0.7 / 255.0
        axes[3].imshow(img_dark)
        axes[3].set_title(f"4. NMS Top {top_k} BBox", fontsize=15, fontweight='bold', color='darkgreen')
        axes[3].axis('off')

        min_area = (h_orig * w_orig) * 0.005
        min_val, max_val = heatmap_res.min(), heatmap_res.max()

        boxes = []
        scores = []

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
                axes[3].add_patch(rect)

                label_text = f"#{rank + 1}: {score:.2f}"
                y_offset = max(5, y1 - 8 - (rank % 3) * 12)

                axes[3].text(x1, y_offset, label_text, color='white', fontsize=10, fontweight='bold',
                             bbox=dict(facecolor=color, alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))
        else:
            print("No matching regions found.")

        plt.tight_layout()
        plt.show()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    searcher = RADSegSCGASearcher()


    print("\n--- Running Text-to-Image Search ---")

    # searcher.run_search("football.png", query_text="soccer", negative_text="background", top_k=10,
    #                     temperature=50.0)
    searcher.run_search("football.png", query_text="kids", negative_text="background", top_k=10,
                        temperature=50.0)
