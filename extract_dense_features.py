import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as T


class RADSegFeatureExtractor:
    def __init__(self, model_version="c-radio_v4-h", lang_model="siglip2-g", device="cuda"):
        self.device = device
        print(f"Loading RADSeg model ({model_version}, {lang_model}) to {self.device}...")

        # Load the official model via torch.hub
        self.radseg = torch.hub.load(
            'RADSeg-OVSS/RADSeg', 'radseg_encoder',
            model_version=model_version, lang_model=lang_model, device=self.device, predict=False
        )
        self.radseg.model.eval()
        print("Model loaded successfully.")

    @torch.no_grad()
    def extract_and_save(self, image_path, output_path=None):
        """
        Extract SCGA dense features and visual aligned features from an image.
        Saves both the original 4D spatial tensors (1, C, H, W) and
        flattened 2D tensors (H*W, C) as a dictionary in a .pt file.
        """
        if not os.path.exists(image_path):
            print(f"Error: Image '{image_path}' not found.")
            return None

        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_features.pt"

        print(f"Processing image: {image_path}")
        img = Image.open(image_path).convert('RGB')

        # Image preprocessing
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # 1. Extract base dense features (Dense SCGA Features) -> Shape: (1, C_s, H_s, W_s)
        scga_feat = self.radseg.encode_image_to_feat_map(img_tensor)

        # 2. Map to semantic space (Aligned Dense Features) -> Shape: (1, C_v, H_v, W_v)
        visual_aligned = self.radseg.align_spatial_features_with_language(scga_feat, onehot=False)

        # 3. Flatten features to 2D (H*W, C)
        # Squeeze batch dimension, permute to (H, W, C), and reshape to (H*W, C)
        B_s, C_s, H_s, W_s = scga_feat.shape
        scga_feat_2d = scga_feat.squeeze(0).permute(1, 2, 0).reshape(-1, C_s)

        B_v, C_v, H_v, W_v = visual_aligned.shape
        visual_aligned_2d = visual_aligned.squeeze(0).permute(1, 2, 0).reshape(-1, C_v)

        # 4. Save them as a dictionary
        features_dict = {
            'scga_feat': scga_feat.cpu(),
            'scga_feat_2d': scga_feat_2d.cpu(),
            'visual_aligned': visual_aligned.cpu(),
            'visual_aligned_2d': visual_aligned_2d.cpu()
        }

        torch.save(features_dict, output_path)

        print(f"Successfully saved features to: {output_path}")
        print(f" - scga_feat (4D) shape:        {features_dict['scga_feat'].shape}")
        print(f" - scga_feat_2d (2D) shape:     {features_dict['scga_feat_2d'].shape}")
        print(f" - visual_aligned (4D) shape:   {features_dict['visual_aligned'].shape}")
        print(f" - visual_aligned_2d (2D) shape:{features_dict['visual_aligned_2d'].shape}\n")

        return output_path


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dense features from an image using RADSeg.")

    # Positional argument for the image path
    parser.add_argument("image_path", type=str, help="Path to the input image (e.g., football.png)")

    # Optional arguments
    parser.add_argument("--output_path", type=str, default=None, help="Custom path for the output .pt file")
    parser.add_argument("--model_version", type=str, default="c-radio_v4-h", help="RADSeg model version")
    parser.add_argument("--lang_model", type=str, default="siglip2-g", help="Language model version")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")

    args = parser.parse_args()

    extractor = RADSegFeatureExtractor(
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device
    )

    print("\n--- Extracting Dense Features ---")

    # Extract and save using parsed arguments
    output_file = extractor.extract_and_save(args.image_path, args.output_path)

    # Example: How to load the saved features
    # if output_file and os.path.exists(output_file):
    #     print(f"--- Loading features from {output_file} ---")
    #     loaded_features = torch.load(output_file)
    #
    #     # Retrieve the 4D spatial tensors (1, C, H, W)
    #     loaded_scga_4d = loaded_features['scga_feat']
    #     loaded_aligned_4d = loaded_features['visual_aligned']
    #
    #     # Retrieve the flattened 2D tensors (H*W, C)
    #     loaded_scga_2d = loaded_features['scga_feat_2d']
    #     loaded_aligned_2d = loaded_features['visual_aligned_2d']
    #
    #     print("Features loaded successfully!")
    #     print(f"Loaded 2D Aligned shape: {loaded_aligned_2d.shape}")