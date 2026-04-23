import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import hashlib
import random
from typing import Tuple, List, Optional

class QIMWatermark:
    def __init__(self, delta=50.0, block_size=8, seed=42):
        self.delta = delta
        self.block_size = block_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def _dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def _idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def _quantize(self, coeff, bit):
        interval = self.delta
        if bit == 0:
            quantized = np.round(coeff / interval) * interval
        else:
            quantized = np.round((coeff - interval/2) / interval) * interval + interval/2
        return quantized  
    
    def _extract_bit(self, coeff):
        interval = self.delta
        q0 = np.round(coeff / interval) * interval
        q1 = np.round((coeff - interval/2) / interval) * interval + interval/2
        dist0 = abs(coeff - q0)
        dist1 = abs(coeff - q1)
        return 0 if dist0 <= dist1 else 1
    
    def _generate_watermark(self, length, key=None):
        if key:
            hash_obj = hashlib.sha256(key.encode())
            random.seed(int.from_bytes(hash_obj.digest()[:4], 'big'))
        watermark = np.random.randint(0, 2, length)
        random.seed(self.seed)
        return watermark
    
    def _select_coefficients(self, n_blocks, n_coeffs_per_block):
        available_positions = []
        for i in range(self.block_size):
            for j in range(self.block_size):
                if i + j > 0 and i + j < self.block_size + 2:
                    available_positions.append((i, j))
        selected = []
        for block_idx in range(n_blocks):
            rng = random.Random(self.seed + block_idx)
            positions = rng.sample(available_positions, n_coeffs_per_block)
            for pos in positions:
                selected.append((block_idx, pos[0], pos[1]))
        return selected
    
    def embed(self, image_path, watermark_key=None):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        original_img = img.copy()
        h, w = img.shape
        n_blocks_h = h // self.block_size
        n_blocks_w = w // self.block_size
        n_blocks = n_blocks_h * n_blocks_w
        n_coeffs_per_block = 2
        watermark_length = n_blocks * n_coeffs_per_block
        watermark = self._generate_watermark(watermark_length, watermark_key)
        selected_coeffs = self._select_coefficients(n_blocks, n_coeffs_per_block)
        watermarked_img = np.zeros_like(img, dtype=np.float64)
        embedded_coeffs_info = []
        for block_h in range(n_blocks_h):
            for block_w in range(n_blocks_w):
                block_idx = block_h * n_blocks_w + block_w
                y_start = block_h * self.block_size
                y_end = y_start + self.block_size
                x_start = block_w * self.block_size
                x_end = x_start + self.block_size
                block = img[y_start:y_end, x_start:x_end].astype(np.float64)
                dct_block = self._dct2(block)
                for coeff_idx in range(n_coeffs_per_block):
                    global_idx = block_idx * n_coeffs_per_block + coeff_idx
                    if global_idx >= watermark_length:
                        break
                    pos = selected_coeffs[block_idx * n_coeffs_per_block + coeff_idx]
                    row, col = pos[1], pos[2]
                    original_coeff = dct_block[row, col]
                    bit = watermark[global_idx]
                    quantized_coeff = self._quantize(original_coeff, bit)
                    dct_block[row, col] = quantized_coeff
                    embedded_coeffs_info.append({
                        'block': (block_h, block_w),
                        'position': (row, col),
                        'original': original_coeff,
                        'quantized': quantized_coeff,
                        'bit': bit
                    })
                watermarked_block = self._idct2(dct_block)
                watermarked_img[y_start:y_end, x_start:x_end] = watermarked_block
        watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
        return watermarked_img, watermark, embedded_coeffs_info
    
    def extract(self, watermarked_image, original_image, watermark_length, embedded_coeffs_info):
        h, w = watermarked_image.shape
        extracted_bits = []
        for info in embedded_coeffs_info:
            block_h, block_w = info['block']
            row, col = info['position']
            y_start = block_h * self.block_size
            y_end = y_start + self.block_size
            x_start = block_w * self.block_size
            x_end = x_start + self.block_size
            block = watermarked_image[y_start:y_end, x_start:x_end].astype(np.float64)
            dct_block = self._dct2(block)
            coeff = dct_block[row, col]
            bit = self._extract_bit(coeff)
            extracted_bits.append(bit)
            if len(extracted_bits) >= watermark_length:
                break
        return np.array(extracted_bits[:watermark_length])
    
    def apply_attack(self, image, attack_type, **kwargs):
        attacked = image.copy().astype(np.float64)
        if attack_type == 'gaussian_noise':
            mean = kwargs.get('mean', 0)
            sigma = kwargs.get('sigma', 10)
            noise = np.random.normal(mean, sigma, attacked.shape)
            attacked += noise
            attacked = np.clip(attacked, 0, 255)
        elif attack_type == 'jpeg_compression':
            quality = kwargs.get('quality', 50)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', attacked.astype(np.uint8), encode_param)
            attacked = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        elif attack_type == 'salt_pepper':
            salt_prob = kwargs.get('salt_prob', 0.01)
            pepper_prob = kwargs.get('pepper_prob', 0.01)
            noise = np.random.random(attacked.shape)
            attacked[noise < salt_prob] = 255
            attacked[noise > 1 - pepper_prob] = 0
        return attacked.astype(np.uint8)
    
    def calculate_psnr(self, original, watermarked):
        return peak_signal_noise_ratio(original, watermarked, data_range=255)
    
    def calculate_ber(self, original_watermark, extracted_watermark):
        if len(original_watermark) != len(extracted_watermark):
            raise ValueError("Watermarks must have the same length")
        errors = np.sum(original_watermark != extracted_watermark)
        return errors / len(original_watermark)
    
    def visualize_results(self, original_img, watermarked_img, attacked_img=None, title="Results"):
        if attacked_img is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(original_img, cmap='gray')
            axes[0].set_title(f'Original Image')
            axes[0].axis('off')
            axes[1].imshow(watermarked_img, cmap='gray')
            axes[1].set_title(f'Watermarked Image')
            axes[1].axis('off')
            axes[2].imshow(attacked_img, cmap='gray')
            axes[2].set_title(f'Attacked Image')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_img, cmap='gray')
            axes[0].set_title(f'Original Image')
            axes[0].axis('off')
            axes[1].imshow(watermarked_img, cmap='gray')
            axes[1].set_title(f'Watermarked Image')
            axes[1].axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def main():
    IMAGE_URL = "img.png"
    print("=" * 60)
    print("QIM Digital Watermarking System")
    print("=" * 60)
    qim = QIMWatermark(delta=50.0, block_size=8, seed=42)
    try:
        print("\n[1] Embedding watermark...")
        watermarked_img, original_watermark, coeffs_info = qim.embed(IMAGE_URL)
        original_img = cv2.imread(IMAGE_URL, cv2.IMREAD_GRAYSCALE)
        psnr = qim.calculate_psnr(original_img, watermarked_img)
        print(f"    PSNR: {psnr:.2f} dB")
        print("\n[2] Extracting watermark (no attack)...")
        extracted_watermark = qim.extract(watermarked_img, original_img, len(original_watermark), coeffs_info)
        ber = qim.calculate_ber(original_watermark, extracted_watermark)
        print(f"    BER: {ber:.4f} ({ber*100:.2f}%)")
        qim.visualize_results(original_img, watermarked_img, title="Original vs Watermarked Images")
        print("\n[3] Testing robustness against attacks...")
        attacks = [
            ('Gaussian Noise', 'gaussian_noise', {'sigma': 10}),
            ('JPEG Compression', 'jpeg_compression', {'quality': 50}),
            ('Salt & Pepper Noise', 'salt_pepper', {'salt_prob': 0.01, 'pepper_prob': 0.01})
        ]
        for attack_name, attack_type, params in attacks:
            print(f"\n    Testing {attack_name}...")
            attacked_img = qim.apply_attack(watermarked_img, attack_type, **params)
            extracted_after_attack = qim.extract(attacked_img, original_img, len(original_watermark), coeffs_info)
            ber_after = qim.calculate_ber(original_watermark, extracted_after_attack)
            print(f"        BER after {attack_name}: {ber_after:.4f} ({ber_after*100:.2f}%)")
            qim.visualize_results(original_img, watermarked_img, attacked_img, title=f"Original vs Watermarked vs {attack_name}")
        print("\n" + "=" * 60)
        print("Watermarking completed successfully")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease make sure:")
        print("1. You have installed all required libraries")
        print("2. The image path is correct")
        print("3. The image is a valid grayscale image")

if __name__ == "__main__":
    main()
