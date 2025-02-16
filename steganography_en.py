import os
import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---- Function: Palette-Based Quantization ----
def quantize_image(image, n_colors=16):
    """
    Reduces the number of gray levels in an image using K-Means clustering.
    Returns the quantized image.
    """
    h, w = image.shape
    pixels = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = quantized_pixels.reshape(h, w)
    return quantized_image.astype(np.uint8)

# ---- Directories for Cover, Secret, and Output Images ----
cover_dir = 'cover_images/'
secret_dir = 'secret_images/'
stego_output_dir = 'stego_image/'
secret_coeffs_dir = 'secret_coeffs/'
os.makedirs(stego_output_dir, exist_ok=True)
os.makedirs(secret_coeffs_dir, exist_ok=True)

# List and sort files numerically (assuming numeric filenames like 1.jpg, 2.jpg, etc.)
cover_files = sorted(os.listdir(cover_dir), key=lambda x: int(os.path.splitext(x)[0]))
secret_files = sorted(os.listdir(secret_dir), key=lambda x: int(os.path.splitext(x)[0]))

# Ensure we process only the matching image pairs
num_images = min(len(cover_files), len(secret_files))
embedding_strength = 0.1  # Embedding strength for HL and HH sub-bands

# Loop through matched images
for i in range(num_images):
    cover_path = os.path.join(cover_dir, cover_files[i])
    secret_path = os.path.join(secret_dir, secret_files[i])
    
    # Load images in grayscale mode
    cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    secret_image = cv2.imread(secret_path, cv2.IMREAD_GRAYSCALE)
    
    if cover_image is None or secret_image is None:
        print(f"Error reading images: {cover_path} or {secret_path}. Skipping.")
        continue

    # Resize the secret image to match cover image dimensions
    secret_image = cv2.resize(secret_image, (cover_image.shape[1], cover_image.shape[0]))

    # Apply palette-based quantization
    quantized_secret = quantize_image(secret_image, n_colors=16)

    # ---- Apply DWT on the Cover Image ----
    coeffs_cover = pywt.dwt2(cover_image, "haar")
    LL_cover, (LH_cover, HL_cover, HH_cover) = coeffs_cover

    # ---- Apply DWT on the Quantized Secret Image ----
    coeffs_secret = pywt.dwt2(quantized_secret, "haar")
    LL_secret, (LH_secret, HL_secret, HH_secret) = coeffs_secret

    # ---- Save Secret Image's LL and LH Coefficients for Decryption ----
    np.save(os.path.join(secret_coeffs_dir, f"LL_secret_{i+1}.npy"), LL_secret)
    np.save(os.path.join(secret_coeffs_dir, f"LH_secret_{i+1}.npy"), LH_secret)

    # ---- Embed the Secret's Detail Components into the Cover Image ----
    HL_stego = HL_cover + embedding_strength * HL_secret
    HH_stego = HH_cover + embedding_strength * HH_secret

    # Combine the unmodified LL and LH sub-bands of the cover with the modified HL and HH
    coeffs_stego = (LL_cover, (LH_cover, HL_stego, HH_stego))

    # Reconstruct the stego image using inverse DWT
    stego_image = pywt.idwt2(coeffs_stego, "haar")

    # Save the stego image
    output_filename = os.path.join(stego_output_dir, f"stego_image_{i+1}.jpg")
    cv2.imwrite(output_filename, stego_image)

    print(f"Processed {cover_files[i]} and {secret_files[i]}. Saved {output_filename}.")
