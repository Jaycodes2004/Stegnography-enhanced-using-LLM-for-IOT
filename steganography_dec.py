import os
import re
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# ---- Directories for Input and Output ----
stego_dir = "stego_image/"
cover_dir = "cover_images/"
secret_coeffs_dir = "secret_coeffs/"  # Directory where LL and LH secret coefficients are saved
output_dir = "extracted_secrets/"
os.makedirs(output_dir, exist_ok=True)

# Function to extract numeric portion from a filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0

# List and sort files numerically based on the numeric portion of the filename
stego_files = sorted(os.listdir(stego_dir), key=extract_number)
cover_files = sorted(os.listdir(cover_dir), key=extract_number)

# Process only as many image pairs as available
num_iterations = min(len(stego_files), len(cover_files))
embedding_strength = 0.1  # Must match the embedding strength used during encryption

# Loop through each iteration
for i in range(num_iterations):
    stego_path = os.path.join(stego_dir, stego_files[i])
    cover_path = os.path.join(cover_dir, cover_files[i])
    
    # Check if both images exist
    if not os.path.exists(stego_path) or not os.path.exists(cover_path):
        print(f"Missing files: {stego_path} or {cover_path}. Skipping iteration {i+1}.")
        continue

    # Load the stego and cover images in grayscale
    stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    if stego_img is None or cover_img is None:
        print(f"Error loading {stego_path} or {cover_path}. Skipping iteration {i+1}.")
        continue

    # If sizes mismatch, resize stego image to match cover image dimensions
    if stego_img.shape != cover_img.shape:
        print(f"Size mismatch: {stego_path} and {cover_path}. Resizing stego image to match cover image.")
        stego_img = cv2.resize(stego_img, (cover_img.shape[1], cover_img.shape[0]))
    
    # ---- Apply DWT on the Stego Image ----
    coeffs_stego = pywt.dwt2(stego_img, "haar")
    LL_stego, (LH_stego, HL_stego, HH_stego) = coeffs_stego

    # ---- Apply DWT on the Cover Image ----
    coeffs_cover = pywt.dwt2(cover_img, "haar")
    LL_cover, (LH_cover, HL_cover, HH_cover) = coeffs_cover

    # ---- Load Saved Secret Coefficients from Encryption (Non-Blind System) ----
    ll_secret_file = os.path.join(secret_coeffs_dir, f"LL_secret_{i+1}.npy")
    lh_secret_file = os.path.join(secret_coeffs_dir, f"LH_secret_{i+1}.npy")
    
    if not os.path.exists(ll_secret_file) or not os.path.exists(lh_secret_file):
        print(f"Secret coefficients for iteration {i+1} not found. Skipping iteration.")
        continue
    
    LL_secret = np.load(ll_secret_file)
    LH_secret = np.load(lh_secret_file)

    # ---- Extract the Secret's HL and HH Components ----
    # Ensure HL and HH sub-bands have matching shapes (using minimum dimensions)
    hl_shape = (min(HL_stego.shape[0], HL_cover.shape[0]), min(HL_stego.shape[1], HL_cover.shape[1]))
    hh_shape = (min(HH_stego.shape[0], HH_cover.shape[0]), min(HH_stego.shape[1], HH_cover.shape[1]))
    
    HL_stego = HL_stego[:hl_shape[0], :hl_shape[1]]
    HL_cover = HL_cover[:hl_shape[0], :hl_shape[1]]
    HH_stego = HH_stego[:hh_shape[0], :hh_shape[1]]
    HH_cover = HH_cover[:hh_shape[0], :hh_shape[1]]
    
    HL_recovered = (HL_stego - HL_cover) / embedding_strength
    HH_recovered = (HH_stego - HH_cover) / embedding_strength

    # ---- Reconstruct the Secret Image's DWT Coefficients ----
    # Using the saved LL and LH coefficients from encryption (non-blind system)
    coeffs_recovered = (LL_secret, (LH_secret, HL_recovered, HH_recovered))
    recovered_secret = pywt.idwt2(coeffs_recovered, "haar")
    
    # Normalize the recovered secret image to the range [0,255]
    recovered_secret = np.clip(recovered_secret, 0, 255).astype(np.uint8)
    
    # Save the extracted secret image with a unique filename
    output_filename = os.path.join(output_dir, f"extracted_secret_{i+1}.jpg")
    cv2.imwrite(output_filename, recovered_secret)
    
    print(f"Iteration {i+1}/{num_iterations} completed. Saved {output_filename}.")
