# -*- coding: utf-8 -*-
"""
Demo: Visualize preprocessing steps
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from core.detector import FaceDetector
from core.embedder import FaceEmbedder


def visualize_preprocessing(image_path):
    """Visualize từng bước preprocessing"""
    
    print("="*70)
    print("PREPROCESSING VISUALIZATION")
    print("="*70)
    
    # Load models
    print("\n[1/6] Loading models...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    print("✓ Models loaded")
    
    # Load and align face
    print(f"\n[2/6] Loading and aligning face...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Cannot load: {image_path}")
        return
    
    faces = detector.detect_and_crop(image)
    if not faces:
        print("✗ No face detected!")
        return
    
    _, aligned = faces[0]
    print(f"✓ Face aligned: {aligned.shape}")
    
    # Step-by-step preprocessing
    print("\n[3/6] Step-by-step preprocessing...")
    
    # Original (BGR, uint8, [0,255])
    step0 = aligned.copy()
    print(f"\n  Step 0: Original (aligned)")
    print(f"    Type: {type(step0)}")
    print(f"    Shape: {step0.shape}")
    print(f"    Dtype: {step0.dtype}")
    print(f"    Range: [{step0.min()}, {step0.max()}]")
    print(f"    Format: BGR")
    print(f"    Sample pixel [80,80]: {step0[80, 80]}")
    
    # Step 1: BGR → RGB
    step1 = cv2.cvtColor(step0, cv2.COLOR_BGR2RGB)
    print(f"\n  Step 1: BGR → RGB")
    print(f"    Type: {type(step1)}")
    print(f"    Shape: {step1.shape}")
    print(f"    Dtype: {step1.dtype}")
    print(f"    Range: [{step1.min()}, {step1.max()}]")
    print(f"    Format: RGB")
    print(f"    Sample pixel [80,80]: {step1[80, 80]}")
    print(f"    Change: BGR {step0[80, 80]} → RGB {step1[80, 80]}")
    
    # Step 2: uint8 → float32
    step2 = torch.from_numpy(step1).float()
    print(f"\n  Step 2: uint8 → float32")
    print(f"    Type: {type(step2)}")
    print(f"    Shape: {step2.shape}")
    print(f"    Dtype: {step2.dtype}")
    print(f"    Range: [{step2.min():.1f}, {step2.max():.1f}]")
    print(f"    Sample pixel [80,80]: {step2[80, 80]}")
    
    # Step 3: Normalize [0,255] → [-1,1]
    step3 = (step2 - 127.5) / 128.0
    print(f"\n  Step 3: Normalize [0,255] → [-1,1]")
    print(f"    Type: {type(step3)}")
    print(f"    Shape: {step3.shape}")
    print(f"    Dtype: {step3.dtype}")
    print(f"    Range: [{step3.min():.3f}, {step3.max():.3f}]")
    print(f"    Sample pixel [80,80]: {step3[80, 80]}")
    print(f"    Formula: (value - 127.5) / 128.0")
    
    # Show calculation
    original_pixel = step2[80, 80]
    normalized_pixel = step3[80, 80]
    print(f"\n    Calculation for pixel [80,80]:")
    for i, (orig, norm, channel) in enumerate(zip(original_pixel, normalized_pixel, ['R', 'G', 'B'])):
        print(f"      {channel}: ({orig:.1f} - 127.5) / 128.0 = {norm:.3f}")
    
    # Step 4: HWC → CHW
    step4 = step3.permute(2, 0, 1)
    print(f"\n  Step 4: HWC → CHW (permute dimensions)")
    print(f"    Type: {type(step4)}")
    print(f"    Shape: {step4.shape}")
    print(f"    Dtype: {step4.dtype}")
    print(f"    Range: [{step4.min():.3f}, {step4.max():.3f}]")
    print(f"    Before: (H, W, C) = (160, 160, 3)")
    print(f"    After:  (C, H, W) = (3, 160, 160)")
    print(f"    Sample pixels [*, 80, 80]:")
    print(f"      Channel 0 (R): {step4[0, 80, 80]:.3f}")
    print(f"      Channel 1 (G): {step4[1, 80, 80]:.3f}")
    print(f"      Channel 2 (B): {step4[2, 80, 80]:.3f}")
    
    # Step 5: Add batch dimension
    step5 = step4.unsqueeze(0)
    print(f"\n  Step 5: Add batch dimension")
    print(f"    Type: {type(step5)}")
    print(f"    Shape: {step5.shape}")
    print(f"    Dtype: {step5.dtype}")
    print(f"    Range: [{step5.min():.3f}, {step5.max():.3f}]")
    print(f"    Before: (C, H, W) = (3, 160, 160)")
    print(f"    After:  (B, C, H, W) = (1, 3, 160, 160)")
    print(f"    Sample pixels [0, *, 80, 80]:")
    print(f"      Batch 0, Channel 0 (R): {step5[0, 0, 80, 80]:.3f}")
    print(f"      Batch 0, Channel 1 (G): {step5[0, 1, 80, 80]:.3f}")
    print(f"      Batch 0, Channel 2 (B): {step5[0, 2, 80, 80]:.3f}")
    
    # Final tensor
    print(f"\n  ★ FINAL MODEL INPUT ★")
    print(f"    Type: torch.Tensor")
    print(f"    Shape: (1, 3, 160, 160)")
    print(f"    Dtype: float32")
    print(f"    Range: [-1, 1]")
    print(f"    Format: RGB")
    print(f"    Total elements: {1 * 3 * 160 * 160:,}")
    
    # Extract embedding
    print("\n[4/6] Extracting embedding...")
    embedding = embedder.extract(aligned, check_quality=False)
    
    if embedding:
        print(f"✓ Embedding extracted")
        print(f"  Length: {len(embedding)}")
        print(f"  L2 norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Range: [{min(embedding):.3f}, {max(embedding):.3f}]")
    else:
        print("✗ Embedding extraction failed")
    
    # Visualize
    print("\n[5/6] Creating visualization...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Original steps
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(step0, cv2.COLOR_BGR2RGB))
    ax1.set_title('Step 0: Original\n(160,160,3) BGR uint8 [0,255]', fontsize=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(step1)
    ax2.set_title('Step 1: BGR→RGB\n(160,160,3) RGB uint8 [0,255]', fontsize=10)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(step1)  # Same visual, different dtype
    ax3.set_title('Step 2: uint8→float32\n(160,160,3) RGB float32 [0,255]', fontsize=10)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    # Denormalize for display
    step3_display = ((step3.numpy() + 1) * 127.5).astype(np.uint8)
    ax4.imshow(step3_display)
    ax4.set_title('Step 3: Normalize\n(160,160,3) RGB float32 [-1,1]', fontsize=10)
    ax4.axis('off')
    
    # Row 2: Channel separation
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(step4[0].numpy(), cmap='Reds')
    ax5.set_title('Step 4: Red Channel\n(160,160) [-1,1]', fontsize=10)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(step4[1].numpy(), cmap='Greens')
    ax6.set_title('Step 4: Green Channel\n(160,160) [-1,1]', fontsize=10)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(step4[2].numpy(), cmap='Blues')
    ax7.set_title('Step 4: Blue Channel\n(160,160) [-1,1]', fontsize=10)
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    # Reconstruct RGB from CHW
    step4_rgb = step4.permute(1, 2, 0).numpy()
    step4_rgb_display = ((step4_rgb + 1) * 127.5).astype(np.uint8)
    ax8.imshow(step4_rgb_display)
    ax8.set_title('Step 4: CHW Format\n(3,160,160) [-1,1]', fontsize=10)
    ax8.axis('off')
    
    # Row 3: Final tensor and embedding
    ax9 = fig.add_subplot(gs[2, 0:2])
    # Show tensor as heatmap
    tensor_flat = step5[0].reshape(3, -1).numpy()
    im = ax9.imshow(tensor_flat, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax9.set_title('Step 5: Final Tensor (1,3,160,160)\nFlattened view', fontsize=10)
    ax9.set_xlabel('Spatial dimension (160×160)')
    ax9.set_ylabel('Channel (R, G, B)')
    plt.colorbar(im, ax=ax9, label='Value')
    
    if embedding:
        ax10 = fig.add_subplot(gs[2, 2:4])
        emb_array = np.array(embedding)
        ax10.plot(emb_array, linewidth=0.5, alpha=0.7)
        ax10.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax10.set_title(f'Embedding (512-dim)\nL2 norm={np.linalg.norm(emb_array):.3f}', 
                      fontsize=10)
        ax10.set_xlabel('Dimension')
        ax10.set_ylabel('Value')
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim([-1, 1])
    
    plt.suptitle('Preprocessing Pipeline Visualization', fontsize=14, fontweight='bold')
    plt.savefig('preprocessing_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: preprocessing_visualization.png")
    
    # Summary
    print("\n[6/6] Summary")
    print("="*70)
    print("\nPreprocessing transforms:")
    print("  Input:  (160, 160, 3) uint8 BGR [0, 255]")
    print("  Output: (1, 3, 160, 160) float32 RGB [-1, 1]")
    print("\nSteps:")
    print("  1. BGR → RGB: Color space conversion")
    print("  2. uint8 → float32: Data type conversion")
    print("  3. [0,255] → [-1,1]: Normalization")
    print("  4. HWC → CHW: Dimension permutation")
    print("  5. Add batch: (C,H,W) → (B,C,H,W)")
    print("\nModel input:")
    print("  torch.Tensor(1, 3, 160, 160) float32 RGB [-1, 1]")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    
    plt.show()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python demo_preprocessing.py <image_path>")
        print("\nExample:")
        print("  python demo_preprocessing.py test_face.jpg")
        print("  python demo_preprocessing.py Silent-Face-Anti-Spoofing/images/sample/image_F1.jpg")
        return
    
    image_path = sys.argv[1]
    visualize_preprocessing(image_path)


if __name__ == "__main__":
    main()
