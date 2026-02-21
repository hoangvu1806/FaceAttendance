# -*- coding: utf-8 -*-
"""
Visualize Face Embeddings - Hiển thị embedding space
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from db import EmbeddingDB
import json


def load_embeddings():
    """Load embeddings từ database"""
    db = EmbeddingDB()
    data = db.get_all()
    
    embeddings = []
    labels = []
    names = []
    
    for name, emb_list in data.items():
        # Handle both formats
        if isinstance(emb_list[0], list):
            person_embeddings = emb_list
        else:
            person_embeddings = [emb_list]
        
        for emb in person_embeddings:
            embeddings.append(emb)
            labels.append(name)
            names.append(name)
    
    return np.array(embeddings), labels, list(set(names))


def visualize_tsne(embeddings, labels, names):
    """Visualize embeddings using t-SNE"""
    print("Running t-SNE (this may take a while)...")
    
    # t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(names)))
    name_to_color = {name: colors[i] for i, name in enumerate(names)}
    
    # Plot each person
    for name in names:
        mask = np.array(labels) == name
        points = embeddings_2d[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[name_to_color[name]], 
                   label=name, 
                   s=100, 
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=1)
        
        # Draw cluster boundary
        if len(points) > 2:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], 
                            'k-', alpha=0.3, linewidth=0.5)
            except:
                pass
    
    plt.title('Face Embeddings Visualization (t-SNE)', fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig('embeddings_tsne.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: embeddings_tsne.png")
    
    plt.show()


def visualize_pca(embeddings, labels, names):
    """Visualize embeddings using PCA"""
    print("Running PCA...")
    
    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(names)))
    name_to_color = {name: colors[i] for i, name in enumerate(names)}
    
    # Plot each person
    for name in names:
        mask = np.array(labels) == name
        points = embeddings_2d[mask]
        
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[name_to_color[name]], 
                   label=name, 
                   s=100, 
                   alpha=0.6,
                   edgecolors='black',
                   linewidth=1)
    
    plt.title(f'Face Embeddings Visualization (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2%}', 
             fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig('embeddings_pca.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: embeddings_pca.png")
    
    plt.show()


def analyze_embeddings(embeddings, labels, names):
    """Analyze embedding statistics"""
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS")
    print("="*60)
    
    print(f"\nTotal embeddings: {len(embeddings)}")
    print(f"Total persons: {len(names)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Per-person statistics
    print("\n" + "-"*60)
    print("Per-Person Statistics:")
    print("-"*60)
    
    for name in names:
        mask = np.array(labels) == name
        person_embs = embeddings[mask]
        
        # Intra-class similarity
        if len(person_embs) > 1:
            sims = []
            for i in range(len(person_embs)):
                for j in range(i+1, len(person_embs)):
                    sim = np.dot(person_embs[i], person_embs[j])
                    sims.append(sim)
            
            avg_sim = np.mean(sims)
            std_sim = np.std(sims)
            min_sim = np.min(sims)
            max_sim = np.max(sims)
            
            print(f"\n{name}:")
            print(f"  Samples: {len(person_embs)}")
            print(f"  Intra-class similarity: {avg_sim:.4f} ± {std_sim:.4f}")
            print(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
        else:
            print(f"\n{name}:")
            print(f"  Samples: {len(person_embs)}")
            print(f"  (Need >1 sample for similarity)")
    
    # Inter-class similarity
    print("\n" + "-"*60)
    print("Inter-Class Similarity Matrix:")
    print("-"*60)
    
    if len(names) > 1:
        # Calculate average embedding per person
        avg_embeddings = {}
        for name in names:
            mask = np.array(labels) == name
            avg_embeddings[name] = np.mean(embeddings[mask], axis=0)
        
        # Similarity matrix
        print("\n" + " "*15, end="")
        for name in names:
            print(f"{name[:10]:>12}", end="")
        print()
        
        for name1 in names:
            print(f"{name1[:15]:15}", end="")
            for name2 in names:
                sim = np.dot(avg_embeddings[name1], avg_embeddings[name2])
                print(f"{sim:12.4f}", end="")
            print()
    
    # Overall statistics
    print("\n" + "-"*60)
    print("Overall Statistics:")
    print("-"*60)
    
    # L2 norms (should all be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {np.mean(norms):.6f}")
    print(f"  Std: {np.std(norms):.6f}")
    print(f"  Range: [{np.min(norms):.6f}, {np.max(norms):.6f}]")
    
    # Embedding statistics
    print(f"\nEmbedding Values:")
    print(f"  Mean: {np.mean(embeddings):.6f}")
    print(f"  Std: {np.std(embeddings):.6f}")
    print(f"  Range: [{np.min(embeddings):.6f}, {np.max(embeddings):.6f}]")
    
    # Sparsity
    zero_ratio = np.sum(np.abs(embeddings) < 0.01) / embeddings.size
    print(f"\nSparsity:")
    print(f"  Near-zero values: {zero_ratio:.2%}")


def main():
    print("="*60)
    print("Face Embeddings Visualization")
    print("="*60)
    
    # Load embeddings
    print("\nLoading embeddings from database...")
    embeddings, labels, names = load_embeddings()
    
    if len(embeddings) == 0:
        print("No embeddings found in database!")
        print("Please enroll some users first using the main app.")
        return
    
    print(f"✓ Loaded {len(embeddings)} embeddings from {len(names)} persons")
    
    # Analyze
    analyze_embeddings(embeddings, labels, names)
    
    # Visualize
    if len(embeddings) >= 3:  # Need at least 3 points for visualization
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        
        try:
            visualize_pca(embeddings, labels, names)
        except Exception as e:
            print(f"PCA visualization failed: {e}")
        
        try:
            visualize_tsne(embeddings, labels, names)
        except Exception as e:
            print(f"t-SNE visualization failed: {e}")
    else:
        print("\nNeed at least 3 embeddings for visualization")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
