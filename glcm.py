import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GLCMAnalyzer:
    def __init__(self, image_path):
        self.image = self._image_to_gray(image_path)
        self.features = {}
        self.heatmaps = {}
        self.surface_plots = {}
        
    @staticmethod
    def _image_to_gray(image_path):
        return np.array(Image.open(image_path).convert('L'))

    def compute_glcm(self, dx=1, dy=0, levels=256):

        glcm = np.zeros((levels, levels), dtype=np.float64)
        rows, cols = self.image.shape

        for i in range(rows - abs(dy)):
            for j in range(cols - abs(dx)):
                val1 = self.image[i, j]
                val2 = self.image[i + dy, j + dx]
                glcm[val1, val2] += 1

        # symmetrize 
        glcm = glcm + glcm.T
        glcm /= glcm.sum() if glcm.sum() > 0 else 1
        return glcm

    @staticmethod
    def extract_features(glcm):
        eps = 1e-12 #this is don to avoid numerical instability
        i, j = np.indices(glcm.shape)
        features = {}
        
        #featur. THESE ARE THE HARACLICK FEATURES.
        features['contrast'] = np.sum(glcm * (i - j)**2)
        features['dissimilarity'] = np.sum(glcm * np.abs(i - j))
        features['homogeneity'] = np.sum(glcm / (1 + np.abs(i - j)))
        features['energy'] = np.sum(glcm**2)
        
        #common features
        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum(glcm * (i - mean_i)**2))
        std_j = np.sqrt(np.sum(glcm * (j - mean_j)**2))
        
        features['correlation'] = np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j + eps)
        features['entropy'] = -np.sum(glcm * np.log2(glcm + eps))
        #Higher order statistics.
        features['variance'] = np.sum(glcm * (i - mean_i)**2)
        features['sum_average'] = np.sum(glcm * (i + j))
        features['sum_variance'] = np.sum(glcm * ((i + j) - features['sum_average'])**2)
        features['sum_entropy'] = -np.sum(glcm * np.log2(glcm + eps) * (i + j))
        features['diff_variance'] = np.var(np.sum(glcm, axis=0))
        features['diff_entropy'] = -np.sum(np.sum(glcm, axis=0) * np.log2(np.sum(glcm, axis=0) + eps))
        
       
        hxy = features['entropy']
        hx = -np.sum(np.sum(glcm, axis=1) * np.log2(np.sum(glcm, axis=1) + eps))
        hy = -np.sum(np.sum(glcm, axis=0) * np.log2(np.sum(glcm, axis=0) + eps))
        hxy1 = -np.sum(glcm * np.log2(np.outer(np.sum(glcm, axis=1), np.sum(glcm, axis=0)) + eps))
        features['icm1'] = (hxy - hxy1) / max(hx, hy) if max(hx, hy) > 0 else 0
        
        return features

    def visualize_glcm(self, glcm, save_path=None):
        
        plt.figure(figsize=(8, 6))
        plt.imshow(glcm, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.title("GLCM Heatmap")
        if save_path:
            plt.savefig(save_path + '_2d.png', bbox_inches='tight')
        plt.close()
        
      
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.indices(glcm.shape)
        ax.plot_surface(x, y, glcm, cmap='viridis')
        ax.set_title("3D GLCM Visualization")
        if save_path:
            plt.savefig(save_path + '_3d.png', bbox_inches='tight')
        plt.close()
