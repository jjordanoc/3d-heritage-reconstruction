Optimizing point clouds for mobile consumption renders data reduction imperative. The primary challenge lies in significantly reducing the point count to accommodate resource-constrained environments while preserving the perceptual fidelity of the original geometry. To address this, our implementation necessitates a robust point cloud simplification method. We evaluate two distinct downsampling strategies:

* Uniform downsample: A stochastic approach that uniformly selects a subset of points from the original cloud. While computationally efficient, this method is agnostic to geometric features and may degrade structural details in low-density regions.

* Chordline-scaled Voxel Simplification: A variant of the standard voxel grid filter. In this method, the space is subdivided into a 3D grid of cubic volume elements (voxels). All points falling within a voxel are approximated by their centroid. Our implementation scales the voxel dimensions based on chordline estimates, allowing for adaptive density reduction that better preserves underlying surface structures compared to uniform grid approaches.


## Dataset and Evaluation Metrics

We utilize the full point clouds generated from the Parque San Martin and UTEC Auditorium datasets as the baseline for testing these simplification methods. Each experiment consists of applying one of the downsampling algorithms to the baseline cloud and subsequently evaluating the geometric degradation using PointcloudSSIM (Alexiou & Ebrahimi, "Towards a Point Cloud Structural Similarity Metric").

For this study, we utilize the implementation provided by Oliveira et al. (https://github.com/tjoliveira/point-cloud-ssim). While the software calculates a matrix of various estimators (including std, var, mean_ad, median_ad, and qcd) and pooling methods (mean, mse, rms), we specifically isolate the Coefficient of Variation (coef_var) as the estimator and the Mean as the pooling operation. We selected the coef_var because it provides a relative measure of dispersion, making it particularly robust for evaluating simplification tasks where absolute density changes, but relative geometric structures must be preserved. 

## Experimental Setup and Registration

To ensure that the metric reflects geometric similarity rather than misalignment errors, a rigorous registration pipeline was employed for all experiments. The procedure was conducted as follows:

Both the source and reference point clouds were initially downsampled using a voxel size of 0.050. Following this, surface normals were estimated with a search radius of 0.100, and Fast Point Feature Histograms (FPFH) were computed utilizing a search radius of 0.250.

With the features extracted, a coarse alignment was performed using RANSAC registration on the downsampled clouds. Given the downsampling voxel size of 0.050, a liberal distance threshold of 0.075 was applied to maximize convergence probability. Finally, to achieve precise alignment, a Point-to-Plane Iterative Closest Point (ICP) registration was applied to the original (non-downsampled) point clouds. For this refinement stage, a strict distance threshold of 0.020 was enforced to ensure tight geometric correspondence.

We selected the parameters for number of points in such a way that they return simplified clouds with up to a 10% variance in the quantity of points with respect to each other in the scene. We provide the number of points for each combination of experiment and simplification result.

**Table 1: Simplification Results for Parque San Martin Dataset**

| Method | Point Count ($N$) | Reduction Ratio | SSIM Score ($\rho$) |
| :--- | :--- | :---: | :---: |
| **Original (Baseline)** | 7,355,286 | 0% | 1.000 |
| **Uniform Downsample** | 959,105 | 87.0% | 0.752 |
| **Scaled Voxel** | 961,830 | 86.9% | **0.763** |

**Table 2: Simplification Results for Auditorio UTEC Dataset**

| Method | Point Count ($N$) | Reduction Ratio | SSIM Score ($\rho$) |
| :--- | :--- | :---: | :---: |
| **Original (Baseline)** | 8,719,138 | 0% | 1.000 |
| **Uniform Downsample** | 3,677,643 | 57.8% | 0.824 |
| **Scaled Voxel** | 3,467,043 | 60.2% | **0.851** |

We find that Voxel simplification preserves better detail than a simple uniform decimation. We believe that the mechanism that leads to this is the fact that the effect of quasi-random removal of points affects low density regions disproportionatelly, wich in turn produces a higher loss of fidelity. We note that voxel performed particularly well on outside conditions where far away objects with a lower density of points are more common. Furthermore, while not presented on this study we found that othermmethods such as farthest point simplification require a disproportionate ammount of computational time, wich invalidates its use for our platform.

## Appendix A: Full SSIM Result Matrices

The following matrices represent the complete output from the PointcloudSSIM evaluation. The rows correspond to the estimators: ['std', 'var', 'mean_ad', 'median_ad', 'coef_var', 'qcd']. The columns correspond to the pooling methods: ['mean', 'mse', 'rms']. The values used in the main text correspond to the intersection of 'coef_var' (Row 5) and 'mean' (Column 1).

**A.1 Parque San Martin: Scaled Voxel**
| Est \ Pool | Mean | MSE | RMS |
| :--- | :--- | :--- | :--- |
| std | 0.3380 | 0.1773 | 0.4211 |
| var | 0.1773 | 0.0966 | 0.3108 |
| mean_ad | 0.3348 | 0.1751 | 0.4185 |
| median_ad | 0.3099 | 0.1601 | 0.4001 |
| **coef_var** | **0.7632** | 0.6067 | 0.7789 |
| qcd | 0.6964 | 0.5230 | 0.7232 |

**A.2 Parque San Martin: Uniform Downsample**
| Est \ Pool | Mean | MSE | RMS |
| :--- | :--- | :--- | :--- |
| std | 0.4116 | 0.1985 | 0.4455 |
| var | 0.1985 | 0.0709 | 0.2664 |
| mean_ad | 0.4147 | 0.2039 | 0.4515 |
| median_ad | 0.4171 | 0.2242 | 0.4735 |
| **coef_var** | **0.7519** | 0.5911 | 0.7688 |
| qcd | 0.6501 | 0.4667 | 0.6831 |

**A.3 Auditorio UTEC: Uniform Downsample**
| Est \ Pool | Mean | MSE | RMS |
| :--- | :--- | :--- | :--- |
| std | 0.6991 | 0.5107 | 0.7147 |
| var | 0.5107 | 0.3022 | 0.5497 |
| mean_ad | 0.6947 | 0.5063 | 0.7115 |
| median_ad | 0.6347 | 0.4473 | 0.6688 |
| **coef_var** | **0.8237** | 0.6955 | 0.8340 |
| qcd | 0.7007 | 0.5294 | 0.7276 |

**A.4 Auditorio UTEC: Scaled Voxel**
| Est \ Pool | Mean | MSE | RMS |
| :--- | :--- | :--- | :--- |
| std | 0.6925 | 0.5312 | 0.7289 |
| var | 0.5312 | 0.3702 | 0.6085 |
| mean_ad | 0.6831 | 0.5195 | 0.7208 |
| median_ad | 0.6341 | 0.4635 | 0.6808 |
| **coef_var** | **0.8515** | 0.7411 | 0.8609 |
| qcd | 0.7554 | 0.6055 | 0.7781 |