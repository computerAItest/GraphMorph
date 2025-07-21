# GraphMorph: Equilibrium Adjustment Regularized Dual-Stream GCN for 4D-CT Lung Imaging with Sliding Motion
We propose an Equilibrium Adjustment Regularized Dual-Stream Graph Convolutional Network (GraphMorph) to address the sliding motion problem in 4D-CT lung registration. This approach ensures physically consistent deformations while balancing smoothness and discontinuity at sliding interfaces. By incorporating the Adaptive Graph Convolutional Attention (AGCA) module and the Cross-Scale Contextual-Aware Aggregation (CSCA) module, GraphMorph significantly improves registration accuracy in sliding regions. Experimental results demonstrate that GraphMorph achieves an impressive average target registration error (TRE) of 0.96mm.
In this study, we have constructed a standardized 4D-CT lung multi-phase dataset to promote future research in adaptive radiation therapy for lung cancer. We are currently anonymizing the data to protect patient privacy by removing any private information. We are also further organizing and converting the data format to ensure quicker and easier access. We have already uploaded three sets of data for testing at the maximum inspiration and expiration phases, and additional data will be uploaded shortly.

# Dataset
## Inclusion Criteria
The in-house clinical dataset includes several inclusion criteria: (a) each patient underwent multiple sets of lung CT images at different phases during treatment; (b) the lesion volume must be greater than 5mm; (c) preoperative planning CT and admission CT imaging results were recorded; (d) images with significant motion artifacts were excluded.
## Data Modality
The in-house clinical dataset includes paired CT imaging pre-processing scans taken at different time points for patients, including preoperative planning CT and multi-phase CT data acquired during surgery using respiratory gating techniques.
## Landmark Annotation
The landmarks we used are primarily based on key vascular or tracheal junctions and prominent anatomical features, which are identified from imaging the same patient at different time points. These landmarks demonstrate high consistency across different evaluators.
We conducted an inter-rater consistency analysis to validate the reliability of the manual contour/landmark annotations. Specifically, we calculated the Kappa coefficient by comparing the annotations made by multiple evaluators to assess the level of agreement. If the Kappa value is below 0.6, we will further review and refine the annotations. If the Kappa value is 0.6 or above, it indicates significant consistency, and the annotations are considered reliable for subsequent analysis.
<img src="https://github.com/computerAItest/GraphMorph/blob/main/GraphMorph/data/landmark.pdf?raw=true" width="900" alt="demo"/><br/>

## 4D-CT Clinical Data Demo
<img src="https://github.com/computerAItest/GraphMorph/blob/main/GraphMorph/data/data_Demo.png?raw=true" width="900" alt="demo"/><br/>

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/computerAItest/GraphMorph/blob/main/GraphMorph/data/4d_slice_registered.gif?raw=true" width="300" alt="demo"/>
    <img src="https://github.com/computerAItest/GraphMorph/blob/main/GraphMorph/data/4d_volume_registered.gif?raw=true" width="300" alt="demo"/>
</div>

Post-registration 4D-CT 2D rendering  &nbsp;&nbsp; Post-registration 4D-CT 3D rendering

