# Plant Tissue Segmentation with Machine Learning

This repository contains the code and analysis used in our study:  
**"Using Machine Learning and Neural Networks to Draft a Blueprint of Plant Tissues."**

We compare manual tracing with **Ilastik's** trace on plant microscopy images to evaluate accuracy, speed, and scalability for cell boundary detection and morphological analysis.

## Contents

📁 Contents
- biological_analysis_figures/ – Plots and visuals from biological tissue analysis 
- cell_analysis_figures/ – Figures focused on cell shape, size, and amount of cell
- connected_skeletons/ – Skeletonized versions of segmented images with connected components
- final-test-set/ – Dataset used for final evaluation and benchmarking (manual traces, skeleton maps, microscopy, Ilastik segementations, ... )
- manual_scripts/ – Python scripts for finding areas per cell regions
- manual_trace/ – Hand-traced segmentations used as ground truth
- publication_figures/ – Figures of the overlap analysis
- simple_segmentation/ – Ilastik basic segmentation outputs
- skeletized/ – Skeleton maps derived from original segmentation results
- square_analysis_figures/ – Area distribution plots, cell counting results anf overlap analyis summary for each image

📄 Key Scripts
- analysis_based_on_area.py – Analyzes segmented regions based on area (and count cell)
- batch_analysis.py / batch_analysis_F1.py – Batch processing and Dice/F1 score calculation
- blob_to_trace.py – Converts blob-like segmentation into cleaner cell outlines
- connect_endpoints.py – Closes open segments in skeletons by connecting endpoints
- manual_area_image.py – Compares manual tracing with Ilastik and skeleton maps using visual error maps and quantitative analysis
- manual_images.py – Iterates through all images, applies the comparison pipeline, and generates summary statistics and combined visualizations.
- neural_net_comparison.py – Generates a multi-panel summary figure comparing neural network and traditional segmentation methods.

## Methods

- **Manual Tracing**: Used as the gold standard for evaluating segmentation accuracy  
- **Ilastik**: Pixel classification based on user-labeled regions  
- **Skeletonization**: Used for vector representation and post-processing analysis

## Metrics

We assessed performance using:
- Dice coefficient
- Over/under segmentation heatmaps
- Cell count accuracy
- Cell size distribution comparison

## How to Run

1. Clone the repository:
git clone https://github.com/Cl0t1lde/OpenSourceMicroscopyImages.git
cd OpenSourceMicroscopyImages

2. (Optional but recommended) Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate (On Windows: venv\Scripts\activate)

3. Install dependencies:
pip install -r requirements.txt

4. Run the main analysis pipeline:
python manual_images.py

5. To run any other script:
python script_name.py (replace script_name with the desired filename)

## Results

Executing the analysis pipeline generates detailed visualizations and quantitative summaries that compare manual and automated segmentation methods, providing insights into accuracy, error types, and morphological features of plant tissues.

## Contact & Citation

For questions, collaboration, or support, please contact:

Aitor Arias Sánchez - a.ariassanchez@student.maastrichtuniversity.nl

Josephine Benoist - j.benoist@student.maastrichtuniversity.nl

Rafael Diederen - r.diederen@student.maastrichtuniveristy.nl

Clotilde Papot - c.papot@student.maastrichtuniversity.nl

Lionel Stijns - l.stijns@student.maastrichtuniversity.nl

This codebase was developed for the analyses in the following study:

Using Machine Learning and Neural Networks to Draft a Blueprint of Plant Tissues
Aitor Arias Sanchez, Clotilde Papot, Josephine Benoist, Rafael Diederen, Lionel Stijns. 
Supervised by Ruth Grosseholz
(June 11, 2025)

