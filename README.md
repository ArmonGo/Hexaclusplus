# Hexaclus++

This repository contains the implementation for the models included in the experimental comparison as presented in:

 HexaClus++: Uncertainty-Aware Hexagonal Supervised Spatial Clustering

**Abstract**: Due to the heterogeneity and uneven distribution of geospatial data, predictive geospatial techniques are often required to provide both accurate target predictions and uncertainty quantification. However, existing approaches typically only provide post-hoc uncertainty estimates, rather than using uncertainty to guide the spatial model training process itself. To address these limitations, we introduce HexaClus++, a supervised, uncertainty-aware, spatial-clustering-based approach that integrates both accuracy and uncertainty to guide local model training on clustered spatial hexagonal regions. To evaluate the performance of our technique, we conduct experiments on property valuation datasets using diverse machine learning approaches. The results demonstrate that HexaClus++, when combined with different base learners, achieves competitive RMSE values while also delivering superior uncertainty calibration, reliable prediction intervals, and enhanced interpretability through regional uncertainty maps and spatially varying feature importance.

## Data set 

For the data sets used in the paper, see

[**London property prices**](https://www.kaggle.com/datasets/jakewright/house-price-data)

[**New York property prices**](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)

[**Paris property prices**](https://www.kaggle.com/datasets/benoitfavier/immobilier-france)

Data retrieval date: 2026-08-25
