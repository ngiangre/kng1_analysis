# kng1_analysis

[![DOI](https://zenodo.org/badge/424333512.svg)](https://zenodo.org/badge/latestdoi/424333512)


Code repository (meant to be viewed and not executed) for 'Alterations in the kallikrein-kinin system predicts death after heart transplant' by Giangreco et al.

The clinical characteristics table-one tables in the paper were created using tableone.R

The prediction methods and computations were performed on a multicluster machine using: 

* `survival_analysis_individual_marker_predictions.py` for overall survival, 

* `survival_analysis_individual_marker_predictions_wn_year.py` for 1-year survival, 

* `survival_analysis_individual_marker_predictions_wpgdcov.py` for overall survival accounting for PGD status,

* `survival_analysis_individual_marker_predictions_pgd.py` for PGD, and  

* `survival_analysis_individual_marker_predictions_wcovs.py` for overall survival accounting for site-of-origin. 

The results and figures were generated within paper_figure_code.ipynb. Further figure collation was done using Biorender.
