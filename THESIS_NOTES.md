# Thesis Notes & Justifications

## Chapter 4 – Data & Preprocessing

### Dataset
- Name: Urban Traffic Speed Dataset – Guangzhou, China
- Source: Zenodo DOI: 10.5281/zenodo.1205229
- Size: 1,855,572 observations after cleaning
- Roads: 214 urban segments
- Period: 2016-08-01 to 2016-09-30
- Interval: 10 minutes

### Outlier Removal
- Removed 17 rows with speed > 100 km/h
- Justification: sensor unreliability at extreme speeds
  in urban context, 0.0009% of data – negligible impact

### Congestion Threshold – 30 km/h
- International standards vary 20-30 km/h
  (Rao & Rao, 2012)
- China official standard: below 20 km/h
  (Su et al., 2019 – GJG[2008] 262)
- We chose 30 km/h because:
  1. Enables early warning before official
     congestion develops
  2. 20 km/h captures only 5.3% → extreme
     class imbalance, unsuitable for ML
  3. 30 km/h captures 18.6% → balanced
     and meaningful for binary classification
- Threshold analysis:
  Below 20 km/h: 5.3%
  Below 25 km/h: 10.2%
  Below 30 km/h: 18.6%
  Below 35 km/h: 33.4%
  Below 40 km/h: 52.5%

### EDA Key Findings
- Mean speed: 39.0 km/h
- Peak congestion: 18:00 weekday (59%)
- Weekend congestion lower and smoother
- Road 147 most congested (71.6%)
- Road 154 least congested (0.2%)
- Class distribution: 18.6% congestion,
  81.4% no congestion

## Chapter 5 – Models
(add notes here)

## Chapter 6 – Evaluation
(add results here)