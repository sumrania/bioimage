# Biological Image Classification


Multi-class classificiation using active learning of fluorescent microscopy images based on their subcellular localization patterns.

* [Report](report.pdf)

The algorithm I used is a Query-by-committee based approach as explained by [Nigam and McCallum](nigampdf). Ideally an active learning algorithm chooses only those points whichwill minimize the error upon querying their label. In QBC, this error is approximated by a group of features.