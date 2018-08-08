# Neural Article Pair Modeling for Wikipedia Sub-article Matching

This repository is used for the resources of the following paper.

Muhao Chen, Changping Meng, Gang Huang, Carlo Zaniolo. Neural Article Pair Modeling for Wikipedia Sub-article Matching. In *ECML-PKDD*, 2018.

    @inproceedings{chen2018subarticle,
	  title={Neural Article Pair Modeling for Wikipedia Sub-article Matching},
	  author={Chen, Muhao and Meng, Changping and Huang, Gang and Xue, Zijun and Zaniolo, Carlo},
	  booktitle={ECML-PKDD},
	  year={2018},
	  organization={Springer}
	}

## Dataset

A simplified version of the dataset is available at [this link](http://yellowstone.cs.ucla.edu/~muhao/subarticle/WAP-196k-simplified.zip). This large dataset contains over 190k pairs of wikipedia articles and a portion of features described in our paper. If you decide to use such resources, you have to agree and comply with the attached license.

## Code

Make sure you have following required packages installed:

    Python 2.7
    Tensorflow >= 1.2.1
    Keras that supports corresponding Tensorflow version

To run the cross-validation code, you also need to download these accompanying files from [this link](http://yellowstone.cs.ucla.edu/~muhao/subarticle/accomp/).
