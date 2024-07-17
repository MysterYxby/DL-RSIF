# The implementation of remote sensing images pan sharpening methods
Replication of Remote Sensing Image Pansharpening Task Code and Comparison of Evaluation Metrics.（遥感图像Pansharpening任务代码复现与指标对比）
## Code Illustration
~~

## Abbreviation Explanation
| Abbreviation | Explanation |
|:-------:|:-------:|
|ICCV|IEEE International Conference on Computer Vision|
|JSTARS|IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing|
|LGRS|IEEE Geoscience and Remote Sensing Letters|
|TGRS|IEEE Transactions on Geoscience and Remote Sensing|
|NMSB|Number of Multi-Spectral Bands|
|GSD|Ground Sample Distance|
|RRT|Reduced Resolution Testing|
|FRT|Full Resolution Testing Set|

## Methods
| Methods | Year | Author | From | 
|:-------:|:-------:|:-------:|:-------:|
| PNN | 2016 | Masi et al. | [Remote Sensing](https://www.mdpi.com/2072-4292/8/7/594) | 
|DRPNN| 2017 |Wei et al. | [LGRS](10.1109/LGCRS.2017.2736020) |
|MSDCNN| 2018 |Yuan et al.|[JSTARS](10.1109/JSTARS.2018.2794888)| 
| PanNet | 2017 | Yang et al. | [ICCV](https://arxiv.org/abs/1908.05900) |
| DiCNN | 2019 | He et al. | [JSTARS](https://ieeexplore.ieee.org/document/8667040) |
| BDPN  | 2019 |Zhang et al.|[TGRS](10.1109/TGRS.2019.2900419) |
|FusionNet |2021 |Deng el al.| [TGRS](10.1109/TGRS.2020.3031366) |
## Datasets

| Name | NMSB| Type | GSD | Sets | Sample Number | Image Size | From |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|GaoFen-2|4|PAN<br>MS|0.8m<br>3.2m|Training<br>Validation<br>RRT<br>FRT|19809<br>2201<br>20<br>20|$64 \times 64 \times 1, 64 \times 64 \times 4$<br>$64 \times 64 \times 1, 64 \times 64 \times 4$<br>$256 \times 256 \times 1, 256 \times 256 \times 4$<br>$512 \times 512 \times 1, 512 \times 512 \times 4$|[1]|

### Reference
- [1] L.-j. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza, “Machine learning in pansharpening: A benchmark, from shallow to deep networks,” IEEE Geoscience and Remote Sensing Magazine, p. 279–315, Sep 2022.
- [2] demo

## Quality Assessment Indices
~~

