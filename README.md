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
|IF| Information Fusion |
|NMSB|Number of Multi-Spectral Bands|
|GSD|Ground Sample Distance|
|RRT|Reduced Resolution Testing|
|FRT|Full Resolution Testing|

## Methods
| Methods | Year | Author | From | 
|:-------:|:-------:|:-------:|:-------:|
| PNN | 2016 | Masi et al. | [Remote Sensing](https://doi.org/10.3390/rs8070594) | 
|DRPNN| 2017 |Wei et al. | [LGRS](https://doi.org/10.1109/LGRS.2017.2736020) |
| PanNet | 2017 | Yang et al. | [ICCV](https://doi.org/10.1109/ICCV.2017.193) |
|MSDCNN| 2018 |Yuan et al.|[JSTARS](https://doi.org/10.1109/IGARSS.2017.8127731)| 
| DiCNN | 2019 | He et al. | [JSTARS](https://doi.org/10.1109/JSTARS.2019.2898574) |
| BDPN  | 2019 |Zhang et al.|[TGRS](https://doi.org/10.1109/TGRS.2019.2900419) |
|FusionNet |2021 |Deng et al.| [TGRS](https://doi.org/10.1109/TGRS.2020.3031366) |
|P2Sharpen|2022| Zhang  et al.|[IF](https://doi.org/10.1016/j.inffus.2022.10.010)|
|L-PNN|2023| Ciotola et al.| [TGRS](https://doi.org/10.1109/TGRS.2023.3299356)|
## Datasets

| Name | NMSB| Type | GSD | Sets | Sample Number | Image Size | From |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|GaoFen-2|4|PAN<br>MS|0.8m<br>3.2m|Training<br>Validation<br>RRT<br>FRT|19809<br>2201<br>20<br>20|$64 \times 64 \times 1, 16 \times 16 \times 4$<br>$64 \times 64 \times 1, 16 \times 16 \times 4$<br>$256 \times 256 \times 1, 64 \times 64 \times 4$<br>$512 \times 512 \times 1, 128 \times 128 \times 4$|[1]|
|QuickBird|4|PAN<br>MS|0.61m<br>2.44m|Training<br>Validation<br>RRT<br>FRT|17139<br>1905<br>20<br>20|$64 \times 64 \times 1, 16 \times 16 \times 4$<br>$64 \times 64 \times 1, 16 \times 16 \times 4$<br>$256 \times 256 \times 1, 64 \times 64 \times 4$<br>$512 \times 512 \times 1, 128 \times 128 \times 4$|[1]|
|WorldView-2|8|PAN<br>MS|0.46m<br>1.84m|Training<br>Validation<br>RRT<br>FRT|-<br>-<br>20<br>20|$--, --$<br>$--, --$<br>$256 \times 256 \times 1, 64 \times 64 \times 4$<br>$512 \times 512 \times 1, 128 \times 128 \times 4$|[1]|
|Worldview-3|8|PAN<br>MS|0.31m<br>1.24m|Training<br>Validation<br>RRT<br>FRT|9714<br>1080<br>20<br>20|$64 \times 64 \times 1, 16 \times 16 \times 4$<br>$64 \times 64 \times 1, 16 \times 16 \times 4$<br>$256 \times 256 \times 1, 64 \times 64 \times 4$<br>$512 \times 512 \times 1, 128 \times 128 \times 4$|[1]|
## Quality Assessment Indices
~~
## Reference
- [1] L.-j. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza, “Machine learning in pansharpening: A benchmark, from shallow to deep networks,” IEEE Geoscience and Remote Sensing Magazine, p. 279–315, Sep 2022.
- [2] demo
