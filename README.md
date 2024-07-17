@[toc]
# The implementation of remote sensing images pan sharpening methods
Replication of Remote Sensing Image Pansharpening Task Code and Comparison of Evaluation Metrics.（遥感图像Pansharpening任务代码复现与指标对比）
## Code Illustration
~~
## Methods
### Abbreviation Explanation

### Table
| Methods | Year | From | Author | 
|:-------:|:-------:|:-------:|:-------:|
| PNN | 2016 | [Remote Sensing](https://www.mdpi.com/2072-4292/8/7/594) | Masi et al. |
| PanNet | 2017 | [ICCV](https://arxiv.org/abs/1908.05900) | Yang et al. |
| DiCNN | 2019 | [JSTARS](https://ieeexplore.ieee.org/document/8667040) | He et al. |

## Datasets
### Abbreviation Explanation
| Abbreviation | Explanation |
|:-------:|:-------:|
|NMSB|Number of Multi-Spectral Bands|
|GSD|Ground Sample Distance|
|RRT|Reduced Resolution Testing|
|FRT|Full Resolution Testing Set|

### Table
| Name | NMSB| Type | GSD | Sets | Sample Number | Image Size | From |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|GaoFen-2|4|PAN<br>MS|0.8m<br>3.2m|Training<br>Validation<br>RRT<br>FRT|19809<br>2201<br>20<br>20|$64 \times 64 \times 1, 64 \times 64 \times 4$<br>$64 \times 64 \times 1, 64 \times 64 \times 4$<br>$256 \times 256 \times 1, 256 \times 256 \times 4$<br>$512 \times 512 \times 1, 512 \times 512 \times 4$|[1]|

### Reference
- [1] L.-j. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza, “Machine learning in pansharpening: A benchmark, from shallow to deep networks,” IEEE Geoscience and Remote Sensing Magazine, p. 279–315, Sep 2022.
- [2] demo

## Quality Assessment Indices
~~

