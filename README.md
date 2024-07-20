# The implementation of remote sensing images pan sharpening methods
Replication of Remote Sensing Image Pansharpening Task Code and Comparison of Evaluation Metrics.（遥感图像Pansharpening任务代码复现与指标对比）
## Code Illustration
~~

## Abbreviation Explanation
| Abbreviation | Explanation |
|:-------:|:-------:|
|RSE|Remote Sensing of Environment|
|WHISPERS|Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing|
|PERS|Photogrammetric Engineering and Remote Sensing|
|IJRS|International Journal of Remote Sensing|
|IGARSS|IEEE International Geoscience and Remote Sensing Symposium|
|ICCV|IEEE International Conference on Computer Vision|
|JSTARS|IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing|
|LGRS|IEEE Geoscience and Remote Sensing Letters|
|TGRS|IEEE Transactions on Geoscience and Remote Sensing|
|IF| Information Fusion |
|NMSB|Number of Multi-Spectral Bands|
|GSD|Ground Sample Distance|
|RRT|Reduced Resolution Testing|
|FRT|Full Resolution Testing|
## Methods--Tradition
| Methods | Year | Author | From | 
|:-------:|:-------:|:-------:|:-------:|
|Brovey| 1987 |Gillespie et al. | [RSE](https://doi.org/10.1016/0034-4257(87)90088-5) [1]|
|PCA|1989| Chavez et al.| PERS [8]
|IHS |1990 |Carper et al.| [PERS](https://arxiv.org/pdf/2203.04286) [5]|
|SFIM|2000| Liu et al.| [IJRS](https://doi.org/10.1080/014311600750037499) [9]|
| GS | 2000 | Laben et al. | U.S. Patent [3]|
|Wavelet|2001| King et al.| [IGARSS](10.1109/IGARSS.2001.976657) [10]|
|MTF_GLP|2002| Aiazzi et al.| [TGRS](https://doi.org/10.1109/TGRS.2002.803623) 7|
|MTF_GLP_HPM|2006| Aiazzi  et al.|[PERS](https://doi.org/10.14358/PERS.72.5.591) 6|
| GSA  | 2007 |Aiazzi et al.|[TGRS](https://doi.org/10.1109/TGRS.2007.901007) 4|
| CNMF | 2012 | Yokoya et al. | [TGRS](https://doi.org/10.1109/TGRS.2011.2161320) 11|
|GFPCA| 2015 |Liao et al.|[WHISPERS](https://doi.org/10.1109/WHISPERS.2015.8075405) 2| 


## Methods--Deep Learning
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
|WorldView-2|8|PAN<br>MS|0.46m<br>1.84m|Training<br>Validation<br>RRT<br>FRT|-<br>-<br>20<br>20|$-, -$<br>$-, -$<br>$256 \times 256 \times 1, 64 \times 64 \times 8$<br>$512 \times 512 \times 1, 128 \times 128 \times 8$|[1]|
|Worldview-3|8|PAN<br>MS|0.31m<br>1.24m|Training<br>Validation<br>RRT<br>FRT|9714<br>1080<br>20<br>20|$64 \times 64 \times 1, 16 \times 16 \times 8$<br>$64 \times 64 \times 1, 16 \times 16 \times 8$<br>$256 \times 256 \times 1, 64 \times 64 \times 8$<br>$512 \times 512 \times 1, 128 \times 128 \times 8$|[1]|
## Quality Assessment Indices
~~
## Reference
1. A. R. Gillespie, A. B. Kahle, and R. E. Walker, “Color enhancement of highly correlated images-II. Channel ratio and “Chromaticity” Transform techniques,” Remote Sensing of Environment, vol. 22, no. 3, pp. 343–365, August 1987.
2. W. Liao et al., "Two-stage fusion of thermal hyperspectral and visible RGB image by PCA and guided filter," 2015 7th Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing (WHISPERS), Tokyo, 2015, pp. 1-4.
3. C. A. Laben and B. V. Brower, “Process for enhancing the spatial resolution of multispectral imagery using pan-sharpening,” Eastman Kodak Company, Tech. Rep. US Patent # 6,011,875, 2000.
4. B. Aiazzi, S. Baronti, and M. Selva, “Improving component substitution Pansharpening through multivariate regression of MS+Pan data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3230–3239, October 2007.
5. W. Carper, T. Lillesand, and R. Kiefer, “The use of Intensity-Hue-Saturation transformations for merging SPOT panchromatic and multispectral image data,” Photogrammetric Engineering and Remote Sensing, vol. 56, no. 4, pp. 459–467, April 1990.
6. B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, “MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,” Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591–596, May 2006.
7. B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli, “Context-driven fusion of high spatial and spectral resolution images based on oversampled multiresolution analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 40, no. 10, pp. 2300–2312, October 2002.
8. P. S. Chavez Jr. and A. W. Kwarteng, “Extracting spectral contrast in Landsat Thematic Mapper image data using selective principal component analysis,” Photogrammetric Engineering and Remote Sensing, vol. 55, no. 3, pp. 339–348, March 1989.
9. J. Liu, “Smoothing filter based intensity modulation: a spectral preserve image fusion technique for improving spatial details,” International Journal of Remote Sensing, vol. 21, no. 18, pp. 3461–3472, December 2000.
10. King R L, Wang J. A wavelet based algorithm for pan sharpening Landsat 7 imagery [C]//IGARSS 2001. Scanning the Present and Resolving the Future. Proceedings. IEEE 2001 International Geoscience and Remote Sensing Symposium (Cat. No. 01CH37217). IEEE, 2001, 2: 849-851.
11. N. Yokoya, T. Yairi, and A. Iwasaki, "Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion," IEEE Trans. Geosci. Remote Sens., vol. 50, no. 2, pp. 528-537, 2012.
12. L.-j. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza, “Machine learning in pansharpening: A benchmark, from shallow to deep networks,” IEEE Geoscience and Remote Sensing Magazine, p. 279–315, Sep 2022.
13. demo
