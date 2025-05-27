***Dyformer: A Dynamic Feature Adaptation Model for Long Sequence Time-Series Forecasting***

Abstract-Long Sequence Time-series Forecasting (LSTF) is of great significance in practical applications such as power consumption planning and meteorological forecasting. 
The core challenge is to efficiently model long-distance temporal dependencies. Although the Informer model has made significant progress in the LSTF task by improving the attention mechanism, there are still deficiencies in the dynamic adaptability of features and computational efficiency. 
In order to break through this bottleneck, this paper proposes an improved Informer model based on dynamic adaptive architecture——Dynamic-Informer (Dyformer). 
The core innovation of Dyformer is to propose a dynamic feature adaptation mechanism, including Dynamic Attention, Dynamic Convolution and Gating Mechanism. 
Dynamic attention enhances the dynamics of attention mechanism through input-sensitive weight allocation strategy, and can adaptively adjust the attention weight according to the characteristics and context information of input data, so as to pay more attention to important features and improve the modeling ability of complex time series data. 
The dynamic convolution layer dynamically generates convolution kernel parameters according to the input data, optimizes the local feature extraction process, so that the convolution operation can better adapt to different input modes, flexibly extract local features, and further enhance the model 's feature expression ability for time series data. 
The gated mechanism automatically filters noise information through a learnable threshold, retains key features, effectively removes irrelevant or redundant information, avoids noise interference in model prediction, and improves the prediction accuracy of the model. The synergy of these mechanisms significantly improves the modeling ability and prediction accuracy of the model for complex time series data.


## Requirements

- Python 3.6
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

The ETT dataset used in the paper can be downloaded in the repo [ETDataset](https://github.com/zhouhaoyi/ETDataset).
The required data files should be put into `data/ETT/` folder. A demo slice of the ETT data is illustrated in the following figure. Note that the input of each dataset is zero-mean normalized in this implementation.

<p align="center">
<img src="./img/data.png" height = "168" alt="" align=center />
<br><br>
<b>Figure 3.</b> An example of the ETT data.
</p>
