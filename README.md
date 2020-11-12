# ToGCN-V2X
This is the official implementation of the following [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9247476):

> *Topological Graph Convolutional Network-Based Urban Trafﬁc Flow and Density Prediction*
>
> Han Qiu, Qinkai Zheng, Mounira Msahli, Meikang Qiu, Gerard Memmi, Jialiang Lu
>
> *Abstract*: With the development of modern Intelligent Transportation System (ITS), reliable and efﬁcient transportation information sharing becomes more and more important. Although there are promising wireless communication schemes such as Vehicle-to-Everything (V2X) communication standards, information sharing in ITS still faces challenges such as the V2X communication overload when a large number of vehicles suddenly appeared in one area. This ﬂash crowd situation is mainly due to the uncertainty of trafﬁc especially in the urban areas during trafﬁc rush hours and will signiﬁcantly increase the V2X communication latency. In order to solve such ﬂash crowd issues, we propose a novel system that can accurately predict the trafﬁc ﬂow and density in the urban area that can be used to avoid the V2X communication ﬂash crowd situation. By combining the existing grid-based and graph-based trafﬁc ﬂow prediction methods, we use a Topological Graph Convolutional Network (ToGCN) followed with a Sequence-tosequence (Seq2Seq) framework to predict future trafﬁc ﬂow and density with temporal correlations. The experimentation on a real-world taxi trajectory trafﬁc data set is performed and the evaluation results prove the effectiveness of our method.

If you have any question, please raise an issue or contact ```qinkai.zheng1028@gmail.com```. 

## Requirements

* numpy==1.19.1
* torch==1.3.0
* torchsummary==1.5.1
* tensorboardX==2.1

