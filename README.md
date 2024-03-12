# Visual-Explanations-for-Spatio-temporal-Networks


Hi, here I am storing all the articles and works that are related to explainable artificial intelligence and video data, in particular, in the field of phase recognition of surgical operations :hugs:


## **XAI in Videos**

**Repositories**

:desktop_computer: Repositories CNN

1. [Visually explaining 3D-CNN predictions for video classification with an adaptive occlusion sensitivity analysis](https://github.com/uchiyama33/AOSA)
   Model R3D [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
   Datset UCF-101

2. [Grad-CAM-for-video-and-regression-task](https://github.com/UtopAIBuilder/Grad-CAM-for-video-and-regression-task)
   - First model uses 3D convolution over time and space
   - Second uses vanilla LSTM over 2D CNN model
   - Third model uses LSTM over 2D CNN model with attention

3. [Spatio-Temporal-Perturbations-for-Video-Attribution](https://github.com/shinkyo0513/Spatio-Temporal-Perturbations-for-Video-Attribution)
   
4. [Pytorch code for paper "TCAM: Temporal Class Activation Maps for Object Localization in Weakly-Labeled Unconstrained Videos"](https://github.com/sbelharbi/tcam-wsol-video?tab=readme-ov-file)


:desktop_computer: Repositories Transformers

1. Transformer-Explainability  [GitHub](https://github.com/hila-chefer/Transformer-Explainability)
   
Implementation of Chefer, Hila et al. “Transformer Interpretability Beyond Attention Visualization.” 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020): 782-791. [Article](https://arxiv.org/abs/2012.09838). ViT explainability for images

:desktop_computer: Repositories SSL

1. ...


:desktop_computer: Repositories LSTM

1. CS230-CNN-LSTM: Workflow Recognition from Surgical Videos using LSTM [GitHub](https://github.com/madhuhegde/C230_CNN_LSTM)

:star2: :desktop_computer: *Grad-CAM Repositories*

1. A Simple pytorch implementation of GradCAM and GradCAM++ (for Images)
   [GitHub](https://github.com/1Konny/gradcam_plus_plus-pytorch)
2. pytorch-grad-cam
   [GitHub](https://github.com/jacobgil/pytorch-grad-cam)
   https://jacobgil.github.io/pytorch-gradcam-book/HuggingFace.html
3. Grad_CAM_plus_plus
   [GitHub](https://github.com/adityac94/Grad_CAM_plus_plus)
4. Class activation maps for your PyTorch models (CAM, Grad-CAM, Grad-CAM++, Smooth Grad-CAM++, Score-CAM, SS-CAM, IS-CAM, XGrad-CAM, Layer-CAM) 
   [GitHub](https://github.com/frgfm/torch-cam)

:star2: :desktop_computer: *Other*

1. [A toolbox to iNNvestigate neural networks' predictions!](https://github.com/albermax/innvestigate?tab=readme-ov-file#usage-and-examples)


**:writing_hand: Articles** 
      
*2024 year*
      
   **1. Gaya-Morey, F. Xavier et al. “Local Agnostic Video Explanations: a Study on the Applicability of Removal-Based Explanations to Video.” (2024).**
      [Article](https://www.semanticscholar.org/paper/Local-Agnostic-Video-Explanations%3A-a-Study-on-the-Gaya-Morey-Buades-Rubio/ec8f1b25935904e8866d51d93e27ce0894e324a4)

The key results of this paper include the adaptation of six explanation techniques for video data, the evaluation and comparison of these methods using different models and datasets, and the finding that 3D RISE, 3D LIME, and 3D Kernel SHAP outperform other methods.

   **2. Kowal, Matthew et al. “Understanding Video Transformers via Universal Concept Discovery.” ArXiv abs/2401.10831 (2024): n. pag.**
      [Article](https://www.semanticscholar.org/paper/Understanding-Video-Transformers-via-Universal-Kowal-Dave/f717c3581620b198fc22b5f5bd946286f4ab1e0b)

      
*2023 year*
      
   **1. Gulshad, Sadaf et al. “Hierarchical Explanations for Video Action Recognition.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2023): 3703-3708.**
      [Article](https://www.semanticscholar.org/paper/Hierarchical-Explanations-for-Video-Action-Gulshad-Long/8aa729cff93d270bab251ece1db8044bcb7318c9)
      
The goal of this paper is to propose a model called HIPE (Hierarchical Prototype Explainer) for video action recognition that provides multi-level explanations by learning hierarchical prototypes.
      
The key results of this paper are that the proposed Hierarchical Prototype Explainer (HIPE) outperforms a non-hierarchical approach on the UCF-101 dataset and performs equally well on the ActivityNet dataset. HIPE provides multi-level explanations, allowing for a deeper understanding of the spatiotemporal parts that contribute to different levels of classification.

   **2. Jalal, Nour Aldeen et al. “Laparoscopic Video Analysis Using Temporal, Attention, and Multi-Feature Fusion Based-Approaches.” Sensors (Basel, Switzerland) 23 (2023): n. pag.**
      [Article](https://www.semanticscholar.org/paper/Surgical-phase-classification-and-operative-skill-Pradeep-Sinha/d82ec32fa2b932ff76175282d8c49ed51db23fe4?citedSort=relevance&citedPage=3)

*2022 year*
      
   **1. Hartley, Thomas et al. “SWAG-V: Explanations for Video using Superpixels Weighted by Average Gradients.” 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2022): 1576-1585.**
      [Article](https://www.semanticscholar.org/paper/SWAG-V%3A-Explanations-for-Video-using-Superpixels-by-Hartley-Sidorov/414bfba7768b9f44ea53be05647d9a7e913eb354#citing-papers)
      
   **2. Ji, Yi et al. “Spatial-temporal Concept based Explanation of 3D ConvNets.” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 15444-15453.**
      [Article](https://www.semanticscholar.org/paper/Spatial-temporal-Concept-based-Explanation-of-3D-Ji-Wang/4f9e122682617710546ce5beaba02778c098bbc3)
      
The goal of this paper is to propose a 3D ACE (Automatic Concept-based Explanation) framework for interpreting 3D ConvNets in video recognition. The framework aims to discover spatial-temporal concepts of different importance levels and explore their influence on a target task, such as action classification.
      
The key results of this paper include the development of a 3D ACE framework for interpreting 3D ConvNets, the discovery of spatial-temporal concepts of different importance levels, and the exploration of the influence of these concepts on a target task, such as action classification.

      [GitHub](https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions)


   **3. Mänttäri, Joonatan et al. “Interpreting video features: a comparison of 3D convolutional networks and convolutional LSTM networks.” Asian Conference on Computer Vision (2020).**
      [Article](https://arxiv.org/abs/2002.00367

*2021 year*
             
   **1.Li, Zhenqiang et al. “Towards Visually Explaining Video Understanding Networks with Perturbation.” 2021 IEEE Winter Conference on Applications of Computer Vision (WACV) (2020): 1119-1128.**
      [Article](https://www.semanticscholar.org/paper/Towards-Visually-Explaining-Video-Understanding-Li-Wang/b9338b7de4b849cb094aa4cbd5b85f9935a4ae00)


*2020 year*

   **1. Sümer, Ömer et al. “Attention Flow: End-to-End Joint Attention Estimation.” 2020 IEEE Winter Conference on Applications of Computer Vision (WACV) (2020): 3316-3325.**
      [Article](https://www.researchgate.net/publication/338569697_Attention_Flow_End-to-End_Joint_Attention_Estimation)

      The goal of this paper is to address the problem of understanding joint attention in third-person social scene videos and propose a method, Attention Flow, that learns joint attention in an end-to-end fashion.

      The key results of this paper are that the proposed Attention Flow method successfully learns joint attention in third-person social scene videos, achieving an accuracy of 78.1% in detecting joint attention and localizing co-attention bounding boxes with an L2 distance of 62.84.

      The paper uses saliency-augmented attention maps and two novel convolutional attention mechanisms to determine relevant features and improve joint attention localization.

   **2. Interpreting Video Features: a Comparison of 3D Convolutional Networks and Convolutional LSTM Networks Joonatan Mänttäri, Sofia Broomé, John Folkesson, Hedvig Kjellström. Joint first authorship ACCV 2020, 15th Asian Conference on Computer Vision, to appear.**

 [Article](https://arxiv.org/pdf/2002.00367.pdf)
 
 [GitHub](https://interpreting-video-features.github.io/)
      
      
*2019 year*
      
   **1. Stergiou, Alexandros et al. “Saliency Tubes: Visual Explanations for Spatio-Temporal Convolutions.” 2019 IEEE International Conference on Image Processing (ICIP) (2019): 1830-1834.**
      [Article](https://www.semanticscholar.org/paper/Saliency-Tubes%3A-Visual-Explanations-for-Stergiou-Kapidis/7a307c21fdd9a3edff092fe0485399714e53fd7a#citing-papers)
      [GitHub](https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions)
   
      
The methods used in the paper are Saliency Tubes, a generalized attention mechanism for explaining CNN decisions, and 3D Convolutional Neural Networks (CNNs) for video classification and recognition.
      
The key results of this paper are the proposal of Saliency Tubes as a method to visualize the activation maps of 3D CNNs, the demonstration of Saliency Tubes on existing video recognition models for action classification and egocentric action recognition, and the improvement of interpretability of 3D CNNs by revealing the spatio-temporal regions that are most informative for predicting action classes.

*2016 year*

   **1. Selvaraju, Ramprasaath R. et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” International Journal of Computer Vision 128 (2016): 336 - 359.**
      [Article](https://z-p3-scontent-muc2-1.xx.fbcdn.net/v/t39.8562-6/78951195_2176196352683826_3137830233747488768_n.pdf?_nc_cat=110&ccb=1-7&_nc_sid=b8d81d&_nc_ohc=0RkVz8D3XhwAX-0Fe7i&_nc_ht=z-p3-scontent-muc2-1.xx&oh=00_AfA8BvrkIG0iZrLqhZSrSvTAHfzz6JmKw9FCc7BYxkkaIw&oe=65D00F9D)

   
## **Video**

1. [OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark](https://github.com/open-mmlab/mmaction2)

2. [3D-Machine-Learning](https://github.com/timzhang642/3D-Machine-Learning)

## **Survey Papers for XAI and/or medicine**

1. [Official repository of the paper "Explainable Deep Learning Methods in Medical Image Classification: A Survey", ACM Computing Surveys (CSUR) 2023](https://github.com/CristianoPatricio/Explainable-Deep-Learning-Methods-in-Medical-Image-Classification-A-Survey)

2. [Surgical-Phase-Recognition](https://github.com/maxboels/Surgical-Phase-Recognition)

///////////////////////////////////////////////////////////////////////////
## **Surgical phase recognition on Cholec80**

1. Czempiel, Tobias et al. “TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks.” ArXiv abs/2003.10751 (2020): n. pag. [Article](https://arxiv.org/pdf/2003.10751.pdf) [GitHub](https://github.com/tobiascz/TeCNO)


Abdulbaki Alshirbaji T, Jalal NA, Docherty PD, Neumuth T, Möller K. Robustness of Convolutional Neural Networks for Surgical Tool Classification in Laparoscopic Videos from Multiple Sources and of Multiple Types: A Systematic Evaluation. Electronics. 2022; 11(18):2849. https://doi.org/10.3390/electronics11182849 

[Article](https://www.mdpi.com/2079-9292/11/18/2849)

2. Ramesh, Sanat et al. “Dissecting Self-Supervised Learning Methods for Surgical Computer Vision.” Medical image analysis 88 (2022): 102844 .**
      [Article](https://www.semanticscholar.org/reader/c118fd58ee236828baab90e4aba660df88ea4174)
      [GitHub](https://github.com/CAMMA-public/SelfSupSurg)


## **Good Links**
https://opensource.salesforce.com/OmniXAI/latest/tutorials/vision/gradcam_torch.html

https://github.com/interpretml/interpret

https://github.com/C-Ritam98/GradCAM-on-Video

http://www.diva-portal.se/smash/get/diva2:1316346/FULLTEXT01.pdf

https://www.kaggle.com/code/asraf047/lstm-grad-cam

https://www.sciencedirect.com/science/article/pii/S1746809421003980#fig0010
