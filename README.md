# 1. 프로젝트 개요

**“마스크 착용 상태 분류”**

카메라로 촬영한 사람 얼굴 이미지의 **마스크 착용 여부, 성별, 연령**을 판단하는 Task

일정: 12월 11일(월) 10:00 ~ 12월 21일(목) 19:00


**“Dataset”**

- 전체 **4500명, 7개 사진**으로 구성
    - id, race, gender, age
    - mask 5개, incorrect 1개, normal 1개
- Dataset 구분
  
    <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/dataset_%EA%B5%AC%EB%B6%84.png" width="700" height="20"/>

    

**“Class description”**

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/class_description.png" width="500" height="500"/>

**“Baseline code”**

📂dataset.py : model train, validation을 위한 dataset 정의

📂inference.py : model prediction

📂loss.py : loss functions 정의 (cross-entropy, focal, label_smoothing, f1 score)

📂model.py : model 정의

📂train.py : model training

**“평가 방법”**

- **Submission**
    
    test dataset을 사용해 성능 평가, csv 파일 제출
    
    submission.csv → 각 이미지들에 대해 mask_class(ans)를 예측한 값이 ,(콤마)로 구분
    
- **Evaluation**
    
    제출한 csv 파일에 대해 **F1 Score**를 통해 평가
    
    F1 is calculated as follows:
    
    $$
    F_1 = 2\ *\ \frac{precision\ *\ recall}{precision\ +\ recall}
    $$
    
    where:
    
    $$
    precision=\frac{TP}{TP\ +\ FP}
    \qquad
    recall=\frac{TP}{TP\ +\ FN}
    $$
    

# 2. 프로젝트 팀 구성 및 역할

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EA%B9%80%EC%84%B8%EC%A7%84_%ED%94%84%EB%A1%9C%ED%95%84.png" width="200" height="200"/>

**김세진_T6019**

데이터 전처리, EDA, 아이디어 제공

- - -

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EA%B9%80%ED%83%9C%EC%96%91_%ED%94%84%EB%A1%9C%ED%95%84.jpg" width="200" height="200"/>

**김태양_T6044**

모델 테스트, 튜닝, 오류 검출, 수정

- - -

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%B0%95%EC%A7%84%EC%98%81_%ED%94%84%EB%A1%9C%ED%95%84.jpg" width="200" height="200"/>

**박진영_T6063**

아이디어 제공, 모델 오류 및 검출

- - -

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EC%9D%B4%EC%84%A0%EC%9A%B0_%ED%94%84%EB%A1%9C%ED%95%84.jpeg" width="200" height="200"/>

   **이선우_T6125**

   모델 선정, 테스트, 튜닝

- - -

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EC%A7%84%EB%AF%BC%EC%A3%BC_%ED%94%84%EB%A1%9C%ED%95%84.png" width="200" height="200"/>


**진민주_T6171**

데이터 전처리, EDA, 모델 선정

# 3. 프로젝트 수행 절차 및 방법

<img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/timeline.png" width="600" height="400"/>

- - -

🔧 환경 구성

AI Stage에서 제공되는 서버를 사용했습니다.

- GPU: Tesla V100
- OS: Ubuntu 18.04.5LTS
- CPU: Intel Xeon
- Python : 3.8.5

그리고 github와 wandb 연동, 필요한 라이브러리 설치했습니다.

- - -

🔍 EDA를 통한 데이터 분석 후 데이터 전처리

- 데이터 분석
    
    EDA를 통한 데이터를 분석했을 때의 특징은 다음과 같았습니다.
    
    - image
        - 대부분의 이미지는 인물이 정중앙에 있다.
            
            <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D_image.png" width="500" height="300"/>
            
    - labelling
        - 잘못된 라벨링이 있다.
        
            <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D_labelling.png" width="500" height="300"/>
        
    
    - mask
        - 마스크 유무의 RGB가 다르다.
        
            <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D_mask.png" width="500" height="300"/>
        
        - Mask5번에 존재하는 스카프형 마스크의 데이터 양이 매우 적다.
        - 일반적인 마스크의 형태가 아닌 경우도 있다.
    - age
        
        <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D_age.png" width="500" height="300"/>
        
        - 나이의 불균형이 심하다.
        - 60대가 적다.
    - gender
        
        <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D_gender.png" width="500" height="300"/>
        
        - 성별 불균형이 존재한다.
    
    Mask와 Gender의 불균형은 Age의 불균형 보다 학습에 크게 영향을 미치지 않을 것이라 판단하였습니다.
    
    해당 문제를 해결하기 위해 57, 58, 59세의 데이터는 삭제하고, 60세 데이터는 Custom Cutmix와 Custom Mixup으로 데이터를 증강시켰습니다.
    
    또한 Mask5번에 존재하는 긴 마스크의 데이터 양이 매우 적어서 같은 성별의 얼굴부분과 긴 마스크부분을 Custom Cutmix로 데이터 증강시켰습니다.
    

- - -

- 데이터 전처리
    - Augmentation
        
        
        | Augmentation | 사용 이유 | 성능 |
        | --- | --- | --- |
        | Resize | 모델마다 특정한 이미지 크기를 요구해서 적용 | 특정한 크기를 요구하지 않으면 사이즈가 클 때 성능 상승 |
        | CenterCrop | 얼굴이 중앙에 있고, 배경의 영향을 받지 않게 하기 위해서 적용 | 성능 상승 |
        | RandomHorizontalFlip  | 이미지의 다양성을 높이기 일부 이미지들에 좌우 반전을 적용 | 성능 상승 |
        | GrayScale | 이미지의 색상에 대한 일반화 성능을 위해 적용 | 성능 상승 |
        | rembg | 배경의 영향을 받지 않게 하기 위해서 적용 | 성능 하락 |
        | RandomSharpness | 나이를 특정하는게 가장 어려워서 주름을 잘 보이게 하기 위해서 적용 | 성능 살짝 상승했으나 실제 제출시 하락 |
        | ColorJitter | 이미지의 색상에 대한 일반화 성능을 위해 적용 | GrayScale이 더 좋음 |
        | CutMix/ Mixup | 이미지 갯수 늘리기 위해 적용 | 성능 하락 |
        
        → 이외에도 다양하게 적용해보았습니다.
        

- - -

🔨 Baseline Code 변경

- 기존의 Baseline Code에서 multi-label 코드로 변경
    - Mask, Age, Gender 총 3가지 라벨을 합쳐서 최종 라벨로 추론하는게 좋다고 생각하고 변경했습니다.
        - **같은 모델, 같은 Loss로 학습**
        - 같은 모델, 다른 Loss로 학습
        - 다른 모델, 같은 Loss로 학습
        - 다른 모델, 다른 Loss로 학습
- 앙상블 코드 추가
    - 하나의 모델로 추론하는 것보단 여러 모델의 결과를 합치는게 좋다고 생각해서 추가했습니다.
        - **Soft Voting**
        - Hard Voting

- - -

📊 다양한 모델 테스트

- ResNet18, ResNet50, DensNet121, EfficientNetV2-small, **Swin Transformer V2-tiny**, ConvNext-tiny, MobileNetV3-large, VIT - small, Swin Transformer V2 - large, Eva02 - large, **ConvNext - XXlarge**, VIT - large, **MetaFormer**, BeitV2 - large
- 모든 파라미터를 얼리고 사용하는 것보다 끝 부분의 파라미터를 얼리지 않고 사용하는 편이 성능이 더 좋게 나왔습니다.

- - -

👫 개인 실험 및 모델 공유

- 자신이 사용해본 모델, 추가해본 Augmentation, 하이퍼 파라미터 등을 공유하는 시간을 가졌습니다.

- - -

⚙️ HyperParameter Tuning

Grid Search를 통해서 제일 좋은 결과가 나온 것을 사용했습니다.

- Epoch: **3**, **5**, 6, 10
- Batch Size: 16, 32, **64, 128**
- Valid Batch Size: **1000** (메모리가 부족할 때 반마다 낮춰서 사용)
- Learning Rate: 1e-3, **1e-4**, 1e-5, 1e-6
- Criterion: CrossEntropy, **Focal loss**, F1_loss, Label_smoothing

- - -

# 4. 프로젝트 수행 결과

- wandb 그래프
    - MultilabelDataset 비교, Re-labeled dataset, baseAugmentation, Resize 224, center crop
        
      <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/wandb_%EA%B2%B0%EA%B3%BC1.png" width="500" height="300"/>
        
    
    - Augmentation 비교, BaseAugmentation vs centercrop(320, 256) + RandomHorizonalFlip(0.5) + Resize(128, 96) + ColorJitter + AddGausianNoise
        
      <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/wandb_%EA%B2%B0%EA%B3%BC2.png" width="500" height="300"/>
        
    
    - Dataset loader 비교, MaskBaseDataset, MaskSplitByProfileDataset, MultiLabelDataset
        
      <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/wandb_%EA%B2%B0%EA%B3%BC3.png" width="500" height="300"/>
        
    
    - f1 score가 높았던 모델들 비교, Multi model, ConvNeXT tiny, ConvNeXT Large, Swin v2 Large, Eva Large
        
      <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/wandb_%EA%B2%B0%EA%B3%BC4.png" width="500" height="300"/>
        
    
- 팀 코드 설명
    - 사용 모델
        - Multi-model confusion with multi-label(ConvNext tiny, ResNeXT50 tiny, Swin s3 tiny)
        - Ensemble with large models(ConvNext Large, Swin v2 Large, Eva Large)
    - 데이터 처리 및 로드 방법
        - Multi-label loader: class 분리 mask 3, gender 2, age 3, using fast prefetcher loader
        - Split by profile loader: MaskSplitByProfileDataset
    - loss, optimizer, scheduler, inference 방식
        - Loss: Focal loss
        - Optimizer: AdamW와 Adam 성능 비교 후 AdamW 선정, Lion 사용
        - Scheduler: StepLR, CosineAnnealingLR
        - Inference 방식: 단일 inference, ensemble (soft voting)

- 최종 수행 결과 (리더보드)
    - 최종 선정 모델 성능
        - ConvNext tiny, ResNeXT50, Swin s3 tiny 모델 3개를 각각 mask, gender, age를 예측하도록 구성하여 최종 class를 예측하는데 사용
        - ConvNext Large, Swin v2 Large, Eva Large 모델을 각자 최종 class를 예측하도록 학습하고, inference 시 각 결과를 앙상블하여 최종적으로 예측하는데 사용
    - 선정 방식 (f1 score 기준, tiny model, large model)
        - 대회 진행 중 제출한 모델들의 f1 score를 기준으로 최종 모델을 선정함
        - 또한, dataset가 제한적이라고 생각하여 tiny 모델을 앙상블한 결과와 large 모델을 앙상블한 결과 2가지를 최종 선정
    - 리더보드 캡쳐(대회 진행, 최종)
    <img src="https://github.com/boostcampaitech6/level1-imageclassification-cv-03/blob/main/readme_image/%EB%A6%AC%EB%8D%94%EB%B3%B4%EB%93%9C.png" width="700" height="100"/>
   
    

# 5. 자체 평가 의견

- 회고
    - 잘한 점
        - wandb를 적극적으로 활용해서 성능비교를 했다.
        - 다양한 라이브러리(albumentations, rembg, timm, sklearn 등)를 사용해봤다.
        - 아이디어 공유가 활발했다.
        - 정말 다양한 시도를 해보고, 제출을 활발하게 했다.
    - 아쉬운 점
        - 협업 툴(Github)을 제대로 활용하지 못했다.
        - 계획적으로 실험하지 못했다.
        - 모델링에만 너무 집중해서 데이터 전처리가 성공적이지 못했다.
        - 다양한 실험을 했지만, 중간 과정을 정리하지 않았던게 아쉬웠다.
    - 깨달은 점
        - 데이터 전처리의 중요성을 확실하게 깨달았다.
        - 협업 툴을 통한 버전관리의 필요성을 느꼈다.
        - 가설 세우기, 계획적인 실험, 검증 과정들의 중요성을 느꼈다.
        - 분류 문제라고 분류에만 집착한게 아쉬웠다.

# **개인 평가**

- 김세진_T6019
    
    나는 내 학습목표를 달성하기 위해 무엇을 어떻게 했는가? 
    
    - 데이터를 분석하고, 부족한 데이터를 전처리했습니다.
        - Custom Cutmix와 Mixup을 구현해서 60세 데이터를 증강시켰습니다.
        - 60세 데이터와 구별하기 어려운 57,58,59세 데이터들은 Down Sampling했습니다.
        - 남성이 꽃무늬 스카프를 착용하고 있는 데이터가 매우 부족하여 Cutmix로 증강시켰습니다.
        - 중간 나이대에서 30~40세 데이터가 부족한 것을 파악하고, Cutmix로 증강시켰습니다.
    - 3가지의 라벨(Mask, Gender, Age)로 나누어서 학습시켰습니다.
    - 하이퍼 파라미터에서 많은 것을 바꿔가며 직접 학습시켜봤습니다.
    
    마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
    
    - 데이터셋의 문제점(데이터 불균형)은 파악했지만 해결하지 못했습니다.
    - 문제를 해결할 방안에 대한 가설을 세우고 시도해봤지만 해결하지 못했습니다.
    - 협업 툴을 효과적으로 사용하지 못했습니다.
    - Base코드에만 의존해서 다양한 시도를 더 해보지 못한게 아쉽습니다.
    
    한계/교훈을 바탕으로 다음 프로젝트에서 시도해보고 싶은 점은 무엇인가?
    
    - 데이터 전처리를 확실하게 하고싶습니다.
    - 문제를 해결할 때는 하나씩 차근차근 해봐야겠습니다.
    - Base코드에만 의존하지 않고, 직접 처음부터 구현도 해보고싶습니다.

- - -

- 김태양_T6044
    
    나는 내 학습목표를 달성하기 위해 무엇을 어떻게 했는가?
    
    - baseline 코드를 기반으로 모듈화 작업 충실하기
        - baseline 코드를 이해하기 위해 가장 기본적인 VGG16 모델과 Resnet18 모델을 통해 코드를 실행해 보면서 동작하는 프로세스를 눈으로 확인해 보았습니다.
        - [dataset.py](http://dataset.py) 파일에서 사용되는 Augmentation과 dataset class들을 사용해 보면서 새로운 Multi label dataset class를 구성해 보고 모델에 적용하였습니다.
        - 마스터 클래스 중 모델의 학습 속도 개선 및 성능 향상을 위해 알려주신 Mixed precision training 기법과 set gradient None, Asynchronous data transfer to device, Fast prefetcher 등을 적용해 보며 코드의 모듈화를 충실하게 진행하였습니다.
        - 마지막으로 이런 모듈화를 하나씩 추가할 때 마다 github에 해당 작업에 관한 commit을 남기고 정리하여 추후에도 다시 확인할 수 있도록 하였습니다.
    - 다양한 모델을 사용해 보고 성능 향상하기
        - 가장 기본적인 VGG16, Resnet18, Resnet50부터 최신 논문에서 성능이 좋은 Vision Transformer 기반으로 하는 ViT, Swin Transformer, Eva 등을 사용하여 모델 성능을 비교하였습니다.
        - 데이터가 부족하고 inbalance한 문제가 있다는 것을 확인할 수 있었고, Parameter가 큰 모델은 조금의 epoch만 학습해도 overfitting이 발생하는 것을 확인하였습니다.
        - 이러한 overfitting을 해결하기 위해 다양한 Augmentation 기법과 Optimizer를 사용해 보았고, 그 중 Center Crop과 Random Horizonal Flip Augmentation이 성능이 잘 나왔습니다.
        - 이러한 성능 비교는 Wandb를 통해 실험 관리를 진행 하였고, 팀원들과 공유하기 위해 logging name을 정리하여 추후에도 확인하기 쉽도록 하였습니다.
        - 또한, 모델의 Class는 mask, gender, age 3가지의 sub labels로 연관되어 있기 때문에 하나의 모델로 전체 class를 예측하는 것 보다 각 label을 예측하는 것이 성능이 좋을 것이라는 가설을 세우고 모델을 실험하였습니다.
        - 이러한 가설은 동일한 모델(Resnet18)을 사용하여 검증 하였을 때 multi label을 따로 예측하는 것이 성능이 높아지는 것을 확인함으로써 증명하였습니다.
        - 이후 ConvNeXT tiny, Swin s3 tiny, ResNeXT50 등의 모델을 앙상블 기법을 통해 추론하는데 사용하며 성능을 높이는 방법도 적용하였습니다.
        - 최종적으로 나온 모델은 ConvNeXT tiny 모델을 통해 mask label을, ResNeXT50 모델을 통해 gender label을, Swin s3 tiny 모델을 통해 age label을 각각 예측하고, 이를 토대로 최종 class를 예측하는 모델을 구현하였고, 이 모델이 팀 내의 public f1 score에서 제일 좋은 성능을 보였습니다.
    - 마주한 한계와 아쉬운 점
        - 처음에 EDA를 진행하여 전체적인 데이터의 불균형, Age label에서의 심한 데이터 불균형 문제 등을 파악하였습니다.
        - 그러나, 모델링에 치중하여 많은 시간을 사용하였습니다. 이는 이후 모델 성능을 더 높이기 위한 데이터 불균형 문제를 해결하는 데이터 전처리 과정을 추가적으로 진행하였다면 더 좋은 결과가 있었을 것이라고 후회하게 되었습니다.
        - 또한, 데이터 전처리 과정을 최소한만 진행하여 최적의 모델을 찾아가는 것 보다 데이터 전처리에 시간을 투자하여 잘 정제된 데이터 세트를 구성하는 것이 좋은 결과로 이어졌을 것이라는 생각이 들었습니다.
        - 팀원들과 잘 상의해서 계획적으로 실험을 진행 하였다면 더 좋은 성적이 나왔을 것 같다는 아쉬움이 남았고, 데이터 전처리 과정을 조금 소홀히 하였다는 생각에 다음 프로젝트에서는 이런 부분도 꼼꼼하게 진행하고자 합니다.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 시도해 보고자 하는 점
        - 우선 프로젝트를 바로 진행하는 것이 아닌 체계적인 프로젝트 관리를 위해 사용할 tools(github, wandb, gpu server 등)를 팀원들과 상의해서 명확하게 사용 방법을 공유하고 실천하는 것을 시도할 것입니다.
        - 또한, 데이터 전처리를 위한 EDA를 꼼꼼하게 진행하여 더 좋은 결과를 얻을 수 있도록 프로젝트에서 해결하고자 하는 목표를 뚜렷하게 할 계획입니다.

- - -

- 박진영_T6063
    
    **************학습목표**************
    
    딥러닝 프로젝트 전체 과정이 어떻게 이루어지는지 배우자!
    
    베이스라인 코드는 어떻게 이루어져있고, 앞으로 내가 어떻게 설계해야 하는지, 새로운 모델이나 기술(앙상블, 멀티 라벨 등)을 코드에 적용하기 위해선 어떤 걸 바꿔야 하는지 등 기초적인 부분을 다지는 것을 목표로 삼았다.
    
    **모델 개선 방법**
    
    미션과 강의에 나온 내용들을 많이 참고하고, 그 내용을 바탕으로 하나씩 고쳐나갔다.
    
    - 데이터 전처리 및 증강
        
        rembg, cutmix를 적용하며 배경을 지운 데이터셋, 나이 데이터가 추가된 데이터셋을 사용하여 모델 학습을 진행했다.
        
        어떤 조합으로 augmentation을 적용해야 효과 있는지 파악하기 위해, Gaussian Noise, Color Jitter, Center Crop, Resize를 사용했을 때와 사용하지 않았을 때의 차이를 비교해보았다. 결과적으로 noise와 color jitter를 사용하지 않고, center crop을 크게 했을 때 효과가 좋았다. 그리고 당연하게도 resize를 pretrained된 모델 input 사이즈에 맞게 적용하며 모델을 학습시킬 때 성능이 높게 나왔다.
        
    - pretrained model
        
        pretrained된 여러 모델을 비교해가며 어떤 모델을 사용했을 때 성능이 가장 좋은지 판단하였다. 그 결과, convnext → swin → vit → resnet50 순으로 성능이 좋았다.
        
    - multi-label
        
        과적합 문제 및 모델 예측 성능을 높이기 위해 18개의 클래스를 한번에 맞추는 것이 아니라 3개의 과제로 나누어 문제를 해결하는 multi label task로 바꾸어 진행하였다. (train 과정에서 model input을 하나가 아닌 mask, gender, age 3개를 받도록 수정하였다.)
        
    - ensemble
        
        여러 모델에서 나온 결과를 비교하여 모델 중 확률이 제일 높은 라벨을 선택하는 앙상블 기법을 사용하여 성능을 높였다. 앙상블 방법은 멀티 라벨을 적용한 모델 3개의 결과를 inference에서 합쳐 나온 결과를 활용하는 방법을 사용했다. (fold 5개 결과를 합치는 방법을 응용) 그 결과 사용하기 전보다 성능이 조금 더 높아진 것을 확인할 수 있었다.
        
    
    **마주한 한계**
    
    팀원들과 모델 개선 방향에 대한 이야기를 나눌 때, 인사이트를 제시하지 못한 점이 아쉬웠다. 새로 생각한 아이디어나 개선 방향도 과연 이게 도움이 될까? 라는 생각과 의심에 팀원들에게 선뜻 제안하지 못했다.
    
    여러 다양한 기법들을 적용해볼까 하다가 시간이 부족하거나 실력이 못 미친다는 생각에 시도하지 않고 넘어간 방법이 많다는 점이 많이 아쉽다. 일단 한 번 해보고 잘 안 되는 부분은 팀원들에게 도움을 받던 인터넷에 찾아보던 적극적으로 나서야 했는데 그러지 못했다는 게 아쉬웠다.
    
    기초 지식이 많이 부족하다는 것을 깨달았다. 저저번주에 공부했던 cv 기초 내용에서 공부했던 내용들이 당시에 과제에 포함되지 않은 내용이라, 상대적으로 덜 중요해보여서 집중하지 않고 넘어갔던 부분들이 실전에서 생각보다 많이 쓰인다는 점을 깨달았고 결코 덜 중요한 내용은 없다는 것을 알게 되었다.
    
    **개선 방향**
    
    먼저, 적극적으로 임하고 모르는 거나, 생각나는 아이디어 등 팀원들과 공유하고 싶은 내용은 바로바로 공유하고 어떤 모델을 변경하거나 새로운 기법을 적용하거나 하이퍼 파라미터를 수정하는 등의 여러 변경 사항을 그냥 이게 좋으니까, 그냥 다들 하니까가 아니라 왜 하는지 어떤 근거와 이유로 이 선택을 했는지 분명하게 밝히고 정의해야 한다는 것을 꺠달았다.
    
    어떤 부분을 놓쳤는지, 어떤 부분을 개선해야 하는지, 지금껏 어떤 부분을 변경했고 왜 그랬는지 놓치지 않는 꼼꼼함과 사용한 이유를 명확히 밝힐 수 있는 확실함을 챙기는 것을 다음 프로젝트의 목표로 삼아야 겠다.
    

- - -

- 이선우_T6125
    - 나는 내 학습목표를 달성하기 위해 무엇을 어떻게 했는가?
        - 현존하는 여러 모델들에 대한 논문을 읽고 직접 사용해보며 여러 지식을 쌓을 수 있었다.
        - rembg나 autoagment같은 여러 툴을 사용해보았다.
        - 일반화를 시키기 위한 여러 방식을 적용해보았다.
    - 마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
        - 데이터 전처리에 대한 이해가 부족했던 것 같다.
        - 직접 데이터를 확인하는 시간을 좀 더 가져야했을 것 같다.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 시도해보고 싶은 점은 무엇인가?
        - Stratified K-fold나 Random weighted sampler와 같은 전처리에도 공을 들여보고 싶다.
        - 코드가 보기 좋게 리팩토링 해보고 싶다.

- - -

- 진민주_T6171
    - 나는 내 학습목표를 달성하기 위해 무엇을 어떻게 했는가?
        - 학습 목표
            
            이번 대회를 통해서 공부하고자 했던 것은 다음과 같습니다.
            
            - 전반적인 Baseline 코드가 어떻게 구성되어 있는지 파악하기
            - 미션을 통해서 공부한 것들을 실제로 적용해보기
            - 대회에서의 주요한 문제점을 어떻게 해결해야할지 고민하고 직접 적용해보기
        - 어떻게 했는지
            - Baseline 코드를 계속 숙지하려고 했고 Mask, Gender, Age는 관계없는 문제라 각각 해결할 수 있도록 multi-label classfication으로 바꿔보았습니다. multi-label 코드에서도 각각 라벨의 모델을 다르게 해보거나 같게 해보기도 했습니다.
            - 성능이 크게 향상되지 않았지만 multi-label 코드에 미션 7에서의 ‘Statified KFold & Out-Of-Fold Ensemble with TTA’를 실제로 적용해보고 싶어서 적용해보기도 했습니다.
            - Mask, Gender, Age 중에 제일 예측하기 어려웠던 Age를 어떻게 해결해야할지 고민했습니다. 그래서 사진만으로 구분을 줄 수 있는 것은 주름이라고 생각해서 주름을 선명하게 하기 위한 RandomSharpness, Color Jitter 등을 사용해보았습니다. 이외에도 성능을 향상시키기 위해 다양한 기법을 적용시켜봤습니다.
    - 마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
        - 데이터 전처리를 처음부터 했어야 했는데 체계적으로 접근하지 못한 것 같습니다. 그리고 데이터 불균형을 해결하기 위해 해볼 수 있는 아이디어가 많이 부족했습니다.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 시도해보고 싶은 점은 무엇인가?
        - 22일에 스페셜 피어세션과 솔루션 발표를 통해서 데이터의 문제점을 어떻게 파악하고 데이터 불균형을 해결하기 위한 방법이 어떤 것이 있는지 많이 배웠습니다. 이외에도 공부가 많이 되었습니다. 아직 대회가 남았고 부스트캠프가 마지막이 아니니까 계속 공부하면서 다양한 솔루션을 얻고 많이 적용해보고 싶습니다.
