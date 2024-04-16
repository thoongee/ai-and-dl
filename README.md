## 1. Implement LeNet-5

### **Number of model parameters of LeNet-5 : 61706**

**모델 구조**

- C1: Convolutional layer with 6 feature maps and a 5x5 kernel
- S2: Subsampling/Max Pooling layer
- C3: Convolutional layer with 16 feature maps, each connected to subsets of the S2 feature maps, with a 5x5 kernel
- S4: Subsampling/Max Pooling layer
- C5: Convolutional layer with 120 feature maps with a 5x5 kernel (as a fully connected layer)
- F6: Fully connected layer with 84 units
- OUTPUT: A Gaussian connection layer which is typically replaced by a fully connected layer with 10 units for digit classification

**파라미터 수**

- C1: 6 feature maps×(5×5 filters+1 bias)=156
- S2: 0
- C3: 16 feature maps×(6×5×5 filters+1 bias)=2416
- S4: 0
- C5: 120 feature maps×(16×5×5 filters+1 bias)=48120
- F6: 84 units×(120 inputs+1 bias)=10164
- OUTPUT: 10 units×(84 inputs+1 bias)=850
- 총 파라미터 수 =  156+2416+48120+10164+850= 61706

## 2. Implement CustomMLP

### **Number of model parameters of CustomMLP : 60713**

**모델 설명**

- LeNet-5는 컨볼루션 레이어와 풀링 레이어를 포함한 CNN(Convolutional Neural Network)입니다. 하지만 CustomMLP는 컨볼루션 레이어를 사용하지 않고 fully connected layer만을 사용하는 구조입니다.  LeNet-5와 같은 복잡한 아키텍처를 가진 CNN과 단순한 MLP의 성능을 비교하여, CNN이 이미지와 같은 데이터에 대해 왜 우수한 성능을 발휘하는지 이해하고자 이러한 구조로 작성하였습니다.
- 이 모델은 입력 데이터의 공간적 구조를 고려하지 않기 때문에, 컨볼루션 신경망에 비해 시각적 패턴을 인식하는 데 덜 효과적일 수 있습니다.
- 각 은닉층 뒤에는 **`ReLU`** (Rectified Linear Unit) 활성화 함수가 사용됩니다. ReLU는 음수 입력에 대해 0을 출력하고, 양수 입력에 대해서는 입력 값을 그대로 출력합니다. 이는 모델이 비선형 문제를 해결하는 데 도움을 줍니다.

**파라미터 수**

- 1st hidden layer: 784×70+70=54950 (input node, 28x28 image)
- 2nd hidden layer: 70×55+55=3905
- 3rd hidden layer: 55×28+28 =1568
- Output: 28×10+10=290
- 총 파라미터 수 : 60713

## 3. Statistics

### 3-1. Average loss values and accuracy at the end of each epoch

| Model | Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
| --- | --- | --- | --- | --- | --- |
| LeNet5 | 1 | 0.2902 | 90.47% | 0.0653 | 97.93% |
|  | 2 | 0.0631 | 98.02% | 0.0553 | 98.26% |
|  | 3 | 0.0456 | 98.60% | 0.0425 | 98.61% |
|  | 4 | 0.0353 | 98.92% | 0.0314 | 98.96% |
|  | 5 | 0.0291 | 99.07% | 0.0357 | 98.84% |
|  | 6 | 0.0253 | 99.21% | 0.0392 | 98.78% |
|  | 7 | 0.0202 | 99.34% | 0.0371 | 98.80% |
|  | 8 | 0.0164 | 99.48% | 0.0318 | 99.02% |
|  | 9 | 0.0146 | 99.55% | 0.0403 | 98.76% |
|  | 10 | 0.0137 | 99.55% | 0.0368 | 99.03% |
| CustomMLP | 1 | 0.4089 | 87.33% | 0.1556 | 95.25% |
|  | 2 | 0.1370 | 95.87% | 0.1176 | 96.36% |
|  | 3 | 0.0997 | 96.94% | 0.0975 | 97.01% |
|  | 4 | 0.0775 | 97.58% | 0.1164 | 96.38% |
|  | 5 | 0.0643 | 97.97% | 0.0970 | 97.09% |
|  | 6 | 0.0512 | 98.39% | 0.0980 | 97.12% |
|  | 7 | 0.0460 | 98.52% | 0.0964 | 97.60% |
|  | 8 | 0.0384 | 98.79% | 0.0924 | 97.44% |
|  | 9 | 0.0339 | 98.88% | 0.1127 | 97.01% |
|  | 10 | 0.0304 | 99.01% | 0.0931 | 97.63% |

### 3-2. Loss and accuracy curves for training and test datasets (LeNet-5, CustomMLP)

<p align="center"><img src="https://github.com/thoongee/mnist-classification/assets/94193480/fdc43824-b446-426c-8cc3-4dab9673f6a7" width="40%">

## 4. Compare the predictive performances of LeNet-5 and CustomMLP

### **Train Performance**

- **LeNet-5**: 첫 에포크에서 90.47%의 정확도로 시작하여, 10번째 에포크에서 99.55%의 정확도에 도달합니다. 훈련 손실도 0.2902에서 0.0137로 급격히 감소합니다.
- **CustomMLP**: 87.33%의 초기 정확도에서 시작하여 10번째 에포크에 99.01%까지 증가합니다. 훈련 손실은 0.4089에서 0.0304로 감소하며, 이는 LeNet-5보다 초기 손실이 높고, 감소율도 덜 급격합니다.

### **Test Performance**

- **LeNet-5**: 테스트 정확도는 97.93%에서 시작하여 10번째 에포크에서 99.03%로 최고점을 찍습니다. 테스트 손실은 0.0653에서 0.0368로 변동성이 있으나 전반적으로 낮은 편입니다.
- **CustomMLP**: 테스트에서 95.25%의 정확도로 시작하여 10번째 에포크에 97.63%까지 증가합니다. 테스트 손실은 0.1556에서 시작하여 0.0931로 줄어들지만, LeNet-5보다 높은 손실 값을 보입니다.

### 정리

- **LeNet-5**는 일관되게 높은 훈련 및 테스트 정확도를 보이며, 특히 테스트 데이터에서의 정확도와 손실 값이 더 우수합니다. 이는 LeNet-5의 합성곱 구조가 이미지 데이터의 공간적 특성을 보다 효과적으로 학습하기 때문으로 볼 수 있습니다.
- **CustomMLP**는 비교적 단순한 완전 연결 구조로 구성되어 있으며, 이는 피처의 공간적 연관성을 무시하고 데이터를 일렬로 평탄화하여 처리합니다. 이로 인해 이미지 데이터에 대한 학습 효율이 떨어질 수 있습니다.

## 5. Regularization techniques to improve LeNet-5 model

1. **Batch normalization**
    - 배치 정규화는 각 convolution layer 또는 완전 fully connected layer 후, 활성화 함수 전에 적용됩니다. 이는 레이어의 출력값을 정규화하여 다음 레이어로 전달하기 전에 안정적인 분포를 가지도록 합니다.
2. **Dropout**
    - 일반적으로 드롭아웃은 0.5의 비율로 적용합니다.

### 5-1. Verify that regularization techniques actually help improve the performance

| Model | Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
| --- | --- | --- | --- | --- | --- |
| LeNet5 | 1 | 0.3203 | 89.66% | 0.0708 | 97.85% |
|  | 2 | 0.0625 | 98.03% | 0.0416 | 98.77% |
|  | 3 | 0.0433 | 98.64% | 0.0373 | 98.90% |
|  | 4 | 0.0329 | 98.94% | 0.0355 | 98.82% |
|  | 5 | 0.0267 | 99.12% | 0.0368 | 98.86% |
|  | 6 | 0.0215 | 99.31% | 0.0338 | 98.88% |
|  | 7 | 0.0194 | 99.39% | 0.0305 | 99.02% |
|  | 8 | 0.0141 | 99.53% | 0.0438 | 98.73% |
|  | 9 | 0.0128 | 99.60% | 0.0334 | 99.06% |
|  | 10 | 0.0103 | 99.66% | 0.0318 | 99.12% |
| RegularizedLenet5 | 1 | 0.2872 | 91.33% | 0.0468 | 98.57% |
|  | 2 | 0.0763 | 97.87% | 0.0506 | 98.51% |
|  | 3 | 0.0568 | 98.43% | 0.0390 | 98.78% |
|  | 4 | 0.0456 | 98.69% | 0.0362 | 98.91% |
|  | 5 | 0.0409 | 98.84% | 0.0279 | 99.06% |
|  | 6 | 0.0347 | 99.02% | 0.0289 | 99.08% |
|  | 7 | 0.0291 | 99.18% | 0.0344 | 99.00% |
|  | 8 | 0.0282 | 99.19% | 0.0255 | 99.23% |
|  | 9 | 0.0227 | 99.33% | 0.0281 | 99.17% |
|  | 10 | 0.0216 | 99.40% | 0.0264 | 99.21% |

<p align="center"><img src="https://github.com/thoongee/mnist-classification/assets/94193480/7a6d9223-4692-4b82-8402-562a03fe1392" width="40%">

### **초기 성능**

- **LeNet5**는 첫 번째 에포크에서 훈련 손실이 0.3203, 훈련 정확도가 89.66%입니다. 테스트 손실과 정확도는 각각 0.0708과 97.85%입니다.
- **RegularizedLeNet5**는 첫 번째 에포크에서 훈련 손실이 0.2872, 훈련 정확도가 91.33%로, 기본 모델보다 훈련 손실이 낮고 정확도가 높습니다. 테스트 손실과 정확도는 0.0468과 98.57%로, 이 또한 기본 모델보다 향상된 수치입니다.

### **Epoch 진행에 따른 성능**

- **LeNet5**는 10번째 에포크까지 훈련 손실이 0.0103으로 감소하고, 훈련 정확도가 99.66%로 상승합니다. 테스트 손실과 정확도는 각각 0.0318과 99.12%입니다.
- **RegularizedLeNet5**는 10번째 에포크에서 훈련 손실이 0.0216으로 기본 모델보다 다소 높지만, 훈련 정확도가 99.40%로 비슷합니다. 테스트 손실과 정확도는 각각 0.0264와 99.21%로, 기본 모델의 테스트 성능을 약간 상회합니다.

### **결론**

- **정규화 효과**: 정규화된 LeNet5 모델은 초기 에포크부터 더 높은 테스트 정확도와 낮은 테스트 손실을 보여주며, 특히 초기 학습 속도와 일반화 능력에서 눈에 띄는 개선을 보여줍니다. 이는 배치 정규화가 모델의 내부 공변량 변화를 감소시키고, 드롭아웃이 과적합을 방지하는 데 도움을 주었다고 해석할 수 있습니다.
- **일관된 성능 유지**: RegularizedLeNet5는 훈련 과정 중 일관되게 높은 성능을 유지하며, 특히 테스트 데이터에서의 성능이 더 안정적입니다. 이는 정규화 기법이 모델이 새로운 데이터에 대해 더 잘 일반화하도록 도와주기 때문입니다.
