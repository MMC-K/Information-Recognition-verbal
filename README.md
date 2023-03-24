

## Information Recognition Model - verbal
#### - To recognize a user's emotional state based on verbal information in the utterance


----------------------------------

## Sentence-level Sentiment Analysis

The code was fine-tuned using the KE-T5 pre-trained language model and sentiment label data sets from AI-HUB, the National Institute of Korean Language, and etc.

The sentence-level sentiment label data set was used, and the model and training information are as follows.


#### How to use

1) Run the command below to download the model parameters (weights)
```bash
    python get_weight.py
```

1) Run the command below to perform a sentiment analysis
```bash
    CUDA_VISIBLE_DEVICES='0' python test.py --model_type T5-mean-m
```


#### Pretrained Language Model
- KE-T5 base
(https://github.com/AIRC-KETI/ke-t5)




#### 모델 학습 환경
- OS : Ubuntu 20.04.4 LTS
- CPU : Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz
- GPU : TITAN-XP (12GB)

#### 하이퍼 파라미터
- optimizer : Adafactor
- learning rate :0.001
- batch size : 16
- max_length = 128
- epoch = 50
- decay_rate = -0.8
- clip_threshold = 1.0

#### 기타
- 모델 학습 파라미터(weight값)은 아래 디렉토리 저장하였습니다.

    (./model/T5-mean-m_KETI-AIR_ke-t5-base_default/weights/best.pth)
