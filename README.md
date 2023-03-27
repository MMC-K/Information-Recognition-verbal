

## Information Recognition Model - verbal
#### - To recognize a user's emotional state based on verbal information in the utterance


----------------------------------

## Sentence-level Sentiment Analysis

The model fine-tuned a pre-trained language model, KE-T5, using sentiment label datasets from AI-HUB, the National Language Institute, and others.

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




#### model train environment
- OS : Ubuntu 20.04.4 LTS
- CPU : Intel(R) Xeon(R) W-2245 CPU @ 3.90GHz
- GPU : TITAN-XP (12GB)

#### Hyper parameter
- optimizer : Adafactor
- learning rate :0.001
- batch size : 16
- max_length = 128
- epoch = 50
- decay_rate = -0.8
- clip_threshold = 1.0

#### Etc
- The model training parameters (weight values) are saved to the directory below after downloading.

    (./model/T5-mean-m_KETI-AIR_ke-t5-base_default/weights/best.pth)
