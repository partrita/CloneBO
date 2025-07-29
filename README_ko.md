# 진화하는 서열의 생성 모델을 이용한 항체의 베이즈 최적화

[Alan N Amin](https://alannawzadamin.github.io), \*[Nate Gruver](https://ngruver.github.io), \*[Yilun Kuang](https://yilunkuang.github.io), \*[Lily Li](https://yucenli.com), [Hunter Elliott](https://www.bighatbio.com/profiles/hunter-elliott), [Calvin McCarter](https://calvinmccarter.com), [Aniruddh Raghu](https://aniruddhraghu.com), [Peyton Greenside](https://www.bighatbio.com/profiles/peyton-greenside), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).
(\* 동일 기여)

[arXiv](https://arxiv.org/abs/2412.07763). Neurips 2024의 AIDrugX 워크숍에서 발표 (스포트라이트, 우수 포스터상 수상).

### 설명
여기서는 **클론 정보 기반 베이즈 최적화(Clone-informed Bayesian Optimization, CloneBO)**를 소개합니다. 이는 생성 모델에게 우리 면역 체계가 항체를 최적화하는 방법을 학습시켜 실험실에서 효율적으로 항체를 최적화하는 베이즈 최적화 절차입니다. 우리 면역 체계는 항체의 특정 부분 서열을 반복적으로 진화시켜 표적에 강하고 안정적으로 결합하도록 항체를 만듭니다. 이 과정에서 관련된 진화 서열 집합인 *클론 계열(clonal family)*이 생성됩니다. 우리는 수십만 개의 클론 계열로 대규모 언어 모델인 **CloneLM**을 훈련시키고, 이를 사용하여 인간 면역 체계 내에서 항체를 최적화할 가능성이 가장 높은 돌연변이를 가진 서열을 설계합니다. 우리는 뒤틀린 순차 몬테카를로(twisted sequential Monte Carlo) 절차를 사용하여 이전 측정값에 맞게 설계를 유도합니다. 우리는 CloneBO가 현실적인 *인 실리코(in silico)* 실험에서 이전 방법보다 훨씬 효율적으로 항체를 최적화하고, *인 비트로(in vitro)* 습식 실험실 실험에서 더 강력하고 안정적인 결합체를 설계함을 보여줍니다.

이 코드베이스는 항체의 반복적인 최적화를 위한 클론 정보 기반 베이즈 최적화(**CloneBO**)를 구현합니다.
그림 3a의 적합도 오라클과 그림 10의 CoV 오라클과 함께 CloneBO를 사용하는 코드가 포함되어 있습니다.

----

### 설치

Python 버전 ```3.12.0```에서 ```pip install .```을 실행하여 종속성을 설치하십시오.
[AbNumber](https://github.com/prihoda/AbNumber)도 설치해 주십시오.
마지막으로 임시 및 로깅 디렉토리 ```mkdir temp data```를 생성하십시오.

적합도 오라클을 사용하려면 [여기](https://huggingface.co/meta-llama/Llama-2-7b-hf)에서 얻을 수 있는 Llama 2 사용 권한이 필요합니다.
권한을 얻은 후 ```huggingface-cli login```을 사용하여 huggingface에 로그인해야 합니다.

### 사전 훈련된 모델

[HuggingFace에서 CloneLM 모델을 호스팅](https://huggingface.co/CloneBO/CloneLM)하고 있습니다.
다음과 같이 heavy chain 모델을 로드할 수 있습니다.
```
model = AutoModelForCausalLM.from_pretrained("CloneBO/CloneLM-Heavy")
tokenizer = AutoTokenizer.from_pretrained("CloneBO/CloneLM-Heavy")
```
아래 스크립트를 실행하면 CloneBO에서 사용할 CloneLM heavy가 자동으로 다운로드됩니다.
또한 [HuggingFace에서 그림 3a의 적합도 오라클을 호스팅](https://huggingface.co/CloneBO/OracleLM)하고 있습니다.

### 사용법

CloneBO의 기본 하이퍼파라미터는 ```configs/basic.cfg```에 저장되어 있습니다.
```python3 run_tsmc.py```를 실행하면 CloneBO가 실행되어 적합도 오라클을 최적화합니다.
이 코드는 결과를 ```CloneBO```라는 프로젝트의 ```wandb``` 실행으로 자동 전송합니다. 원하지 않으면 ```run.wandb```를 ```False```로 설정하십시오.
코드는 현재 80GB 메모리가 있는 GPU에서 실행되도록 최적화되어 있습니다.
더 작은 GPU에서 실행하려면 구성 파일에서 ```n_cond```를 줄이십시오.

노트북 ```run_clonebo.ipynb```에서도 최적화 절차를 실행할 수 있습니다.
노트북의 구성은 ```configs/short_run.cfg```입니다.

구성에서 ```oracle.name``` 인수는 작업을 제어합니다.
```pools.py```에서 사용 가능한 오라클을 볼 수 있으며, 목록은 다음과 같습니다.
* ```clone```
* ```SARSCoV1```
* ```SARSCoV2```
* ```rand_R```

여기서 R은 그림 12b의 실험을 복제하기 위해 적합도 오라클에 추가할 무작위 노이즈의 양을 설명하는 0과 1 사이의 숫자입니다.
covid 오라클의 모델 가중치와 코드는 MIT 라이선스에 따라 [RefineGNN 리포지토리](https://github.com/wengong-jin/RefineGNN)에서 가져왔습니다.
