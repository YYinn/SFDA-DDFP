This is the official code for "DDFP: Data-dependent Frequency Prompt for Source Free Domain Adaptation of Medical Image Segmentation".

# SFDA-DDFP
Domain adaptation addresses the challenge of model performance degradation caused by domain gaps. In the typical setup for unsupervised domain adaptation, labeled data from a source domain and unlabeled data from a target domain are used to train a target model. However, access to labeled source domain data, particularly in medical datasets, can be restricted due to privacy policies. As a result, research has increasingly shifted to source-free domain adaptation (SFDA), which requires only a pretrained model from the source domain and unlabeled data from the target domain data for adaptation. Existing SFDA methods often rely on domain-specific image style translation and self-supervision techniques to bridge the domain gap and train the target domain model. However, the quality of domain-specific style-translated images and pseudo-labels produced by these methods still leaves room for improvement. Moreover, training the entire model during adaptation can be inefficient under limited supervision. In this paper, we propose a novel SFDA framework to address these challenges. Specifically, to effectively mitigate the impact of domain gap in the initial training phase, we introduce preadaptation to generate a preadapted model, which serves as an initialization of target model and allows for the generation of high-quality enhanced pseudo-labels without introducing extra parameters. Additionally, we propose a data-dependent frequency prompt to more effectively translate target domain images into a source-like style. To further enhance adaptation, we employ a style-related layer fine-tuning strategy, specifically designed for SFDA, to train the target model using the prompted target domain images and pseudo-labels. Extensive experiments on cross-modality abdominal and cardiac SFDA segmentation tasks demonstrate that our proposed method outperforms existing state-of-the-art methods.



## 0. Data prepocess
### MMWHS
Follow the instruction of <a href="https://github.com/cchen-cc/SIFA#readme" title="SIFA">SIFA</a>.
### Abdominal 
Original site <a href="https://www.synapse.org/#!Synapse:syn3193805/wiki/217789" title="data">Synapse</a> . 
### Brate2018
Follow the instruction of <a href="https://github.com/icerain-alt/brats-unet.git" title="brats-unet">SIFA</a>.


## 1. Source Model 
### Source Model Training
Change parameters in ```configs/train_source_seg.yaml```.

```
python main_trainer_source.py --config_file configs/train_source_seg.yaml --gpu_id 0
```

### Source Model Testing
Change parameters in ```configs/test_source_seg.yaml```.

```
python main_trainer_source.py --config_file configs/test_source_seg.yaml --gpu_id 0
```

## 2. BN pre-adaptation
Change parameters in ```configs/train_target_adapt_bn.yaml.yaml```.

```
python target_adapte_trainer.py
```

## 3. Target model adaptation
Change parameters in ```configs/train_target_adapt_pmt.yaml```.

### Target Model Training
```
python main_trainer_sfda.py --config_file configs/train_target_adapt_pmt.yaml --gpu_id 0 

```
### Target Model Testing
Change parameters in ```configs/test_target_adapt_pmt.yaml```.
```
python main_trainer_sfda.py --config_file configs/test_target_adapt_pmt.yaml --gpu_id 0 
