# Virtual-Step-and-Auxiliary-Gradients

Code for paper "Improving Transferability of Adversarial Example with Virtual Step and Auxiliary Gradients", IJCAI 2022.

## Introduction

We propose to improve the transferability of adversarial examples through the use of a virtual step and auxiliary gradients,
Extensitive experiments on ImageNet show that the adversarial examples crafted by our method can effectively transfer to
different networks. For single-model attacks, our method outperfoms the state-of-the-art baselines, improving the success
rates by a large margin of 12%~28%.


## Dependencies

+ TensorFlow ≥ 2.3.1 with GPU support
+ imageio ≥ 2.9.0
+ numpy ≥ 1.18.5
+ pandas ≥ 1.1.3


## Instructions

+ `single_model_attack_raw_fgsm.py`, the script to run single-model baseline attacks. By using the on-off flags`using_translation_invariant = False` and `input_diversity(x, opened=False)`, you can swith to "I-FGSM", "DI<sup>2</sup>-FGSM" and "TI<sup>2</sup>-FGSM".
+ `single_model_attack_va_fgsm.py`, the script to run single-model attacks with virtual step and auxiliary gradients. By using the on-off flags`using_translation_invariant = False` and `input_diversity(x, opened=False)`, you can swith to "VA-I-FGSM", "VA-DI<sup>2</sup>-FGSM" and "VA-TI<sup>2</sup>-FGSM".
+ `ensemble_based_attack_raw_fgsm.py`, the script to run ensemble-based baseline attacks. By using the on-off flags`using_translation_invariant = False` and `input_diversity(x, opened=False)`, you can swith to "I-FGSM", "DI<sup>2</sup>-FGSM" and "TI<sup>2</sup>-FGSM".
+ `ensemble_based_attack_va_fgsm.py`, the script to run ensemble-based attacks with virtual step and auxiliary gradients. By using the on-off flags`using_translation_invariant = False` and `input_diversity(x, opened=False)`, you can swith to "VA-I-FGSM", "VA-DI<sup>2</sup>-FGSM" and "VA-TI<sup>2</sup>-FGSM".
+ `utils.py`, some necessary utils.
+ `data`, the directory for storing the experimental data.
