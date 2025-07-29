# Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences

[한국어](./README_ko.md)

[Alan N Amin](https://alannawzadamin.github.io), \*[Nate Gruver](https://ngruver.github.io), \*[Yilun Kuang](https://yilunkuang.github.io), \*[Lily Li](https://yucenli.com), [Hunter Elliott](https://www.bighatbio.com/profiles/hunter-elliott), [Calvin McCarter](https://calvinmccarter.com), [Aniruddh Raghu](https://aniruddhraghu.com), [Peyton Greenside](https://www.bighatbio.com/profiles/peyton-greenside), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).
(\* equal contribution)

[arXiv](https://arxiv.org/abs/2412.07763). Presented at AIDrugX workshop at Neurips 2024 (won spotlight, outstanding poster awards).

### Description
Here we introduce **Clone-informed Bayesian Optimization (CloneBO)**, a Bayesian optimization procedure that efficiently optimizes antibodies in the lab by teaching a generative model how our immune system optimizes antibodies. Our immune system makes antibodies by iteratively evolving specific portions of their sequences to bind their target strongly and stably, resulting in a set of related, evolving sequences known as a *clonal family*. We train a large language model, **CloneLM**, on hundreds of thousands of clonal families and use it to design sequences with mutations that are most likely to optimize an antibody within the human immune system. We guide our designs to fit previous measurements using a twisted sequential Monte Carlo procedure. We show that CloneBO optimizes antibodies substantially more efficiently than previous methods in realistic *in silico* experiments and designs stronger and more stable binders in *in vitro* wet lab experiments. 


This codebase implements Clone-informed Bayesian Optimization (**CloneBO**) for iterative optimization of antibodies.
We include code to use CloneBO with the fitness oracle in Fig. 3a and the CoV oracles in Fig. 10.

----

### Installation

Install dependencies by running ```pip install .``` with Python version ```3.12.0```.
Please also install [AbNumber](https://github.com/prihoda/AbNumber).
Finally create temporary and logging directories ```mkdir temp data```.

To use the fitness oracle, you need permission to use Llama 2 which can be obtained [here](https://huggingface.co/meta-llama/Llama-2-7b-hf).
After obtaining permission you must log into huggingface using ```huggingface-cli login```.

### Pretrained models

We are hosting the [CloneLM models on HuggingFace](https://huggingface.co/CloneBO/CloneLM).
You can load the heavy chain model with 
```
model = AutoModelForCausalLM.from_pretrained("CloneBO/CloneLM-Heavy")
tokenizer = AutoTokenizer.from_pretrained("CloneBO/CloneLM-Heavy")
```
Running the scripts below automatically download CloneLM heavy to use in CloneBO.
We are also hosting the [fitness oracle from Fig. 3a on HuggingFace](https://huggingface.co/CloneBO/OracleLM).

### Usage

The default hyperparameters of CloneBO are stored in ```configs/basic.cfg```.
Running ```python3 run_tsmc.py``` will run CloneBO to optimize the fitness oracle.
This code will automatically send the results to a ```wandb``` run in a project called ```CloneBO```; set ```run.wandb``` to ```False``` if this is undesired.
The code currently is optimized to run on a GPU with 80 GB of memory;
to run on a smaller GPU, decrease ```n_cond``` in the config file.

You may also run the optimization procedure in the notebook ```run_clonebo.ipynb```.
The config for the notebook is ```configs/short_run.cfg```.

In the config, the argument ```oracle.name``` controls the task.
You can have a look at the available oracles in ```pools.py``` but listed, they are:
* ```clone```
* ```SARSCoV1```
* ```SARSCoV2```
* ```rand_R```

Where R is a number between 0 and 1 describing how much random noise to add to the fitness oracle to replicate the experiment in Fig. 12b.
The model weights and code for the covid oracle are taken from the [RefineGNN repo](https://github.com/wengong-jin/RefineGNN) under the MIT licence.
