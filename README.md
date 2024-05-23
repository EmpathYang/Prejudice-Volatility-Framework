# Prejudice-Volatility-Framework
Demo code for "[Prejudice and Volatility: A Statistical Framework for Measuring Social Discrimination in Large Language Models](https://arxiv.org/abs/2402.15481)".

![](figures/framework.png)

LLMs are revolutionizing society fast! 🚀 Ever wondered if your LLM assistant could be biased? Could it affect your mood, important decisions, job prospects, legal matters, healthcare, or even your kid’s future education? 😱 Need a flexible framework to measure this risk?

Check out our new paper: the [Prejudice-Volatility Framework](https://arxiv.org/abs/2402.15481) (PVF)! 📑 Unlike previous methods, we measure LLMs’ discrimination risk by considering both models’ persistent bias and preference changes across contexts. Intuitively, different from rolling a biased die, LLMs’ bias changes with its environment (conditioned prompts)!

Our findings? 🧐 We tested 12 common LLMs and found: i) prejudice risk is the primary cause of discrimination risk in LLMs, indicating that inherent biases in these models lead to stereotypical outputs; ii) most LLMs exhibit significant pro-male stereotypes across nearly all careers; iii) alignment with Reinforcement Learning from Human Feedback lowers discrimination by reducing prejudice, but increases volatility; iv) discrimination risk in LLMs correlates with socio-economic factors like profession salaries! 📊

## Replication
### Environment Setup

Create environment:

```bash
conda create -n pcf python=3.11.5
conda activate pcf
```

Install pytorch and python packages:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd Prejudice-Caprice-Framework
pip install -r requirements.txt
```

### Experiments

Evaluate MaskedLM **bert**:

```bash
cd script
bash bert.sh
```

Evaluate CausalLM **gpt2**:

```bash
cd script
bash gpt.sh
```