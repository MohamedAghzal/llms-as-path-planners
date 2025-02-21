# Evaluation Large Language Models as Path Planners

This repository contains the code for Path Planning from Natural Language (PPNL) dataset for testing the ability of Large Language Models (LLMs) to conduct path planning, which was proposed in the following papers:

- [''Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning''](https://arxiv.org/abs/2310.03249): Testing LLMs ability to conduct path planning and generalize across different environment configurations and number of goals. [[Code](./ppnl-spatial-temporal-reasoning)]


- [''Look Further Ahead: Testing the Limits of GPT-4 in Path Planning''](https://arxiv.org/abs/2406.12000): Testing LLMs ability to conduct path planning using different representations (natural language, code, grids) in complex geometric shapes and their ability to generalize to longer horizon planning scenarios. [[Code](./gpt-4-path-planning)] 

# Citation

If you find this work useful for your own research please cite the following papers:



```
@inproceedings{aghzal2024can,
  title={Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning},
  author={Aghzal, Mohamed and Plaku, Erion and Yao, Ziyu},
  booktitle={ICLR 2024 Workshop on Large Language Model (LLM) Agents}
}

@inproceedings{aghzal2024look,
  title={Look Further Ahead: Testing the Limits of GPT-4 in Path Planning},
  author={Aghzal, Mohamed and Plaku, Erion and Yao, Ziyu},
  booktitle={2024 IEEE 20th International Conference on Automation Science and Engineering},
  year={2024}
}
```