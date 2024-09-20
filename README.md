# Cheaper O1

![demo.png](https://github.com/BirchKwok/cheaper_o1/blob/main/demo.png)
## English

### Introduction
Cheaper O1 is a simple Chain-of-Thought (COT) model product designed as an alternative to OpenAI's O1 model. This project utilizes the glm-4-flash model from Zhipu AI to provide a cost-effective solution for complex problem-solving tasks.

### Features
- Implements a multi-step reasoning process
- Utilizes web search for information gathering
- Provides self-assessment and judgment of answers

### Usage
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your API key:
   Create a file named `api_key` in the project root and paste your Zhipu AI API key.

3. Run the Jupyter notebook:
   ```
   jupyter notebook demo.ipynb
   ```

4. In the notebook, you can use the `solve_problem` function as shown in the example below:

   ```python
   from cheaper_o1 import solve_problem

   thought_chain = solve_problem("9.11和9.8哪个大")
   ```

   This will display the reasoning process and the final answer.

### Code Structure
- `cheaper_o1.py`: Main script containing the core logic
- `webviewer.py`: Module for web search functionality
- `demo.ipynb`: Jupyter notebook demonstrating the usage

### Note
This project is designed for educational and experimental purposes. It may not be suitable for production environments without further development and testing.

## 中文

### 简介
Cheaper O1 是一个简单的思维链（Chain-of-Thought，COT）模型产品，旨在替代 OpenAI 的 O1 模型。本项目使用智谱AI的 glm-4-flash 模型，为复杂问题解决任务提供了一个经济实惠的解决方案。

### 特性
- 实现多步推理过程
- 利用网络搜索收集信息
- 提供答案的自我评估和判断

### 使用方法
1. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

2. 设置 API 密钥：
   在项目根目录创建名为 `api_key` 的文件，并粘贴您的智谱AI API 密钥。

3. 运行 Jupyter notebook：
   ```
   jupyter notebook demo.ipynb
   ```

4. 在 notebook 中，您可以使用 `solve_problem` 函数，如下例所示：

   ```python
   from cheaper_o1 import solve_problem

   thought_chain = solve_problem("9.11和9.8哪个大")
   ```

   这将显示推理过程和最终答案。

### 代码结构
- `cheaper_o1.py`：包含核心逻辑的主脚本
- `webviewer.py`：网络搜索功能模块
- `demo.ipynb`：演示使用方法的 Jupyter 笔记本

### 注意
本项目设计用于教育和实验目的。如果没有进一步的开发和测试，可能不适合用于生产环境.

