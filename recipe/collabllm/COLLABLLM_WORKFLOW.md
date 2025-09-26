# CollabLLM Workflow Documentation

CollabLLM在VERL框架中的完整工作流程，包括交互机制、奖励计算和核心实现细节。

## 📊 概述

CollabLLM实现了一个多轮对话的强化学习框架，其中AI模型作为协作者与用户模拟器进行交互（真多轮，但本文不涉及tool use），通过多个评估指标获得奖励信号用于训练。

## 🏗️ 架构组件

### 1. 核心组件

- **CollabLLMAgentLoop**: 继承自`ToolAgentLoop`，管理模型生成和用户交互
- **CollabLLMInteraction**: 实现用户模拟器，生成真实的用户响应
- **CollabLLMRewardManager**: 处理多指标奖励计算
- **Metrics**: 多个评估指标模块（accuracy、pass_rate、token_amount等）

### 2. 数据流

```
Original Dataset → Process → Agent Loop → User Simulation → Reward Calculation → RL Training
```

## 🔄 详细工作流程

### Phase 1: 数据预处理

**文件**: `process_dataset.py`

```python
# 核心数据结构转换
extra_info = {
    "interaction_kwargs": {
        "name": "collabllm_interaction",
        "task_desc": args.task_desc,  # 从命令行传入
        "single_turn_prompt": row["prompt"]  # 原始问题
    }
}
```

**关键字段**:
- `agent_name`: 设置为 `"collabllm_agent"`
- `task_desc`: 任务描述，指导用户模拟器行为
- `single_turn_prompt`: 完整的用户请求或参考目标

### Phase 2: Agent Loop执行

**文件**: `collabllm_agent_loop.py`

#### 2.1 初始化与首次生成

```python
async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    # 1. 初始化交互组件
    interaction = self.interaction_map[interaction_name]
    await interaction.start_interaction(request_id, **interaction_kwargs)
    
    # 2. 生成模型初始响应
    await self._handle_pending_state(agent_data, sampling_params)
    status = await self._handle_generating_state(agent_data, sampling_params)
```

#### 2.2 多轮交互循环

```python
# 3. 收集多轮交互数据
num_repeats = self.config.actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts
interaction_requests = [deepcopy(agent_data) for _ in range(num_repeats)]

messages_lst = []
for _agent_data in interaction_requests:
    # 运行单个交互循环
    await self.run_agent_data_loop(_agent_data, sampling_params, AgentState.INTERACTING)
    messages_lst.append([Message(**msg) for msg in _agent_data.messages])
```

**关键特性**:
- 如果模型生成不完整(TERMINATED)，跳过后续交互避免奖励黑客攻击
- 每个交互请求独立运行，产生不同的对话轨迹
- 记录完整的对话历史用于奖励计算

### Phase 3: 用户模拟（gppt-4o-mini）

**文件**: `verl/interactions/collabllm_interation.py`

#### 3.1 用户模拟器提示模板（中文概括）

```python
USER_PROMPT_TEMPLATE = """You are role-playing as a human USER interacting with an AI collaborator...

## Guidelines:
- Stay in Character: 保持人类用户身份
- Minimize Effort: 避免过度详细，让AI询问澄清信息
- Knowledge Background: 反映用户的知识水平
- Occasionally Make Mistakes: 模拟真实用户的错误
- Mention Personal Preferences: 包含个人偏好和约束
- Goal-Oriented: 专注于主要目标

## Output Format:
{
    "current_answer": "简述AI当前解决方案",
    "thought": "作为用户的思考过程",
    "response": "用户的实际响应"
}
"""
```

#### 3.2 交互控制

```python
async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs):
    # 1. 解析对话历史
    chat_history = self._parse_messages(messages, strip_sys_prompt=True)
    
    # 2. 构建用户提示
    prompt = USER_PROMPT_TEMPLATE.format(
        task_desc=self.interaction_kwargs.get("task_desc", "general assistance task"),
        single_turn_prompt=self.interaction_kwargs["single_turn_prompt"],
        chat_history=chat_history,
        termination_signal=self.termination_signal
    )
    
    # 3. 生成用户响应
    response = await litellm.acompletion(model=self.user_model, messages=[{"role": "user", "content": prompt}])
    
    # 4. 检查终止条件
    should_terminate_sequence = self.termination_signal in response
    return should_terminate_sequence, response, reward, {}
```

**终止机制**:
- 用户可以发送`"[[TERMINATE CHAT]]"`终止对话
- 系统自动检测并结束交互循环

### Phase 4: 奖励计算

**文件**: `verl/workers/reward_manager/collabllm.py`

#### 4.1 多指标评估

```python
async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False):
    # 1. 准备数据
    message_lst = data.non_tensor_batch["messages"]
    num_repeat_rollouts = len(message_lst[0]["messages"])
    
    # 2. 并行计算多个指标
    tasks = [
        self.compute_score(
            flattened_data_sources[i],
            flattened_messages[i], 
            flattened_ground_truths[i],
            flattened_extra_infos[i],
            self.metrics,
            **self.llm_judge_kwargs
        ) for i in range(len(flattened_data_sources))
    ]
    score_dicts = await asyncio.gather(*tasks)
```

#### 4.2 加权聚合

```python
# 3. 按指标聚合分数
scores_by_metrics = {
    metric: torch.stack([score_dict[metric] for score_dict in score_dicts])
    .view(num_repeat_rollouts, -1)
    .sum(dim=0)
    for metric in self.metrics
}

# 4. 应用指标权重
weighted_scores_by_metrics = {
    metric: torch.clamp(
        scores_by_metrics[metric] * self.metric_weights[metric] / num_repeat_rollouts,
        min=-1.0, max=1.0
    ) for metric in self.metrics
}

# 5. 合并为最终分数
scores = torch.stack([weighted_scores_by_metrics[metric] for metric in self.metrics]).sum(dim=0)
```

### Phase 5: 指标计算详情

**目录**: `recipe/collabllm/metrics/`

#### 5.1 准确性指标 (`accuracy.py`)

```python
async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # 使用LLM判断器评估答案准确性
    response = await litellm.acompletion(
        model=judge_model,
        messages=[{"role": "user", "content": accuracy_prompt}]
    )
    # 解析判断结果
    return float(accuracy_score)
```

#### 5.2 代码通过率 (`pass_rate.py`)

```python
async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # 1. 提取代码
    code = extract_code_from_conversation(messages)
    
    # 2. 运行测试用例
    test_results = await run_code_tests(code, test_cases)
    
    # 3. 计算通过率
    pass_rate = sum(test_results) / len(test_results)
    return pass_rate
```

#### 5.3 令牌效率 (`token_amount.py`)

```python
def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    prompt = extra_info["prompt"]
    future_conv = messages[len(prompt):]
    
    # 计算令牌惩罚
    total_tokens = sum(len(m.content.split()) for m in future_conv)
    return total_tokens  # 作为惩罚项
```

#### 5.4 交互性指标 (`interactivity.py`)

```python
async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # 评估对话的交互质量
    # - 轮次数量
    # - 信息交换效率
    # - 澄清问题的合理性
    return interaction_quality_score
```

