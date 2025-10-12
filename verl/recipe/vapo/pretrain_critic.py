

# 中间的r怎么写， 

# Step 0: 环境与模型准备
policy = load_sft_policy(); policy.eval()
value_model = ValueNetwork(); optimizer = Adam(value_model.parameters(), lr=1e-4)

# Step 1: 轨迹采集
def collect_trajectories(policy, env, num_episodes):
    trajectories = []
    for _ in range(num_episodes):
        obs = env.reset()
        states, rewards = [], []
        done = False
        while not done:
            action = policy.act(obs)
            next_obs, reward, done = env.step(action)
            states.append(obs); rewards.append(reward)
            obs = next_obs
        trajectories.append((states, rewards))
    return trajectories

# Step 2: 预训练循环
for epoch in range(max_epochs):
    trajs = collect_trajectories(policy, env, batch_size)
    all_states, all_returns = [], []
    for states, rewards in trajs:
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + G  # gamma=1
            returns.insert(0, G)
        all_states.extend(states)
        all_returns.extend(returns)

    preds = value_model(torch.tensor(all_states))
    returns = torch.tensor(all_returns)
    vf_loss = F.mse_loss(preds, returns)
    optimizer.zero_grad(); vf_loss.backward(); optimizer.step()

    # 计算 explained_variance
    error_var = torch.var(returns - preds)
    target_var = torch.var(returns)
    explained_vf = 1 - error_var / target_var

    print(f"Epoch {epoch}: vf_loss={vf_loss.item():.4f}, explained_vf={explained_vf.item():.4f}")
    if vf_loss < vf_loss_thresh and explained_vf > ev_thresh:
        break

# Step 3: 保存预训练价值模型
torch.save(value_model.state_dict(), "value_pretrained.ckpt")

# 正例的再强化

# 提前训练一个critic，？ 合理吗 和后面是否

# 不对 直接抄写一个trainer 