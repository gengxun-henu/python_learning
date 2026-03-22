# GitHub Copilot Cloud Agent（云代理）运行完成后的操作指南

## 背景

当你在 GitHub 上使用 GitHub Copilot 编码代理（Cloud Agent）完成一次任务后，代理会把改动提交到一个**新分支**，并创建一个 **Pull Request（PR）**。  
你需要回到本地 VSCode，把这些改动同步下来，然后审查并决定是否接受。

---

## Checkout 和 Apply 是什么意思？

VSCode 的源代码管理或通知栏可能会弹出两个选项：

| 选项 | 含义 | 什么时候用 |
|------|------|-----------|
| **Checkout** | 切换到代理创建的新分支，直接在这个分支上查看和继续工作 | 想在本地完整查看、测试、或继续修改代理的代码时 |
| **Apply** | 把代理提出的改动直接应用到你当前的工作分支 | 只想快速接受改动，不想切换分支时 |

> **建议：优先选 Checkout。** 这样你可以先在本地查看所有改动、运行测试，确认无误后再合并到主分支。

---

## Cloud Agent 运行完成后的完整操作步骤

### 第 1 步：确认代理已完成

打开 GitHub 网页，找到代理创建的 **Pull Request**，确认状态显示完成（"Ready for review" 或有绿色勾）。

### 第 2 步：在 VSCode 中 Checkout 到代理分支

方式一：点击 VSCode 弹出的提示，选择 **Checkout**。  
方式二：用命令行手动切换：

```bash
git fetch origin
git checkout <代理创建的分支名>
```

分支名通常类似：`copilot/your-issue-description`

### 第 3 步：查看改动内容

```bash
git log --oneline -5          # 查看最近几条提交
git diff main                  # 对比和主分支的差异
```

或者在 VSCode 的 **Source Control** 面板里点击每个文件，查看改动详情。

### 第 4 步：本地运行并测试

```bash
python helloworld.py           # 运行对应文件确认正常
```

如果项目有测试文件，也跑一下：

```bash
python -m pytest               # 如果有 pytest 测试
```

### 第 5 步：确认没问题后，合并到主分支

**方式 A：在 GitHub 网页上合并（推荐）**

1. 打开代理创建的 Pull Request 页面
2. 点击 **"Merge pull request"** 按钮
3. 确认合并

**方式 B：用命令行合并**

```bash
git checkout main
git merge <代理创建的分支名>
git push
```

### 第 6 步：清理本地分支（可选）

合并完成后，可以删除代理创建的临时分支：

```bash
git branch -d <代理创建的分支名>
```

---

## 一句话总结工作流

```
Cloud Agent 完成
  → 点 Checkout 切到代理分支
  → 本地查看/测试改动
  → 没问题 → 在 GitHub PR 页面点 Merge
  → 本地 git pull 同步最新 main
```

---

## 常见问题

**Q：Checkout 后看不到改动怎么办？**  
先执行 `git fetch origin` 确保拉取了远端最新内容。

**Q：Apply 和 Checkout 有什么区别？**  
Apply 只把补丁应用到当前分支，不切换分支；Checkout 是完整切换到代理的新分支，更安全，推荐使用。

**Q：我合并后要不要删除代理分支？**  
删不删都行，GitHub 会在 PR 合并后提供一键删除分支的按钮，点一下即可，保持仓库整洁。
