# 使用SSH连接GitHub并上传代码到仓库的完整步骤
## 一、准备工作
### 1. 检查Git是否安装
首先确保本地已安装Git，打开终端，执行以下命令验证：
```bash
git --version
```

### 2. 确认GitHub账户
确保已注册GitHub账户

## 二、生成SSH密钥
SSH密钥用于本地与GitHub的安全认证，无需每次输入账号密码。

### 1. 检查现有SSH密钥
先查看本地是否已有SSH密钥，避免重复生成：
```bash
ls -la ~/.ssh
```
若显示`id_rsa`、`id_ed25519`等密钥文件，可直接跳至「三、配置SSH密钥到GitHub」；若无则继续生成新密钥。

### 2. 生成新SSH密钥
执行以下命令生成密钥（推荐使用更安全的`ed25519`算法，若系统不支持可改用`rsa`）：
```bash
# rsa算法
ssh-keygen -t rsa -b 4096 -C "473293308@qq.com"
```
参数说明：
- `-t`：指定密钥类型；
- `-C`：添加注释（建议填写GitHub注册邮箱）；
- `-b`：rsa算法需指定密钥长度（4096位更安全）。

执行后会出现以下提示：
```bash
Enter file in which to save the key (/Users/你的用户名/.ssh/id_ed25519): 
```
直接按回车使用默认路径即可（若需自定义路径，输入后回车）。

接着会提示设置密码（可选）：
```bash
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
```
- 若设置密码，每次使用SSH连接时需输入；
- 若无需密码，直接按两次回车即可。

生成成功后，本地`~/.ssh`目录下会出现两个文件：
- 私钥：`id_rsa`，切勿泄露；
- 公钥：`id_rsa.pub`，需上传到GitHub。

## 三、配置SSH密钥到GitHub
### 1. 复制SSH公钥内容
#### Windows（Git Bash）
```bash
cat ~/.ssh/id_ed25519.pub | clip
```

### 2. 将公钥添加到GitHub账户
1. 登录GitHub，点击右上角头像 → `Settings`（设置）；
2. 在左侧菜单中选择 `SSH and GPG keys`；
3. 点击右上角 `New SSH key`；
4. 填写信息：
   - `Title`：自定义名称（如「我的MacBook」）；
   - `Key type`：选择 `Authentication Key`；
   - `Key`：粘贴复制的公钥内容（确保完整，无多余空格/换行）；
5. 点击 `Add SSH key`，输入GitHub密码验证即可。

### 3. 测试SSH连接
回到终端，执行以下命令测试是否连接成功：
```bash
ssh -T git@github.com
```
首次连接会提示验证主机指纹，输入`yes`并回车：
```bash
The authenticity of host 'github.com (140.82.112.4)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```
若出现以下提示，说明连接成功：
```bash
Hi my567-321! You've successfully authenticated, but GitHub does not provide shell access.
```

## 四、上传代码到GitHub仓库
### 场景1：本地已有项目，上传到新的GitHub仓库
#### 步骤1：在GitHub创建空仓库
1. 登录GitHub，点击右上角 `+` → `New repository`；
2. 填写仓库信息：
   - `Repository name`：仓库名称（如`my-project`）；
   - `Description`：可选，仓库描述；
   - 选择 `Public`（公开）；
   - **不要勾选** `Initialize this repository with a README`（避免与本地代码冲突）；
3. 点击 `Create repository`。

创建后，GitHub会显示仓库的SSH地址（格式：`git@github.com:你的用户名/仓库名.git`），复制该地址。

#### 步骤2：本地项目关联并推送代码
1. 进入本地项目目录：
   ```bash
   cd C:\Users\bond\Desktop\cv\ex
   ```
2. 初始化Git仓库（若未初始化）：
   ```bash
   git init
   ```
3. 配置Git用户信息（需与GitHub账户一致）：
   ```bash
   git config --global user.name "my567-321"
   git config --global user.email "473293308@qq.com"
   ```
4. 添加所有文件到暂存区：
   ```bash
   git add .
   # 若只需添加指定文件，替换为：git add 文件名
   ```
5. 提交文件到本地仓库：
   ```bash
   git commit -m "init commit"
   ```
6. 关联远程GitHub仓库：
   ```bash
   git remote add origin git@github.com:my567-321/2025-hfut-cv-exp.git
   ```
7. 推送代码到GitHub：
   ```bash
   git push -u origin main
   ```
   `-u` 参数表示设置默认推送分支，后续可直接用 `git push`。

## 五、总结
1. 生成SSH密钥是一次性操作，配置后可长期使用；
2. 每次上传代码的核心流程：`git add` → `git commit` → `git push`；
3. 确保Git用户信息与GitHub账户一致，避免提交记录归属错误；
