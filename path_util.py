import os
import subprocess

def get_gitee_raw_url(local_file_path):
    abs_path = os.path.abspath(local_file_path)
    dirname = os.path.dirname(abs_path)

    try:
        # 1. 获取 Git 根目录
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=dirname).decode().strip()

        # 2. 获取远程 URL (处理 git@gitee.com... 和 .git 后缀)
        remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], cwd=dirname).decode().strip()
        
        # 标准化 URL 格式
        if remote_url.startswith('git@'):
            remote_url = remote_url.replace(':', '/').replace('git@', 'https://')
        if remote_url.endswith('.git'):
            remote_url = remote_url[:-4]

        # 3. 获取分支
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=dirname).decode().strip()
        if not branch: # 处理游离指针情况
            branch = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dirname).decode().strip()

        # 4. 获取相对路径 (并确保 Windows 下路径分隔符为 /)
        relative_path = os.path.relpath(abs_path, repo_root).replace('\\', '/')

        # 5. 拼接 Raw URL (关键点：这里使用 /raw/)
        # Gitee 格式: domain/user/repo/raw/branch/path
        final_url = f"{remote_url}/raw/{branch}/{relative_path}"
        
        return final_url

    except Exception as e:
        return f"Error: {e}"

# --- 假设你本地有这个文件 ---
# 本地路径: /Users/yourname/projects/docs/zh-cn/figures/1.png
# 运行结果将是: https://gitee.com/openharmony/docs/raw/master/zh-cn/figures/1.png

# --- 测试 ---
# 替换成你本地的真实文件路径
my_local_file = "../docs/zh-cn/glossary.md" 
print(get_gitee_raw_url(my_local_file))