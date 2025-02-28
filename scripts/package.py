import os
import shutil
import tarfile
from datetime import datetime

def create_package():
    # 创建临时目录
    temp_dir = "temp_package"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # 需要复制的目录和文件
    dirs_to_copy = [
        "src",
        "config",
        "scripts",
        "models",
        "data"
    ]

    files_to_copy = [
        "setup.py",
        "requirements.txt",
        ".env.example",
        "README.md"
    ]

    # 复制目录
    for dir_name in dirs_to_copy:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(temp_dir, dir_name))

    # 复制文件
    for file_name in files_to_copy:
        if os.path.exists(file_name):
            shutil.copy2(file_name, os.path.join(temp_dir, file_name))

    # 创建tar文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tar_filename = f"sbs_system_{timestamp}.tar.gz"
    
    with tarfile.open(tar_filename, "w:gz") as tar:
        tar.add(temp_dir, arcname="sbs_system")

    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    print(f"打包完成: {tar_filename}")
    print("\n需要传输到新服务器的文件:")
    print(f"1. {tar_filename}")
    print("2. 模型文件 (如果没有包含在models目录中)")
    print("\n在新服务器上的安装步骤:")
    print("1. 解压文件：tar -xzf " + tar_filename)
    print("2. cd sbs_system")
    print("3. pip install -e .")
    print("4. 复制模型文件到models目录")
    print("5. 配置.env文件")
    print("6. python scripts/train.py")

if __name__ == "__main__":
    create_package() 