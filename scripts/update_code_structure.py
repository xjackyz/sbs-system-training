#!/usr/bin/env python
"""
代码结构更新脚本

此脚本用于自动更新项目中的导入路径、创建必要的__init__.py文件，
并确保新的目录结构能够正常工作。
"""

import os
import subprocess
import argparse
from pathlib import Path

def run_command(command):
    """运行shell命令并返回输出"""
    print(f"执行命令: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        return False, result.stderr
    return True, result.stdout

def ensure_init_files(directory):
    """确保所有Python包目录都有__init__.py文件"""
    print(f"检查并创建__init__.py文件在: {directory}")
    success_count = 0
    error_count = 0
    
    for root, dirs, files in os.walk(directory):
        # 跳过隐藏目录和__pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        # 检查是否有.py文件
        has_py_files = any(f.endswith('.py') for f in files)
        
        if has_py_files and '__init__.py' not in files:
            init_path = os.path.join(root, '__init__.py')
            try:
                with open(init_path, 'w') as f:
                    f.write('# 自动生成的__init__.py文件\n')
                print(f"创建: {init_path}")
                success_count += 1
            except Exception as e:
                print(f"创建{init_path}失败: {e}")
                error_count += 1
    
    print(f"成功创建 {success_count} 个__init__.py文件，失败 {error_count} 个")
    return success_count, error_count

def update_imports(directories, dry_run=False):
    """更新导入路径"""
    script_path = os.path.join(os.path.dirname(__file__), 'update_imports.py')
    
    # 构建命令
    cmd_parts = [
        f'python {script_path}',
        ' '.join(directories)
    ]
    
    if dry_run:
        cmd_parts.append('--dry-run')
    
    command = ' '.join(cmd_parts)
    success, output = run_command(command)
    
    if success:
        print("导入路径更新完成")
        print(output)
    else:
        print("导入路径更新失败")
    
    return success, output

def verify_project_structure():
    """验证项目结构是否正确"""
    # 检查关键目录是否存在
    key_directories = [
        'app', 'app/core', 'app/utils',
        'src', 'src/analysis', 'src/bot', 'src/image', 'src/notification', 'src/utils',
        'config', 'config/system', 'config/bot', 'config/model', 'config/training',
        'tests', 'tests/unit', 'tests/integration', 'tests/e2e'
    ]
    
    missing_dirs = []
    for directory in key_directories:
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"警告: 以下关键目录不存在: {', '.join(missing_dirs)}")
        return False
    
    print("项目结构验证通过")
    return True

def run_tests(test_path='tests'):
    """运行测试确保代码更改后仍然可以工作"""
    print(f"运行测试: {test_path}")
    command = f"python -m pytest {test_path} -v"
    success, output = run_command(command)
    
    if success:
        print("测试通过")
    else:
        print("测试失败")
    
    print(output)
    return success, output

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='更新项目代码结构')
    parser.add_argument('--dry-run', action='store_true',
                      help='只显示将要进行的更改，不实际修改文件')
    parser.add_argument('--skip-init', action='store_true',
                      help='跳过创建__init__.py文件')
    parser.add_argument('--skip-tests', action='store_true',
                      help='跳过运行测试')
    parser.add_argument('--test-path', type=str, default='tests',
                      help='指定要运行的测试路径（默认: tests）')
    parser.add_argument('--dirs', type=str, nargs='+',
                      default=['src', 'app', 'tests', 'scripts', 'sbs_bot'],
                      help='要处理的目录列表')
    
    args = parser.parse_args()
    
    # 显示欢迎信息
    print("======================================")
    print("SBS系统代码结构更新工具")
    print("======================================")
    print(f"处理目录: {', '.join(args.dirs)}")
    print(f"模式: {'预览' if args.dry_run else '实际更新'}")
    print("======================================")
    
    # 验证项目结构
    if not verify_project_structure():
        print("警告: 项目结构验证失败。继续可能会导致问题。")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    # 创建__init__.py文件
    if not args.skip_init:
        for directory in args.dirs:
            ensure_init_files(directory)
    
    # 更新导入路径
    update_imports(args.dirs, args.dry_run)
    
    # 运行测试
    if not args.skip_tests and not args.dry_run:
        run_tests(args.test_path)
    
    print("======================================")
    print("处理完成")
    if args.dry_run:
        print("这是预览模式，没有实际更改文件。")
        print("要实际应用更改，请去掉--dry-run参数。")
    print("======================================")

if __name__ == "__main__":
    main() 