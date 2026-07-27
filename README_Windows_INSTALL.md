# gsplat源码版本地编译安装说明
由于官方库未测试windows版本的gsplat安装，所以本版本对源码进行修改，匹配windows系统环境。
## 前提：
1. 安装CUDA Toolkit 12.9

   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

2. 安装VisualStudio 2022

## 安装方式
1. 在build_gsplat.bat中修改Line15，改为本地的对应虚拟环境中的python路径，以及当前gsplat工程所处的路径
2. 检查build_gsplat.bat所有执行命令路径正确后，在终端执行build_gsplat.bat
3. 等待一小时左右编译安装完毕


