### 注意

Windows 安装 pytorch3d 不能通过 conda 安装，只能从源代码构建



### 成功版本

显卡 GeForce GTX 1650 Ti

- CUDA 11.6
- pytorch  1.10.1
- cudatoolkit 11.3.1
- setuptools 59.6
- cub 直接用 CUDA 内置的
- pytorch3d 0.6.2 （直接从github release下的，未改动代码）



### 一些源码报错收集

```
\cast.h(1429): error: too few arguments for template template parameter "Tuple"
```

https://zhuanlan.zhihu.com/p/560277508



```
a member with an-class initiallizer must be const.
```

static 改成 static const



```
subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
```

可能是 cub 的问题，先把 ninja 去掉试试，如果 cub 部分一堆报错考虑换个版本。

cuda 新版都有 cub 集成，不用自己下



```
ImportError: cannot import name '_C' from 'pytorch3d'
```

先 import torch 看看



### 一些链接

- https://zhuanlan.zhihu.com/p/474383974
- https://blog.csdn.net/zzqkz20121221/article/details/121157357
- https://zhuanlan.zhihu.com/p/423096980
- https://github.com/facebookresearch/pytorch3d/issues/1227
- https://github.com/facebookresearch/pytorch3d/issues/1227 （CUB）
- https://zhuanlan.zhihu.com/p/460200485



