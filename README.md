# Team 4 Final Project

# Members

# Data
should make a folder `data` to keep all `.csv` files

# Environment
* python==3.9.15g
* transformers==4.25.1
* torch==1.13.0
* torchvision==0.14.0
* ml-metrics==0.1.4
* pandas==1.5.2
* numpy==1.23.5

##### tips
* if you encounter error 
    ```bash  
    × python setup.py egg_info did not run successfully.  
    ``` 
    when installing `ml-metrics` please refer [\[this workaround\]](https://github.com/pfnet-research/xfeat/issues/9)
* `matplotlib` can't show Mandarin characters problem: 
    list all fonts can be used by matplotlib
    ```python
    from matplotlib import font_manager
    font_set = {f.name for f in font_manager.fontManager.ttflist}
    for f in font_set:
    if 'TW' in f or 'CN' in f:
        print(f)
    ```
    ```bash
    AR PL UKai CN
    AR PL UMing CN
    ````
    pick one ie. 'AR PL UKai CN' as the font to be used. Then do the setting
    ```python
    mpl.rcParams['font.sans-serif'] = ['AR PL UKai CN']
    mpl.rcParams['font.family'] = 'AR PL UKai CN'    

    ``` 
    please refer [\[Matplotlib 中文字體亂碼問題\]](https://dwye.dev/post/matplotlib-font/) 
