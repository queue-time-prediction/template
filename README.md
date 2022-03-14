# 代码编写要求

python的类为双驼峰：

MyClass ✔️

myClass ❌

My_Class ❌

除此以外（文件、包、函数、变量）均为小写下划线：

my_name ✔️

myName ❌

myname ❌

编写api时提供doc注释（方法体下直接输入长字符串可自动生成）：

```python
def hello(self, text):
    '''
    输出hello world
    :param text: 文本
    '''
    print(f'Hello {text}!')
```

