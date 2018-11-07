# vggNetChineseVerticalCodeOCR
中文验证码生成，tensorflow框架中使用vggNet边生成边训练。验证码有噪线，中文为常用汉字，字体为simsum。

---
### enviroment  
opencv2  
tensorflow-gpu  
math  
numpy  

### 运行方法：
训练：  
> python train_captcha.py
预测：  
> python predict_captcha.py 目标图片地址

